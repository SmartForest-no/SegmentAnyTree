import torchnet as tnt
from typing import NamedTuple, Dict, Any, List, Tuple
import torch
import logging
from torch_geometric.nn import knn
import numpy as np
from torch_scatter import scatter_add
from collections import OrderedDict, defaultdict
from torch_geometric.nn.unpool import knn_interpolate
from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.models.model_interface import TrackerInterface
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.models.panoptic.structures import PanopticResults, PanopticLabels
from torch_points3d.core.data_transform import SaveOriginalPosId, SaveLocalOriginalPosId
from torch_points_kernels import instance_iou
from .box_detection.ap import voc_ap
import time
import os
from os.path import exists, join
from torch_points3d.models.panoptic.ply import read_ply, write_ply
time_for_blockMerging=0
log = logging.getLogger(__name__)
block_count=0

class _Instance(NamedTuple):
    classname: str
    score: float
    indices: np.array  # type: ignore
    scan_id: int

    def iou(self, other: "_Instance") -> float:
        assert self.scan_id == other.scan_id
        intersection = float(len(np.intersect1d(other.indices, self.indices)))
        return intersection / float(len(other.indices) + len(self.indices) - intersection)

    def find_best_match(self, others: List["_Instance"]) -> Tuple[float, int]:
        ioumax = -np.inf
        best_match = -1
        for i, other in enumerate(others):
            iou = self.iou(other)
            if iou > ioumax:
                ioumax = iou
                best_match = i
        return ioumax, best_match


class InstanceAPMeter:
    def __init__(self):
        self._pred_clusters = defaultdict(list)  # {classname: List[_Instance]}
        self._gt_clusters = defaultdict(lambda: defaultdict(list))  # {classname:{scan_id: List[_Instance]}

    def add(self, pred_clusters: List[_Instance], gt_clusters: List[_Instance]):
        for instance in pred_clusters:
            self._pred_clusters[instance.classname].append(instance)
        for instance in gt_clusters:
            self._gt_clusters[instance.classname][instance.scan_id].append(instance)

    def _eval_cls(self, classname, iou_threshold):
        preds = self._pred_clusters.get(classname, [])
        allgts = self._gt_clusters.get(classname, {})
        visited = {scan_id: len(gt) * [False] for scan_id, gt in allgts.items()}
        ngt = 0
        for gts in allgts.values():
            ngt += len(gts)

        # Start with most confident first
        preds.sort(key=lambda x: x.score, reverse=True)
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        for p, pred in enumerate(preds):
            scan_id = pred.scan_id
            gts = allgts.get(scan_id, [])
            if len(gts) == 0:
                fp[p] = 1
                continue

            # Find best macth in ground truth
            ioumax, best_match = pred.find_best_match(gts)

            if ioumax < iou_threshold:
                fp[p] = 1
                continue

            if visited[scan_id][best_match]:
                fp[p] = 1
            else:
                visited[scan_id][best_match] = True
                tp[p] = 1

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(ngt)

        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec)
        return rec, prec, ap

    def eval(self, iou_threshold, processes=1):
        rec = {}
        prec = {}
        ap = {}
        for classname in self._gt_clusters.keys():
            rec[classname], prec[classname], ap[classname] = self._eval_cls(classname, iou_threshold)

        for i, classname in enumerate(self._gt_clusters.keys()):
            if classname not in self._pred_clusters:
                rec[classname] = 0
                prec[classname] = 0
                ap[classname] = 0

        return rec, prec, ap


class PanopticTracker(SegmentationTracker):
    """ Class that provides tracking of semantic segmentation as well as
    instance segmentation """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metric_func = {**self._metric_func, "pos": max, "neg": min, "map": max, "Prec": max, "Rec": max}

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._test_area = None
        self._full_vote_miou = None
        self._vote_miou = None
        self._full_confusion = None
        self._iou_per_class = {}
        self._pos = tnt.meter.AverageValueMeter()
        self._neg = tnt.meter.AverageValueMeter()
        self._acc_meter = tnt.meter.AverageValueMeter()
        self._ap_meter = InstanceAPMeter()
        self._scan_id_offset = 0
        self._rec: Dict[str, float] = {}
        self._prec: Dict[str, float] = {}
        self._ap: Dict[str, float] = {}
        self._iou_per_class = {}

    def track(
        self,
        model: TrackerInterface,
        full_res=False, 
        data=None,
        iou_threshold=0.5, #0.25,
        track_instances=True,
        min_cluster_points=10,
        **kwargs
    ):
        """ Track metrics for panoptic segmentation
        """
        self._iou_threshold = iou_threshold
        BaseTracker.track(self, model)
        outputs: PanopticResults = model.get_output()
        labels: PanopticLabels = model.get_labels()

        # Track semantic
        super()._compute_metrics(outputs.semantic_logits, labels.y)

        if not data:
            return
        assert data.pos.dim() == 2, "Only supports packed batches"

        # Object accuracy
        clusters, valid_c_idx = PanopticTracker._extract_clusters(outputs, min_cluster_points)
        if not clusters:
            return

        predicted_labels = outputs.semantic_logits.max(1)[1]
        if torch.max(labels.instance_labels)>0:
            tp, fp, acc = self._compute_acc(
                clusters, predicted_labels, labels, data.batch, labels.num_instances, iou_threshold
            )
            self._pos.add(tp)
            self._neg.add(fp)
            self._acc_meter.add(acc)

        # Track instances for AP
        if track_instances:
            pred_clusters = self._pred_instances_per_scan(
                clusters, predicted_labels, outputs.cluster_scores, data.batch, self._scan_id_offset
            )
            gt_clusters = self._gt_instances_per_scan(
                labels.instance_labels, labels.y, data.batch, self._scan_id_offset
            )
            self._ap_meter.add(pred_clusters, gt_clusters)
            self._scan_id_offset += data.batch[-1].item() + 1
            
        # Train mode or low res, nothing special to do
        if self._stage == "train" or not full_res:
            return

        # Test mode, compute votes in order to get full res predictions
        if self._test_area is None:
            self._test_area = self._dataset.test_data.clone()
            if self._test_area.y is None:
                raise ValueError("It seems that the test area data does not have labels (attribute y).")
            self._test_area.prediction_count = torch.zeros(self._test_area.y.shape[0], dtype=torch.int)
            self._test_area.votes = torch.zeros((self._test_area.y.shape[0], self._num_classes), dtype=torch.float)
            self._test_area.ins_pre = -1*torch.ones(self._test_area.y.shape[0], dtype=torch.int)
            self._test_area.max_instance = 0
            #self._test_area.ori_pos = torch.zeros((self._test_area.pos.shape), dtype=torch.float)
            self._test_area.to(model.device)

        # Gather origin ids and check that it fits with the test set
        inputs = data if data is not None else model.get_input()
        if inputs[SaveOriginalPosId.KEY] is None:
            raise ValueError("The inputs given to the model do not have a %s attribute." % SaveOriginalPosId.KEY)

        originids = inputs[SaveOriginalPosId.KEY]
        global block_count
        #localoriginids = inputs[SaveLocalOriginalPosId.KEY]
        original_input_ids = self._dataset.test_data_spheres[block_count].origin_id
        #print(originids)
        if originids.dim() == 2:
            originids = originids.flatten()
        if originids.max() >= self._test_area.pos.shape[0]:
            raise ValueError("Origin ids are larger than the number of points in the original point cloud.")

        # Set predictions
        self._test_area.votes[originids] += outputs.semantic_logits
        self._test_area.prediction_count[originids] += 1
        #self._test_area.ori_pos[originids] = data.coords
        
        c_scores = [outputs.cluster_scores[i].cpu().numpy() for i in valid_c_idx]
        
        curins_pre = self.get_cur_ins_pre_label(clusters, np.array(c_scores), predicted_labels.cpu().numpy())
        
        #block merging
        global time_for_blockMerging
        T1 = time.perf_counter()
        self._test_area.ins_pre, self._test_area.max_instance = self.block_merging(original_input_ids.cpu().numpy(), originids.cpu().numpy(), curins_pre, self._test_area.ins_pre.cpu().numpy(), self._test_area.max_instance, model.get_opt_mergeTh(), predicted_labels)
        T2 = time.perf_counter()
        #print('time for block merging:%sms' % ((T2 - T1)*1000))
        time_for_blockMerging += T2 - T1
        #print('total time for block merging of embeds:%sms' % ((time_for_blockMerging)*1000))
        #log.info("total time for block merging of embeds:{}ms".format((time_for_blockMerging)*1000))
 
    def get_cur_ins_pre_label(self, clusters, cluster_scores, predicted_semlabels):
        cur_ins_pre_label = -1*np.ones_like(predicted_semlabels)
        idx=np.argsort(cluster_scores)
        for i,j in enumerate(idx):
            cur_ins_pre_label[clusters[j].cpu().numpy()] = i
        return cur_ins_pre_label
        
    def block_merging(self, originids, origin_sub_ids, pre_sub_ins, all_pre_ins, max_instance, th_merge, predicted_sem_labels):       
        #output interme results
        global block_count
        
        if not os.path.exists("viz"):
                os.mkdir("viz")
        val_name = join("viz", "block_sub_"+str(block_count))
        write_ply(val_name,
                    [self._test_area.pos[origin_sub_ids].detach().cpu().numpy(), 
                    pre_sub_ins.astype('int32'),
                    predicted_sem_labels.detach().cpu().numpy().astype('int32'),
                    ],
                    ['x', 'y', 'z', 'preins_label', 'presem_label'])
        
        has_prediction = pre_sub_ins != -1
        assign_index  = knn(self._test_area.pos[origin_sub_ids[has_prediction]], self._test_area.pos[originids], k=1)

        y_idx, x_idx = assign_index
        pre_ins = pre_sub_ins[has_prediction][x_idx.detach().cpu().numpy()]
        #has_prediction = full_ins_pred != -1
        val_name = join("viz", "block_"+str(block_count))
        write_ply(val_name,
                    [self._test_area.pos[originids].detach().cpu().numpy(), 
                    pre_ins.astype('int32'),
                    ],
                    ['x', 'y', 'z', 'preins_label'])
        block_count=block_count+1


        t_num_clusters = np.max(pre_ins)+1
        #print(np.unique(pre_ins))
        #print(t_num_clusters)
        #print("t_num_clusters: {}".format(t_num_clusters))
        #print("max_instance: {}".format(max_instance))
        #print("th_merge: {}".format(th_merge))
        idx = np.argwhere(all_pre_ins[originids] != -1)  #has label
        idx2 = np.argwhere(all_pre_ins[originids] == -1) #no label
        
        #all points have no label
        if len(idx)==0: #t_num_clusters>0:    
            mask_valid = pre_ins != -1                     
            all_pre_ins[originids[mask_valid]] = pre_ins[mask_valid] + max_instance                     
            max_instance = max_instance + t_num_clusters
            #print("test")
            #return torch.from_numpy(all_pre_ins), max_instance
        #all points have labels
        elif len(idx2)==0: 
            return  torch.from_numpy(all_pre_ins), max_instance
        #part of points have labels
        else:                       
            #merge by iou
            new_label = pre_ins.reshape(-1)  
                
            for ii_idx in range(t_num_clusters):   
                new_label_ii_idx = originids[np.argwhere(new_label==ii_idx).reshape(-1)]
                    
                new_has_old_idx = new_label_ii_idx[np.argwhere(all_pre_ins[new_label_ii_idx]!=-1)]  #new prediction already has old label
                new_not_old_idx = new_label_ii_idx[np.argwhere(all_pre_ins[new_label_ii_idx]==-1)] #new prediction has no old label  
                #print(new_has_old_idx)
                #print(len(new_label_ii_idx))  
                #print(len(new_has_old_idx))     
                #print(len(new_not_old_idx))  
                if len(new_has_old_idx)==0:
                    all_pre_ins[new_not_old_idx] = max_instance+1
                    max_instance = max_instance+1
                elif len(new_not_old_idx)==0: 
                    continue
                else:
                    old_labels_ii = all_pre_ins[new_has_old_idx]
                    un = np.unique(old_labels_ii)
                    #print(un)
                    max_iou_ii = 0
                    max_iou_ii_oldlabel = 0
                    for ig, g in enumerate(un):
                        idx_old_all = originids[np.argwhere(all_pre_ins[originids]==g).reshape(-1)]
                        union_label_idx = np.union1d(idx_old_all, new_label_ii_idx)
                        inter_label_idx = np.intersect1d(idx_old_all, new_label_ii_idx)
                        #print(inter_label_idx.size)
                        iou = float(inter_label_idx.size) / float(union_label_idx.size)
                        #print(iou)
                        if iou > max_iou_ii:
                            max_iou_ii = iou
                            max_iou_ii_oldlabel = g
                            
                    if max_iou_ii > th_merge:
                        all_pre_ins[new_not_old_idx] = max_iou_ii_oldlabel
                    else:
                        all_pre_ins[new_not_old_idx] = max_instance+1
                        max_instance = max_instance+1
                        
        return torch.from_numpy(all_pre_ins), max_instance

    def finalise(self, full_res=False, vote_miou=True, ply_output="", track_instances=True, **kwargs):
        per_class_iou = self._confusion_matrix.get_intersection_union_per_class()[0]
        self._iou_per_class = {k: v for k, v in enumerate(per_class_iou)}
        
        if vote_miou and self._test_area:
            # Complete for points that have a prediction
            self._test_area = self._test_area.to("cpu")
            c = ConfusionMatrix(self._num_classes)
            has_prediction = self._test_area.prediction_count > 0
            gt = self._test_area.y[has_prediction].numpy()
            pred = torch.argmax(self._test_area.votes[has_prediction], 1).numpy()
            gt_effect = gt >= 0
            c.count_predicted_batch(gt[gt_effect], pred[gt_effect])
            self._vote_miou = c.get_average_intersection_union() * 100

        if full_res:
            self._compute_full_miou()

        if ply_output:
            has_prediction = self._test_area.prediction_count > 0
            #semantic prediction with color for subsampled cloud
            '''self._dataset.to_ply(
                self._test_area.pos[has_prediction].cpu(),
                torch.argmax(self._test_area.votes[has_prediction], 1).cpu().numpy(),
                ply_output,
            )'''
            
            #self._test_area = self._test_area.to("cpu")
            full_pred = knn_interpolate(
            self._test_area.votes[has_prediction], self._test_area.pos[has_prediction], self._test_area.pos, k=1,
            )
            #semantic prediction with color for full cloud
            '''self._dataset.to_ply(
                self._test_area.pos,
                torch.argmax(full_pred, 1).numpy(),
                "vote1regularfull.ply",
            )'''
            #semantic prediction and GT label full cloud (for final evaluation)
            '''self._dataset.to_eval_ply(
                self._test_area.pos,
                torch.argmax(full_pred, 1).numpy(), #[0, ..]
                self._test_area.y,   #[-1, ...]
                "Semantic_results_forEval.ply",
            )'''
            #instance
            has_prediction = self._test_area.ins_pre != -1
            #full_ins_pred_embed = knn_interpolate(
            #torch.reshape(self._test_area.ins_pre_embed[has_prediction], (-1,1)), self._test_area.pos[has_prediction], self._test_area.pos, k=1,
            #)
            
            self._dataset.to_ins_ply(
                self._test_area.pos[has_prediction].cpu(),
                self._test_area.ins_pre[has_prediction].cpu().numpy(),
                "Instance_subsample.ply",
            )
            
            assign_index  = knn(self._test_area.pos[has_prediction], self._test_area.pos, k=1)

            #assign_index2  = nearest(self._test_area.pos, self._test_area.pos[has_prediction])

            y_idx, x_idx = assign_index
            full_ins_pred = self._test_area.ins_pre[has_prediction][x_idx]

            full_ins_pred = torch.reshape(full_ins_pred, (-1,))
            #instance prediction and GT label full cloud (for final evaluation)
            
            idx_in_cur = [idx for idx, l in enumerate(torch.argmax(full_pred, 1).numpy()) if l in self._dataset.stuff_classes]
            idx_in_cur = np.array(idx_in_cur)
            idx_in_cur.astype(int)
            
            full_ins_pred[idx_in_cur] = -1

            '''self._dataset.to_eval_ply(
                self._test_area.pos,
                full_ins_pred_embed.numpy(),  #[-1, ...]
                self._test_area.instance_labels,  #[0, ..]
                "Instance_Embed_results_forEval.ply",
            )
            self._dataset.to_eval_ply(
                self._test_area.pos,
                full_ins_pred_offset.numpy(),
                self._test_area.instance_labels,
                "Instance_Offset_results_forEval.ply",
            )'''
            
            self._dataset.generate_separate_room(
                self._test_area.pos,
                torch.argmax(full_pred, 1).numpy(),
                full_ins_pred.numpy(),
                full_ins_pred.numpy(),
                )
            
            self._dataset.final_eval()
            #instance prediction with color for "things"
            things_idx = full_ins_pred != -1
            self._dataset.to_ins_ply(
                self._test_area.pos[things_idx],
                full_ins_pred[things_idx].numpy(),
                "Instance_results_withColor.ply",
            )

        if not track_instances:
            return

        rec, prec, ap = self._ap_meter.eval(self._iou_threshold)
        self._ap = OrderedDict(sorted(ap.items()))
        self._rec = OrderedDict({})
        self._prec = OrderedDict({})
        for key, val in sorted(rec.items()):
            try:
                value = val[-1]
            except TypeError:
                value = val
            self._rec[key] = value
        for key, val in sorted(prec.items()):
            try:
                value = val[-1]
            except TypeError:
                value = val
            self._prec[key] = value
            

    @staticmethod
    def _compute_acc(clusters, predicted_labels, labels, batch, num_instances, iou_threshold):
        """ Computes the ratio of True positives, False positives and accuracy
        """
        iou_values, gt_ids = instance_iou(clusters, labels.instance_labels, batch).max(1)
        gt_ids += 1
        instance_offsets = torch.cat((torch.tensor([0]).to(num_instances.device), num_instances.cumsum(-1)))
        tp = 0
        fp = 0
        for i, iou in enumerate(iou_values):
            # Too low iou, no match in ground truth
            if iou < iou_threshold:
                fp += 1
                continue

            # Check that semantic is correct
            sample_idx = batch[clusters[i][0]]
            sample_mask = batch == sample_idx
            instance_offset = instance_offsets[sample_idx]
            gt_mask = labels.instance_labels[sample_mask] == (gt_ids[i] - instance_offset)
            gt_classes = labels.y[sample_mask][torch.nonzero(gt_mask, as_tuple=False)]
            gt_classes, counts = torch.unique(gt_classes, return_counts=True)
            gt_class = gt_classes[counts.max(-1)[1]]
            pred_class = predicted_labels[clusters[i][0]]
            if gt_class == pred_class:
                tp += 1
            else:
                fp += 1
        acc = tp / len(clusters)
        tp = tp / torch.sum(labels.num_instances).cpu().item()
        fp = fp / torch.sum(labels.num_instances).cpu().item()
        return tp, fp, acc

    def _compute_full_miou(self):
        if self._full_vote_miou is not None:
            return

        has_prediction = self._test_area.prediction_count > 0
        log.info(
            "Computing full res mIoU, we have predictions for %.2f%% of the points."
            % (torch.sum(has_prediction) / (1.0 * has_prediction.shape[0]) * 100)
        )

        self._test_area = self._test_area.to("cpu")

        # Full res interpolation
        full_pred = knn_interpolate(
            self._test_area.votes[has_prediction], self._test_area.pos[has_prediction], self._test_area.pos, k=1,
        )

        # Full res pred
        self._full_confusion = ConfusionMatrix(self._num_classes)
        gt_effect = self._test_area.y >= 0
        self._full_confusion.count_predicted_batch(self._test_area.y[gt_effect].numpy(), torch.argmax(full_pred, 1)[gt_effect].numpy())
        self._full_vote_miou = self._full_confusion.get_average_intersection_union() * 100

    @staticmethod
    def _extract_clusters(outputs, min_cluster_points):
        valid_cluster_idx = outputs.get_instances(min_cluster_points=min_cluster_points)
        clusters = [outputs.clusters[i] for i in valid_cluster_idx]
        return clusters, valid_cluster_idx

    @staticmethod
    def _pred_instances_per_scan(clusters, predicted_labels, scores, batch, scan_id_offset):
        # Get sample index offset
        ones = torch.ones_like(batch)
        sample_sizes = torch.cat((torch.tensor([0]).to(batch.device), scatter_add(ones, batch)))
        offsets = sample_sizes.cumsum(dim=-1).cpu().numpy()

        # Build instance objects
        instances = []
        for i, cl in enumerate(clusters):
            sample_idx = batch[cl[0]].item()
            scan_id = sample_idx + scan_id_offset
            indices = cl.cpu().numpy() - offsets[sample_idx]
            instances.append(
                _Instance(
                    classname=predicted_labels[cl[0]].item(), score=scores[i].item(), indices=indices, scan_id=scan_id
                )
            )
        return instances

    @staticmethod
    def _gt_instances_per_scan(instance_labels, gt_labels, batch, scan_id_offset):
        batch_size = batch[-1] + 1
        instances = []
        for b in range(batch_size):
            sample_mask = batch == b
            instances_in_sample = instance_labels[sample_mask]
            gt_labels_sample = gt_labels[sample_mask]
            num_instances = torch.max(instances_in_sample)
            scan_id = b + scan_id_offset
            for i in range(num_instances):
                instance_indices = torch.where(instances_in_sample == i + 1)[0].cpu().numpy()
                instances.append(
                    _Instance(
                        classname=gt_labels_sample[instance_indices[0]].item(),
                        score=-1,
                        indices=instance_indices,
                        scan_id=scan_id,
                    )
                )
        return instances

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        metrics["{}_pos".format(self._stage)] = meter_value(self._pos)
        metrics["{}_neg".format(self._stage)] = meter_value(self._neg)
        metrics["{}_Iacc".format(self._stage)] = meter_value(self._acc_meter)

        if self._has_instance_data:
            mAP = sum(self._ap.values()) / len(self._ap)
            metrics["{}_map".format(self._stage)] = mAP
            mPrec = sum(self._prec.values()) / len(self._prec)
            metrics["{}_mPrec".format(self._stage)] = mPrec
            mRec= sum(self._rec.values()) / len(self._rec)
            metrics["{}_mRec".format(self._stage)] = mRec

        if verbose:
            metrics["{}_iou_per_class".format(self._stage)] = self._iou_per_class

        if verbose and self._has_instance_data:
            metrics["{}_class_prec".format(self._stage)] = self._dict_to_str(self._prec)
            metrics["{}_class_rec".format(self._stage)] = self._dict_to_str(self._rec)
            metrics["{}_class_ap".format(self._stage)] = self._dict_to_str(self._ap)
        return metrics

    @property
    def _has_instance_data(self):
        return len(self._rec)
