from typing import NamedTuple, Dict, Any, List, Tuple
import torchnet as tnt
import logging
import torch
import numpy as np
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.nn import knn
from torch_scatter import scatter_add
from collections import OrderedDict, defaultdict

from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.models import model_interface
from torch_points3d.models.panoptic.structures import PanopticResults, PanopticLabels
from torch_points_kernels import instance_iou
from .box_detection.ap import voc_ap
import time
time_for_blockMerging_offset=0
time_for_blockMerging_embed=0
log = logging.getLogger(__name__)
block_count=0
class _Instance(NamedTuple):
    classname: str
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
        #preds.sort(key=lambda x: x.score, reverse=True)
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

class MyPanopticTracker(SegmentationTracker):
    """ Class that provides tracking of semantic segmentation as well as
    instance segmentation """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metric_func = {**self._metric_func, "pos_embed": max, "neg_embed": min, "pos_offset": max, "neg_offset": max, "map_embed": max, "map_offset": max}
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._test_area = None
        self._full_vote_miou = None
        self._vote_miou = None
        self._full_confusion = None
        self._iou_per_class = {}
        
        self._pos_embed = tnt.meter.AverageValueMeter()
        self._neg_embed = tnt.meter.AverageValueMeter()
        self._pos_offset = tnt.meter.AverageValueMeter()
        self._neg_offset = tnt.meter.AverageValueMeter()
        self._acc_meter_embed = tnt.meter.AverageValueMeter()
        self._acc_meter_offset = tnt.meter.AverageValueMeter()
        self._ap_meter_embed = InstanceAPMeter()
        self._ap_meter_offset = InstanceAPMeter()
        self._scan_id_offset = 0
        self._scan_id_offset2 = 0
        self._rec: Dict[str, float] = {}
        self._ap: Dict[str, float] = {}
        self._rec2: Dict[str, float] = {}
        self._ap2: Dict[str, float] = {}

    def track(self, model: model_interface.TrackerInterface, full_res=False, data=None, iou_threshold=0.25, track_instances=True, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        #super().track(model)
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
        #print(outputs.embed_clusters)
        #print(self._stage)
        if not outputs.embed_clusters:
            return
        predicted_labels = outputs.semantic_logits.max(1)[1]
        #print(torch.max(labels.instance_labels))
        if torch.max(labels.instance_labels)>0:
            tp, fp, acc = self._compute_acc(
                outputs.embed_clusters, predicted_labels, labels, data.batch, labels.num_instances, iou_threshold
            )
            self._pos_embed.add(tp)
            self._neg_embed.add(fp)
            self._acc_meter_embed.add(acc)
        
        #if not outputs.offset_clusters:
        #    return
        
            tp2, fp2, acc2 = self._compute_acc(
                outputs.offset_clusters, predicted_labels, labels, data.batch, labels.num_instances, iou_threshold
            )
            self._pos_offset.add(tp2)
            self._neg_offset.add(fp2)
            self._acc_meter_offset.add(acc2)
        
        # Track instances for AP
        if track_instances:
            pred_clusters_embed = self._pred_instances_per_scan(
                outputs.embed_clusters, predicted_labels, data.batch, self._scan_id_offset
            )
            pred_clusters_offset = self._pred_instances_per_scan(
                outputs.offset_clusters, predicted_labels, data.batch, self._scan_id_offset2
            )
            gt_clusters = self._gt_instances_per_scan(
                labels.instance_labels, labels.y, data.batch, self._scan_id_offset
            )
            self._ap_meter_embed.add(pred_clusters_embed, gt_clusters)
            self._scan_id_offset += data.batch[-1].item() + 1
            self._ap_meter_offset.add(pred_clusters_offset, gt_clusters)
            self._scan_id_offset2 += data.batch[-1].item() + 1
        
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
            self._test_area.ins_pre_embed = -1*torch.ones(self._test_area.y.shape[0], dtype=torch.int)
            self._test_area.ins_pre_offset = -1*torch.ones(self._test_area.y.shape[0], dtype=torch.int)
            self._test_area.max_instance_embed = 0
            self._test_area.max_instance_offset = 0
            self._test_area.to(model.device)

        # Gather origin ids and check that it fits with the test set
        inputs = data if data is not None else model.get_input()
        if inputs[SaveOriginalPosId.KEY] is None:
            raise ValueError("The inputs given to the model do not have a %s attribute." % SaveOriginalPosId.KEY)

        originids = inputs[SaveOriginalPosId.KEY]
        global block_count
        #print(originids)
        original_input_ids = self._dataset.test_data_spheres[block_count].origin_id
        if originids.dim() == 2:
            originids = originids.flatten()
        if originids.max() >= self._test_area.pos.shape[0]:
            raise ValueError("Origin ids are larger than the number of points in the original point cloud.")

        # Set predictions
        self._test_area.votes[originids] += outputs.semantic_logits
        self._test_area.prediction_count[originids] += 1
        #block merging for offsets and embedding features
        global time_for_blockMerging_embed
        T1 = time.perf_counter()
        self._test_area.ins_pre_embed, self._test_area.max_instance_embed = self.block_merging(original_input_ids.cpu().numpy(), originids.cpu().numpy(), outputs.embed_pre.cpu().numpy(), self._test_area.ins_pre_embed.cpu().numpy(), self._test_area.max_instance_embed, model.get_opt_mergeTh())
        T2 = time.perf_counter()
        print('time for block merging of embeds:%sms' % ((T2 - T1)*1000))
        time_for_blockMerging_embed += T2 - T1
        print('total time for block merging of embeds:%sms' % ((time_for_blockMerging_embed)*1000))
        log.info("total time for block merging of embeds:{}ms".format((time_for_blockMerging_embed)*1000))
        
        global time_for_blockMerging_offset
        T1 = time.perf_counter()
        self._test_area.ins_pre_offset, self._test_area.max_instance_offset = self.block_merging(original_input_ids.cpu().numpy(), originids.cpu().numpy(), outputs.offset_pre.cpu().numpy(), self._test_area.ins_pre_offset.cpu().numpy(), self._test_area.max_instance_offset,  model.get_opt_mergeTh())
        T2 = time.perf_counter()
        print('time for block merging of offsets:%sms' % ((T2 - T1)*1000))
        time_for_blockMerging_offset += T2 - T1
        print('total time for block merging of offsets:%sms' % ((time_for_blockMerging_offset)*1000))
        log.info("total time for block merging of offsets:{}ms".format((time_for_blockMerging_offset)*1000))
        block_count = block_count+1
      	#return num_clusters, torch.from_numpy(labels)

    def block_merging(self, originids, origin_sub_ids, pre_sub_ins, all_pre_ins, max_instance, th_merge):
        
        assign_index  = knn(self._test_area.pos[origin_sub_ids], self._test_area.pos[originids], k=1)

        y_idx, x_idx = assign_index
        pre_ins = pre_sub_ins[x_idx.detach().cpu().numpy()]
        
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
        self._iou_per_class = {self._dataset.INV_OBJECT_LABEL[k]: v for k, v in enumerate(per_class_iou)}

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
            has_prediction = self._test_area.ins_pre_embed != -1
            #full_ins_pred_embed = knn_interpolate(
            #torch.reshape(self._test_area.ins_pre_embed[has_prediction], (-1,1)), self._test_area.pos[has_prediction], self._test_area.pos, k=1,
            #)
            assign_index  = knn(self._test_area.pos[has_prediction], self._test_area.pos, k=1)

            #assign_index2  = nearest(self._test_area.pos, self._test_area.pos[has_prediction])

            y_idx, x_idx = assign_index
            full_ins_pred_embed = self._test_area.ins_pre_embed[has_prediction][x_idx]

            #full_ins_pred_embed2 = self._test_area.ins_pre_embed[has_prediction][assign_index2]
            
            has_prediction = self._test_area.ins_pre_offset != -1
            #full_ins_pred_offset = knn_interpolate(
            #torch.reshape(self._test_area.ins_pre_offset[has_prediction], (-1,1)), self._test_area.pos[has_prediction], self._test_area.pos, k=1,
            #)
            assign_index  = knn(self._test_area.pos[has_prediction], self._test_area.pos, k=1)
            y_idx, x_idx = assign_index
            full_ins_pred_offset = self._test_area.ins_pre_offset[has_prediction][x_idx]
            full_ins_pred_embed = torch.reshape(full_ins_pred_embed, (-1,))
            full_ins_pred_offset = torch.reshape(full_ins_pred_offset, (-1,))
            #instance prediction and GT label full cloud (for final evaluation)
            
            idx_in_cur = [idx for idx, l in enumerate(torch.argmax(full_pred, 1).numpy()) if l in self._dataset.stuff_classes]
            idx_in_cur = np.array(idx_in_cur)
            idx_in_cur.astype(int)
            
            full_ins_pred_embed[idx_in_cur] = -1
            full_ins_pred_offset[idx_in_cur] = -1

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
                full_ins_pred_embed.numpy(),
                full_ins_pred_offset.numpy(),
                )
            
            self._dataset.final_eval()
            #instance prediction with color for "things"
            things_idx_embed = full_ins_pred_embed != -1
            self._dataset.to_ins_ply(
                self._test_area.pos[things_idx_embed],
                full_ins_pred_embed[things_idx_embed].numpy(),
                "Instance_Embed_results_withColor.ply",
            )
            things_idx_offset = full_ins_pred_offset != -1
            self._dataset.to_ins_ply(
                self._test_area.pos[things_idx_offset],
                full_ins_pred_offset[things_idx_offset].numpy(),
                "Instance_Offset_results_withColor.ply",
            )
            
        if not track_instances:
            return

        rec, _, ap = self._ap_meter_embed.eval(self._iou_threshold)
        self._ap = OrderedDict(sorted(ap.items()))
        self._rec = OrderedDict({})
        for key, val in sorted(rec.items()):
            try:
                value = val[-1]
            except TypeError:
                value = val
            self._rec[key] = value
        rec2, _, ap2 = self._ap_meter_offset.eval(self._iou_threshold)
        self._ap2 = OrderedDict(sorted(ap2.items()))
        self._rec2 = OrderedDict({})
        for key, val in sorted(rec2.items()):
            try:
                value = val[-1]
            except TypeError:
                value = val
            self._rec2[key] = value
            
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

    @property
    def full_confusion_matrix(self):
        return self._full_confusion

    @staticmethod
    def _pred_instances_per_scan(clusters, predicted_labels, batch, scan_id_offset):
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
                    classname=predicted_labels[cl[0]].item(), indices=indices, scan_id=scan_id
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
                        indices=instance_indices,
                        scan_id=scan_id,
                    )
                )
        return instances

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)
        
        metrics["{}_pos_embed".format(self._stage)] = meter_value(self._pos_embed)
        metrics["{}_neg_embed".format(self._stage)] = meter_value(self._neg_embed)
        metrics["{}_Iacc_embed".format(self._stage)] = meter_value(self._acc_meter_embed)
        
        metrics["{}_pos_offset".format(self._stage)] = meter_value(self._pos_offset)
        metrics["{}_neg_offset".format(self._stage)] = meter_value(self._neg_offset)
        metrics["{}_Iacc_offset".format(self._stage)] = meter_value(self._acc_meter_offset)
        
        if self._has_instance_data:
            mAP1 = sum(self._ap.values()) / len(self._ap)
            metrics["{}_map_embed".format(self._stage)] = mAP1
            mAP2 = sum(self._ap2.values()) / len(self._ap2)
            metrics["{}_map_offset".format(self._stage)] = mAP2

        if verbose:
            metrics["{}_iou_per_class".format(self._stage)] = self._iou_per_class
            if self._vote_miou:
                metrics["{}_full_vote_miou".format(self._stage)] = self._full_vote_miou
                metrics["{}_vote_miou".format(self._stage)] = self._vote_miou
    
        if verbose and self._has_instance_data:
            metrics["{}_class_rec_embed".format(self._stage)] = self._dict_to_str(self._rec)
            metrics["{}_class_ap_embed".format(self._stage)] = self._dict_to_str(self._ap)
            metrics["{}_class_rec_offset".format(self._stage)] = self._dict_to_str(self._rec)
            metrics["{}_class_ap_offset".format(self._stage)] = self._dict_to_str(self._ap)
        return metrics

    @property
    def _has_instance_data(self):
        return len(self._rec)
