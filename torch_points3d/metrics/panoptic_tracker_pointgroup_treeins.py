from sklearn.metrics import f1_score
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
from sklearn.preprocessing import normalize

# @Treeins: a lot of code copied from torch_points3d/metrics/panoptic_tracker_pointgroup_treeins.py, for changes see @Treeins


time_for_blockMerging = 0
log = logging.getLogger(__name__)


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
        self._metric_func = {**self._metric_func, "pos": max, "neg": min, "map": max, "cov": max, "wcov": max,
                             "mIPre": max, "mIRec": max, "F1": max}

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
        self._cov = tnt.meter.AverageValueMeter()
        self._wcov = tnt.meter.AverageValueMeter()
        self._mIPre = tnt.meter.AverageValueMeter()
        self._mIRec = tnt.meter.AverageValueMeter()
        self._F1 = tnt.meter.AverageValueMeter()
        self._ap_meter = InstanceAPMeter()
        self._scan_id_offset = 0
        self._rec: Dict[str, float] = {}
        self._ap: Dict[str, float] = {}
        self._iou_per_class = {}

    def track(
            self,
            model: TrackerInterface,
            full_res=False,
            data=None,
            iou_threshold=0.5,  # 0.25,
            track_instances=True,
            min_cluster_points=10,
            **kwargs
    ):
        """ Track metrics for panoptic segmentation
        """
        # global block_count
        if not hasattr(self, "block_count"):
            self.block_count = 0  # tells us which cylinder or sphere block track(..) is currently tracking
        if not hasattr(self, "cloud_count"):
            self.cloud_count = 0  # tells us from which data file the currently tracked cylinder or sphere block was sampled

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
        # if not clusters:
        #    return
        predicted_labels = outputs.semantic_logits.max(1)[1]

        if clusters:
            if torch.max(labels.instance_labels) > 0:
                tp, fp, acc = self._compute_acc(
                    clusters, predicted_labels, labels, data.batch, labels.num_instances, iou_threshold
                )
                self._pos.add(tp)
                self._neg.add(fp)
                self._acc_meter.add(acc)

                cov, wcov, mPre, mRec, F1 = self._compute_eval(
                    clusters, predicted_labels, labels, data.batch, labels.num_instances, self._num_classes,
                    iou_threshold
                )
                self._cov.add(cov)
                self._wcov.add(wcov)
                self._mIPre.add(mPre)
                self._mIRec.add(mRec)
                self._F1.add(F1)

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
            self._test_area = self._dataset.test_data  # @Treeins
            for i, test_area_i in enumerate(self._test_area):
                if self._test_area[i].y is None:
                    raise ValueError("It seems that the test area data does not have labels (attribute y).")
                self._test_area[i].prediction_count = torch.zeros(self._test_area[i].y.shape[0], dtype=torch.int)
                self._test_area[i].votes = torch.zeros((self._test_area[i].y.shape[0], self._num_classes),
                                                       dtype=torch.float)
                self._test_area[i].ins_pre = -1 * torch.ones(self._test_area[i].y.shape[0], dtype=torch.int)
                self._test_area[i].max_instance = 0
                self._test_area[i].clusters = []
                self._test_area[i].scores = None
                # self._test_area[i].ori_pos = torch.zeros((self._test_area[i].pos.shape), dtype=torch.float)
                self._test_area[i].to(model.device)

        # Gather origin ids and check that it fits with the test set
        inputs = data if data is not None else model.get_input()
        if inputs[SaveOriginalPosId.KEY] is None:
            raise ValueError("The inputs given to the model do not have a %s attribute." % SaveOriginalPosId.KEY)

        originids = inputs[SaveOriginalPosId.KEY]

        # localoriginids = inputs[SaveLocalOriginalPosId.KEY]
        original_input_ids = self._dataset.test_data_spheres[self.block_count].origin_id
        if originids.dim() == 2:
            originids = originids.flatten()
        if originids.max() >= self._test_area[self.cloud_count].pos.shape[0]:
            raise ValueError("Origin ids are larger than the number of points in the original point cloud.")
        # if originids in original_input_ids:
        #    raise ValueError("Origin ids are larger than the number of points in the original point cloud.")

        # Set predictions
        self._test_area[self.cloud_count].votes[originids] += outputs.semantic_logits
        self._test_area[self.cloud_count].prediction_count[originids] += 1
        # self._test_area[self.cloud_count].ori_pos[originids] = data.coords

        # no scorenet
        if outputs.cluster_scores == None:
            c_scores = None
            curins_pre = self.get_cur_ins_pre_label(clusters, c_scores, predicted_labels.cpu().numpy())
            # block merging
            self._test_area[self.cloud_count].ins_pre, self._test_area[
                self.cloud_count].max_instance = self.block_merging(original_input_ids.cpu().numpy(),
                                                                    originids.cpu().numpy(), curins_pre,
                                                                    self._test_area[
                                                                        self.cloud_count].ins_pre.cpu().numpy(),
                                                                    self._test_area[self.cloud_count].max_instance,
                                                                    model.get_opt_mergeTh(), outputs, predicted_labels)
        # if with scorenet and has valid proposals after NMS and score_threshold
        elif clusters:
            c_scores = outputs.cluster_scores[
                valid_c_idx].cpu().numpy()  # [outputs.cluster_scores[i].cpu().numpy() for i in valid_c_idx]
            curins_pre = self.get_cur_ins_pre_label(clusters, c_scores, predicted_labels.cpu().numpy())
            # block merging
            self._test_area[self.cloud_count].ins_pre, self._test_area[
                self.cloud_count].max_instance = self.block_merging(original_input_ids.cpu().numpy(),
                                                                    originids.cpu().numpy(), curins_pre,
                                                                    self._test_area[
                                                                        self.cloud_count].ins_pre.cpu().numpy(),
                                                                    self._test_area[self.cloud_count].max_instance,
                                                                    model.get_opt_mergeTh(), outputs, predicted_labels)

        #    #c_scores = np.array(c_scores)
        #    self._test_area.clusters, self._test_area.scores = self.block_merging_by_score(self._test_area.scores, self._test_area.clusters, c_scores, clusters, original_input_ids, originids)
        #    self._test_area.ins_pre = self.get_cur_ins_pre_label(self._test_area.clusters, self._test_area.scores.cpu().numpy(), self._test_area.ins_pre.cpu().numpy())
        #    self._test_area.ins_pre = torch.tensor(self._test_area.ins_pre).to(model.device)

        self._dump_visuals_fortest(outputs, originids, valid_c_idx)

        self.block_count += 1  # With each call of track(...), we go on cylinder or sphere block further

        # @Treeins: tells us if we have reached the last cylinder or sphere block belonging to the data file represented by self.cloud_count
        if self.spheres_count == self._dataset.test_data_num_spheres[self.cloud_count]:
            self.cloud_count += 1  # go cylinder or shpere blocks sampled from next data file
            self.spheres_count = 0  # tells us which cylinder or sphere block we are currently tracking WITHIN the data file represented by self.cloud_count

    def _dump_visuals_fortest(self, outputs, originids, valid_c_idx):
        if not os.path.exists("viz_for_test_all_proposals"):
            os.mkdir("viz_for_test_all_proposals")
        if not os.path.exists("viz_for_test_valid_proposals"):
            os.mkdir("viz_for_test_valid_proposals")
        if not hasattr(self, "spheres_count"):
            self.spheres_count = 0
        j = 0
        for i, cluster in enumerate(outputs.clusters):
            semantic_prob = outputs.semantic_logits[cluster, :].softmax(dim=1)
            score_i = -1
            if outputs.cluster_scores != None:
                score_i = outputs.cluster_scores[i]
            instance_type_i = outputs.cluster_type[i]
            predicted_semlabels = outputs.semantic_logits[cluster, :].max(1)[1]
            mask_score_i = torch.ones_like(predicted_semlabels)
            if outputs.mask_scores != None:
                mask_score_i = outputs.mask_scores[j: cluster.shape[0] + j, :].squeeze().sigmoid()
                j = j + cluster.shape[0]
            example_name = 'instance_sphere{:d}_instance{:d}_score{:f}_type{:d}'.format(self.spheres_count, i, score_i,
                                                                                        instance_type_i)
            val_name = join("viz_for_test_all_proposals", example_name)
            write_ply(val_name,
                      [self._test_area[self.cloud_count].pos[originids[cluster]].detach().cpu().numpy(),
                       semantic_prob.detach().cpu().numpy(),
                       predicted_semlabels.detach().cpu().numpy().astype('int32'),
                       mask_score_i.detach().cpu().numpy().astype('float32'),
                       self._test_area[self.cloud_count].y[originids[cluster]].detach().cpu().numpy().astype('int32')
                       ],
                      ['x', 'y', 'z',
                       'sem_prob_1', 'sem_prob_2',  # @Treeins: two semantic segmentation classes: non-tree and tree
                       'pre_sem_label', 'mask_score', 'gt_sem_label'])
            if i in valid_c_idx:
                val_name = join("viz_for_test_valid_proposals", example_name)
                write_ply(val_name,
                          [self._test_area[self.cloud_count].pos[originids[cluster]].detach().cpu().numpy(),
                           semantic_prob.detach().cpu().numpy(),
                           predicted_semlabels.detach().cpu().numpy().astype('int32'),
                           mask_score_i.detach().cpu().numpy().astype('float32'),
                           self._test_area[self.cloud_count].y[originids[cluster]].detach().cpu().numpy().astype(
                               'int32')
                           ],
                          ['x', 'y', 'z',
                           'sem_prob_1', 'sem_prob_2',  # @Treeins: two semantic segmentation classes: non-tree and tree
                           'pre_sem_label', 'mask_score', 'gt_sem_label'])
        self.spheres_count += 1

    def get_cur_ins_pre_label(self, clusters, cluster_scores, predicted_semlabels):
        cur_ins_pre_label = -1 * np.ones_like(predicted_semlabels)
        if clusters:
            if np.all(cluster_scores != None):
                idx = np.argsort(cluster_scores)
                for i, j in enumerate(idx):
                    cur_ins_pre_label[clusters[j].cpu().numpy()] = i
            else:
                idx = np.ones(len(clusters), dtype=np.int16)
                for i, j in enumerate(idx):
                    cur_ins_pre_label[clusters[i].cpu().numpy()] = i
        return cur_ins_pre_label

    def block_merging(self, originids, origin_sub_ids, pre_sub_ins, all_pre_ins, max_instance, th_merge, outputs,
                      predicted_sem_labels):

        # output interme results
        has_prediction = pre_sub_ins != -1
        # print(np.any(has_prediction))
        if np.any(has_prediction):
            if not os.path.exists("viz"):
                os.mkdir("viz")
            if hasattr(outputs, 'embed_logits'):
                val_name = join("viz", "block_sub_embed_" + str(self.block_count))
                embed_i = outputs.embed_logits.cpu().detach().numpy()
                sample_embed_logits = normalize(embed_i, axis=0)
                write_ply(val_name,
                          [self._test_area[self.cloud_count].pos[origin_sub_ids].detach().cpu().numpy(),
                           pre_sub_ins.astype('int32'),
                           self._test_area[self.cloud_count].instance_labels[
                               origin_sub_ids].detach().cpu().numpy().astype('int32'),
                           sample_embed_logits.astype('float32'),
                           predicted_sem_labels.cpu().numpy().astype('int32'),
                           self._test_area[self.cloud_count].y[origin_sub_ids].detach().cpu().numpy().astype('int32')
                           ],
                          ['x', 'y', 'z', 'preins_label', 'ins_gt', 'embed1', 'embed2', 'embed3', 'embed4', 'embed5',
                           # ['x', 'y', 'z', 'preins_label','ins_gt','embed1','embed2','embed3','embed4','embed5','embed6','embed7','embed8',
                           'pre_sem_label', 'gt_sem_label'])
            if hasattr(outputs, 'offset_logits'):
                val_name = join("viz", "block_sub_offset_" + str(self.block_count))
                offset_i = outputs.offset_logits.cpu().detach().numpy()
                shifted_cor = offset_i + self._test_area[self.cloud_count].pos[origin_sub_ids].detach().cpu().numpy()
                write_ply(val_name,
                          [self._test_area[self.cloud_count].pos[origin_sub_ids].detach().cpu().numpy(),
                           pre_sub_ins.astype('int32'),
                           self._test_area[self.cloud_count].instance_labels[
                               origin_sub_ids].detach().cpu().numpy().astype('int32'),
                           shifted_cor.astype('float32'),
                           predicted_sem_labels.cpu().numpy().astype('int32'),
                           self._test_area[self.cloud_count].y[origin_sub_ids].detach().cpu().numpy().astype('int32')
                           ],
                          ['x', 'y', 'z', 'preins_label', 'ins_gt', 'center_pre_x', 'center_pre_y', 'center_pre_z',
                           'pre_sem_label', 'gt_sem_label'])

            # assign_index  = knn(self._test_area.pos[origin_sub_ids[has_prediction]], self._test_area.pos[originids], k=1)
            assign_index = knn(self._test_area[self.cloud_count].pos[origin_sub_ids],
                               self._test_area[self.cloud_count].pos[originids], k=1)

            y_idx, x_idx = assign_index

            # pre_ins = pre_sub_ins[has_prediction][x_idx.detach().cpu().numpy()]
            pre_ins = pre_sub_ins[x_idx.detach().cpu().numpy()]
            # has_prediction = full_ins_pred != -1
            val_name = join("viz", "block_" + str(self.block_count))
            write_ply(val_name,
                      [self._test_area[self.cloud_count].pos[originids].detach().cpu().numpy(),
                       pre_ins.astype('int32'),
                       ],
                      ['x', 'y', 'z', 'preins_label'])

            t_num_clusters = np.max(pre_ins) + 1
            # print(np.unique(pre_ins))
            # print(t_num_clusters)
            # print("t_num_clusters: {}".format(t_num_clusters))
            # print("max_instance: {}".format(max_instance))
            # print("th_merge: {}".format(th_merge))
            idx = np.argwhere(all_pre_ins[originids] != -1)  # has label
            idx2 = np.argwhere(all_pre_ins[originids] == -1)  # no label

            # all points have no label
            if len(idx) == 0:  # t_num_clusters>0:
                mask_valid = pre_ins != -1
                all_pre_ins[originids[mask_valid]] = pre_ins[mask_valid] + max_instance
                max_instance = max_instance + t_num_clusters
                # print("test")
                # return torch.from_numpy(all_pre_ins), max_instance
            # all points have labels
            elif len(idx2) == 0:
                return torch.from_numpy(all_pre_ins), max_instance
            # part of points have labels
            else:
                # merge by iou
                new_label = pre_ins.reshape(-1)

                for ii_idx in range(t_num_clusters):
                    new_label_ii_idx = originids[np.argwhere(new_label == ii_idx).reshape(-1)]

                    new_has_old_idx = new_label_ii_idx[
                        np.argwhere(all_pre_ins[new_label_ii_idx] != -1)]  # new prediction already has old label
                    new_not_old_idx = new_label_ii_idx[
                        np.argwhere(all_pre_ins[new_label_ii_idx] == -1)]  # new prediction has no old label
                    # print(new_has_old_idx)
                    # print(len(new_label_ii_idx))
                    # print(len(new_has_old_idx))
                    # print(len(new_not_old_idx))
                    if len(new_has_old_idx) == 0:
                        all_pre_ins[new_not_old_idx] = max_instance + 1
                        max_instance = max_instance + 1
                    elif len(new_not_old_idx) == 0:
                        continue
                    else:
                        old_labels_ii = all_pre_ins[new_has_old_idx]
                        un = np.unique(old_labels_ii)
                        # print(un)
                        max_iou_ii = 0
                        max_iou_ii_oldlabel = 0
                        for ig, g in enumerate(un):
                            idx_old_all = originids[np.argwhere(all_pre_ins[originids] == g).reshape(-1)]
                            union_label_idx = np.union1d(idx_old_all, new_label_ii_idx)
                            inter_label_idx = np.intersect1d(idx_old_all, new_label_ii_idx)
                            # print(inter_label_idx.size)
                            iou = float(inter_label_idx.size) / float(union_label_idx.size)
                            # print(iou)
                            if iou > max_iou_ii:
                                max_iou_ii = iou
                                max_iou_ii_oldlabel = g

                        if max_iou_ii > 0.1:  # th_merge:
                            all_pre_ins[new_not_old_idx] = max_iou_ii_oldlabel
                        else:
                            all_pre_ins[new_not_old_idx] = max_instance + 1
                            max_instance = max_instance + 1
        return torch.from_numpy(all_pre_ins), max_instance

    def non_max_suppression(self, ious, scores, threshold):
        ixs = scores.argsort()[::-1]
        pick = []
        while len(ixs) > 0:
            i = ixs[0]
            pick.append(i)
            iou = ious[i, ixs[1:]]
            remove_ixs = np.where(iou > threshold)[0] + 1
            ixs = np.delete(ixs, remove_ixs)
            ixs = np.delete(ixs, 0)
        return pick

    def block_merging_by_score(self, all_scores, all_clusters, new_scores, new_clusters, originids, origin_sub_ids):
        if not new_clusters:
            return all_clusters, all_scores

        assign_index = knn(self._test_area[self.cloud_count].pos[origin_sub_ids],
                           self._test_area[self.cloud_count].pos[originids], k=1)
        y_idx, x_idx = assign_index
        # pre_ins = pre_sub_ins[has_prediction][x_idx.detach().cpu().numpy()]
        # pre_ins = pre_sub_ins[x_idx.detach().cpu().numpy()]
        new_clusters_backp = []
        for i, cluster in enumerate(new_clusters):
            indices = torch.zeros_like(originids, dtype=torch.uint8).to(cluster.device)
            for elem in cluster:
                indices = indices | (x_idx == elem)
            new_clusters_backp.append(originids[indices].to(cluster.device))

        nms_threshold = 0.3
        # min_cluster_points=100
        # min_score=0.2
        # print(len(all_clusters))
        all_prop_to_be_merged = all_clusters + new_clusters_backp
        if all_scores is None:
            all_prop_scores = new_scores
        else:
            all_prop_scores = torch.cat((all_scores, new_scores))

        '''n_prop = len(all_prop_to_be_merged)
        proposal_masks = torch.zeros(n_prop, self._test_area.y.shape[0]) #.to(cluster.device)
        # for i, cluster in enumerate(self.clusters):
        #     proposal_masks[i, cluster] = 1

        proposals_idx = []
        for i, cluster in enumerate(all_prop_to_be_merged):
            proposal_id = torch.ones(len(cluster)).cuda()*i
            proposals_idx.append(torch.vstack((proposal_id,cluster)).T)
        proposals_idx = torch.cat(proposals_idx, dim=0)
        proposals_idx_filtered = proposals_idx
        #proposals_idx_filtered = proposals_idx[_mask]
        proposal_masks[proposals_idx_filtered[:, 0].long(), proposals_idx_filtered[:, 1].long()] = 1

        intersection = torch.mm(proposal_masks, proposal_masks.t())  # (nProposal, nProposal), float, cuda
        proposals_pointnum = proposal_masks.sum(1)  # (nProposal), float, cuda
        proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
        proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
        cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)
        pick_idxs = self.non_max_suppression(cross_ious.cpu().numpy(), all_prop_scores.cpu().numpy(), nms_threshold)
        '''
        n_prop = len(all_prop_to_be_merged)
        cross_ious = np.identity(n_prop)
        for i, cluster_i in enumerate(all_prop_to_be_merged[:-2]):
            # for j, cluster_j in enumerate(all_prop_to_be_merged):
            t1 = cluster_i.cpu().numpy()
            j = i + 1
            t2 = all_prop_to_be_merged[j].cpu().numpy()
            intersection = np.intersect1d(t1, t2).shape[0]
            union = np.union1d(t1, t2).shape[0]
            iou = intersection / union
            cross_ious[i][j] = iou
            cross_ious[j][i] = iou
        pick_idxs = self.non_max_suppression(cross_ious, all_prop_scores.cpu().numpy(), nms_threshold)
        all_scores = []
        all_clusters = []
        for i in pick_idxs:
            # cl_mask = proposals_idx_filtered[:,0]==i
            # cl = proposals_idx_filtered[cl_mask][:,1].long()
            # if len(cl) > min_cluster_points and self.cluster_scores[i] > min_score:
            # all_scores.append(all_prop_scores[i])
            all_clusters.append(all_prop_to_be_merged[i])
        all_scores = all_prop_scores[pick_idxs]
        return all_clusters, all_scores

    def finalise(self, full_res=False, vote_miou=True, ply_output="", track_instances=True, **kwargs):
        """calculate all evaluation metrics for each test data file and save the semantic segmentation and instance
        prediction and the evaluation metrics values for each test file in the outputs folder"""
        if vote_miou and self._test_area:
            for i, test_area_i in enumerate(self._test_area):  # @Treeins: for each test data file
                per_class_iou = self._confusion_matrix.get_intersection_union_per_class()[0]  # @Treeins
                self._iou_per_class = {k: v for k, v in enumerate(per_class_iou)}
                # Complete for points that have a prediction
                test_area_i = test_area_i.to("cpu")
                c = ConfusionMatrix(self._num_classes)
                has_prediction = test_area_i.prediction_count > 0
                gt = test_area_i.y[has_prediction].numpy()
                pred = torch.argmax(test_area_i.votes[has_prediction], 1).numpy()
                gt_effect = gt >= 0
                c.count_predicted_batch(gt[gt_effect], pred[gt_effect])
                self._vote_miou = c.get_average_intersection_union() * 100

                if ply_output:
                    has_prediction = test_area_i.prediction_count > 0
                    # semantic prediction with color for subsampled cloud
                    self._dataset.to_ply(
                        test_area_i.pos[has_prediction].cpu(),
                        torch.argmax(test_area_i.votes[has_prediction], 1).cpu().numpy(),
                        # ply_output,
                        ply_output + ("_{}.ply".format(i)),
                        # @Treeins: save semantic segmentation prediction of current data file
                    )

                    # self._test_area = self._test_area.to("cpu")
                    full_pred = knn_interpolate(
                        test_area_i.votes[has_prediction], test_area_i.pos[has_prediction], test_area_i.pos, k=1,
                    )
                    # semantic prediction with color for full cloud
                    '''self._dataset.to_ply(
                        self._test_area.pos,
                        torch.argmax(full_pred, 1).numpy(),
                        "vote1regularfull.ply",
                    )'''
                    # semantic prediction and GT label full cloud (for final evaluation)
                    self._dataset.to_eval_ply(
                       test_area_i.pos,
                       torch.argmax(full_pred, 1).numpy(), #[0, ..]
                       test_area_i.y,   #[-1, ...]
                       "semantic_result_{}.ply".format(i)
                    )
                    # instance
                    has_prediction = test_area_i.ins_pre != -1
                    # full_ins_pred_embed = knn_interpolate(
                    # torch.reshape(self._test_area.ins_pre_embed[has_prediction], (-1,1)), self._test_area.pos[has_prediction], self._test_area.pos, k=1,
                    # )

                    # instance prediction with color for subsampled cloud
                    self._dataset.to_ins_ply(
                        test_area_i.pos[has_prediction].cpu(),
                        test_area_i.ins_pre[has_prediction].cpu().numpy(),
                        "Instance_subsample_{}.ply".format(i),
                        # @Treeins: save subsampled instance segmentation prediction of current data file
                    )

                    # assign_index = knn(test_area_i.pos[has_prediction], test_area_i.pos, k=1)
                    
                    print("Doing knn")
                    print("data is on device: {}".format(test_area_i.pos.device))

                    assign_index = knn(test_area_i.pos[has_prediction].cpu(), test_area_i.pos.cpu(), k=1, num_workers=48)

                    print("Done knn")

                    # assign_index2  = nearest(self._test_area.pos, self._test_area.pos[has_prediction])

                    y_idx, x_idx = assign_index

                    full_ins_pred = test_area_i.ins_pre[has_prediction][x_idx]

                    full_ins_pred = torch.reshape(full_ins_pred, (-1,))
                    # instance prediction and GT label full cloud (for final evaluation)

                    # idx_in_cur = [idx for idx, l in enumerate(torch.argmax(full_pred, 1).numpy()) if l in self._dataset.stuff_classes]
                    # idx_in_cur = np.array(idx_in_cur)
                    # idx_in_cur.astype(int)

                    # assign no instance label for points belonging to stuff classes
                    pre_sem_labels_full = torch.argmax(full_pred, 1)
                    for idx, l in enumerate(self._dataset.stuff_classes):
                        idx_in_cur = (pre_sem_labels_full == l).nonzero(as_tuple=True)[0].numpy().astype(int)
                        full_ins_pred[idx_in_cur] = -1

                    # If the distance between the point to be assigned label and its nearest point (the point which already has point) is larger than a threshold (e.g., 1m), assign -1 to this point as well.
                    mat_pos = torch.sub(test_area_i.pos[has_prediction][x_idx], test_area_i.pos[y_idx])
                    mat_pos = mat_pos.pow(2)
                    mat_pos = torch.sum(mat_pos, 1)
                    mat_pos = mat_pos.sqrt()
                    idx_in_cur = mat_pos > 1
                    full_ins_pred[idx_in_cur] = -1
                    # for i,k in enumerate(x_idx):
                    #    if (self._test_area.pos[has_prediction][k]-self._test_area.pos[i]).pow(2).sum().sqrt()>1:
                    #        full_ins_pred[i]=-1

                    # remove instance that contains point number less than 10
                    unique_predicted_inslabels = torch.unique(full_ins_pred)
                    for l in unique_predicted_inslabels:
                        if l == -1:
                            continue
                        label_mask_l = full_ins_pred == l
                        size_l = full_ins_pred[label_mask_l].shape[0]
                        if size_l < 10:
                            full_ins_pred[label_mask_l] = -1

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

                    print("writing evaluation txt")
                    # self._dataset.final_eval(
                    #     torch.argmax(full_pred, 1).numpy(),
                    #     full_ins_pred.numpy(),
                    #     full_ins_pred.numpy(),
                    #     test_area_i.y,
                    #     test_area_i.instance_labels,
                    #     "Evaluation_{}.txt".format(i),  # @Treeins: save evaulation metrics of current data file
                    # )
                    # instance prediction with color for "things"
                    print("writing instance ply")

                    things_idx = full_ins_pred != -1
                    self._dataset.to_ins_ply(
                        test_area_i.pos[things_idx],
                        full_ins_pred[things_idx].numpy(),
                        "result_{}.ply".format(i),
                        # @Treeins: save instance segmentation prediction of current data file
                    )

                    print("writing instance ply done")

                    # print( " labels: maciek:  ", self._test_area[0].instance_labels) # TODO: continue here
                    # # instance prediction with color for "things"
                    # things_idx = full_ins_pred != -1
                    # self._dataset.to_eval_ply(
                    #     test_area_i.pos[things_idx],
                    #     full_ins_pred[things_idx].numpy(),
                    #     self._test_area[things_idx].instance_labels,
                    #     "Instance_results_eval_{}.ply".format(i),
                    #     # @Treeins: save instance segmentation prediction of current data file
                    # )

        if not track_instances:
            return

        rec, _, ap = self._ap_meter.eval(self._iou_threshold)
        self._ap = OrderedDict(sorted(ap.items()))
        self._rec = OrderedDict({})
        for key, val in sorted(rec.items()):
            try:
                value = val[-1]
            except TypeError:
                value = val
            self._rec[key] = value

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
            pred_class = torch.mode(predicted_labels[clusters[i]])
            # predicted_labels[clusters[i][0]]
            if gt_class == pred_class[0]:
                tp += 1
            else:
                fp += 1
        acc = tp / len(clusters)
        tp = tp / torch.sum(labels.num_instances).cpu().item()
        fp = fp / torch.sum(labels.num_instances).cpu().item()
        return tp, fp, acc

    @staticmethod
    def _compute_eval(clusters, predicted_labels, labels, batch, num_instances, num_classe, iou_threshold):

        ins_classcount_have = []
        pts_in_pred = [[] for itmp in range(num_classe)]
        for g in clusters:  # each object in prediction
            tmp = torch.zeros_like(predicted_labels, dtype=torch.bool)
            tmp[g] = True
            sem_seg_i = int(torch.mode(predicted_labels[tmp])[0])
            pts_in_pred[sem_seg_i] += [tmp]

        pts_in_gt = [[] for itmp in range(num_classe)]
        unique_in_batch = torch.unique(batch)
        for s in unique_in_batch:
            batch_mask = batch == s
            un = torch.unique(labels.instance_labels[batch_mask])
            for ig, g in enumerate(un):
                if g == -1:
                    continue
                tmp = (labels.instance_labels == g) & batch_mask.to(num_instances.device)
                sem_seg_i = int(torch.mode(labels.y[tmp])[0])
                if sem_seg_i == -1:
                    continue
                pts_in_gt[sem_seg_i] += [tmp]
                ins_classcount_have.append(sem_seg_i)

        all_mean_cov = [[] for itmp in range(num_classe)]
        all_mean_weighted_cov = [[] for itmp in range(num_classe)]
        # instance mucov & mwcov
        for i_sem in range(num_classe):
            sum_cov = 0
            mean_cov = 0
            mean_weighted_cov = 0
            num_gt_point = 0
            if not pts_in_gt[i_sem] or not pts_in_pred[i_sem]:
                all_mean_cov[i_sem].append(0)
                all_mean_weighted_cov[i_sem].append(0)
                continue
            for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                ovmax = 0.
                num_ins_gt_point = torch.sum(ins_gt)
                num_gt_point += num_ins_gt_point
                for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                    union = (ins_pred | ins_gt)
                    intersect = (ins_pred & ins_gt)
                    iou = float(torch.sum(intersect)) / torch.sum(union)

                    if iou >= ovmax:
                        ovmax = iou
                        ipmax = ip

                sum_cov += ovmax
                mean_weighted_cov += ovmax * num_ins_gt_point

            if len(pts_in_gt[i_sem]) != 0:
                mean_cov = sum_cov / len(pts_in_gt[i_sem])
                all_mean_cov[i_sem].append(mean_cov.item())

                mean_weighted_cov /= num_gt_point
                all_mean_weighted_cov[i_sem].append(mean_weighted_cov.item())

        # print(all_mean_cov)
        total_gt_ins = np.zeros(num_classe)
        at = iou_threshold
        tpsins = [[] for itmp in range(num_classe)]
        fpsins = [[] for itmp in range(num_classe)]
        IoU_Tp = np.zeros(num_classe)
        IoU_Mc = np.zeros(num_classe)
        # instance precision & recall
        for i_sem in range(num_classe):
            if not pts_in_pred[i_sem]:
                continue
            IoU_Tp_per = 0
            IoU_Mc_per = 0
            tp = [0.] * len(pts_in_pred[i_sem])
            fp = [0.] * len(pts_in_pred[i_sem])
            # gtflag = np.zeros(len(pts_in_gt[i_sem]))
            if pts_in_gt[i_sem]:
                total_gt_ins[i_sem] += len(pts_in_gt[i_sem])
            for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                ovmax = -1.
                if not pts_in_gt[i_sem]:
                    fp[ip] = 1
                    continue
                for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                    union = (ins_pred | ins_gt)
                    intersect = (ins_pred & ins_gt)
                    iou = (float(torch.sum(intersect)) / torch.sum(union)).item()

                    if iou > ovmax:
                        ovmax = iou
                        # igmax = ig

                if ovmax > 0:
                    IoU_Mc_per += ovmax
                if ovmax >= at:
                    tp[ip] = 1  # true
                    IoU_Tp_per += ovmax
                else:
                    fp[ip] = 1  # false positive

            tpsins[i_sem] += tp
            fpsins[i_sem] += fp
            IoU_Tp[i_sem] += IoU_Tp_per
            IoU_Mc[i_sem] += IoU_Mc_per

        MUCov = torch.zeros(num_classe)
        MWCov = torch.zeros(num_classe)

        for i_sem in range(num_classe):
            MUCov[i_sem] = np.mean(all_mean_cov[i_sem])
            MWCov[i_sem] = np.mean(all_mean_weighted_cov[i_sem])

        precision = torch.zeros(num_classe)
        recall = torch.zeros(num_classe)
        RQ = torch.zeros(num_classe)
        SQ = torch.zeros(num_classe)
        PQ = torch.zeros(num_classe)
        PQStar = torch.zeros(num_classe)
        ins_classcount = [1]  # @Treeins
        set1 = set(ins_classcount)
        set2 = set(ins_classcount_have)
        set3 = set1 & set2
        list3 = list(set3)
        ################################################################
        ######  recall, precision, RQ, SQ, PQ, PQ_star for things ######
        ################################################################
        for i_sem in ins_classcount:
            ###### metrics for offset ######
            if not tpsins[i_sem] or not fpsins[i_sem]:
                continue
            tp = np.asarray(tpsins[i_sem]).astype(np.float64)
            fp = np.asarray(fpsins[i_sem]).astype(np.float64)
            tp = np.sum(tp)
            fp = np.sum(fp)
            # recall and precision
            if (total_gt_ins[i_sem]) == 0:
                rec = 0
            else:
                rec = tp / total_gt_ins[i_sem]
            if (tp + fp) == 0:
                prec = 0
            else:
                prec = tp / (tp + fp)
            precision[i_sem] = prec
            recall[i_sem] = rec
            # RQ, SQ, PQ and PQ_star
            if (prec + rec) == 0:
                RQ[i_sem] = 0
            else:
                RQ[i_sem] = 2 * prec * rec / (prec + rec)
            if tp == 0:
                SQ[i_sem] = 0
            else:
                SQ[i_sem] = IoU_Tp[i_sem] / tp
            PQ[i_sem] = SQ[i_sem] * RQ[i_sem]
            # PQStar[i_sem] = IoU_Mc[i_sem]/total_gt_ins[i_sem]
            PQStar[i_sem] = PQ[i_sem]

        if torch.mean(precision[list3]) + torch.mean(recall[list3]) == 0:
            F1_score = torch.tensor(0.)
        else:
            F1_score = (2 * torch.mean(precision[list3]) * torch.mean(recall[list3])) / (
                        torch.mean(precision[list3]) + torch.mean(recall[list3]))
        cov = torch.mean(MUCov[list3])
        wcov = torch.mean(MWCov[list3])
        mPre = torch.mean(precision[list3])
        mRec = torch.mean(recall[list3])

        return cov, wcov, mPre, mRec, F1_score

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
        self._full_confusion.count_predicted_batch(self._test_area.y[gt_effect].numpy(),
                                                   torch.argmax(full_pred, 1)[gt_effect].numpy())
        self._full_vote_miou = self._full_confusion.get_average_intersection_union() * 100

    @staticmethod
    def _extract_clusters(outputs, min_cluster_points):
        valid_cluster_idx, clusters = outputs.get_instances(min_cluster_points=min_cluster_points)
        # clusters = [outputs.clusters[i] for i in valid_cluster_idx]
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
            if scores == None:
                instances.append(
                    _Instance(
                        classname=predicted_labels[cl[0]].item(), score=-1, indices=indices, scan_id=scan_id
                    )
                )
            else:
                instances.append(
                    _Instance(
                        classname=predicted_labels[cl[0]].item(), score=scores[i].item(), indices=indices,
                        scan_id=scan_id
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

        metrics["{}_cov".format(self._stage)] = meter_value(self._cov)
        metrics["{}_wcov".format(self._stage)] = meter_value(self._wcov)
        metrics["{}_mIPre".format(self._stage)] = meter_value(self._mIPre)
        metrics["{}_mIRec".format(self._stage)] = meter_value(self._mIRec)
        metrics["{}_F1".format(self._stage)] = meter_value(self._F1)

        if self._has_instance_data:
            mAP = sum(self._ap.values()) / len(self._ap)
            metrics["{}_map".format(self._stage)] = mAP

        if verbose:
            metrics["{}_iou_per_class".format(self._stage)] = self._iou_per_class

        if verbose and self._has_instance_data:
            metrics["{}_class_rec".format(self._stage)] = self._dict_to_str(self._rec)
            metrics["{}_class_ap".format(self._stage)] = self._dict_to_str(self._ap)
        return metrics

    @property
    def _has_instance_data(self):
        return len(self._rec)