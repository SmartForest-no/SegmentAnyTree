import torch
import os
from torch_points_kernels import region_grow
from torch_geometric.data import Data
from torch_scatter import scatter
import random
import numpy as np

from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.models.base_model import BaseModel
from torch_points3d.applications.minkowski import Minkowski
from torch_points3d.core.common_modules import Seq, MLP, FastBatchNorm1d
from torch_points3d.core.losses import offset_loss, instance_iou_loss, mask_loss, instance_ious
from torch_points3d.core.data_transform import GridSampling3D
from .structures import PanopticLabels, PanopticResults
from torch_points3d.utils import is_list
from torch_points3d.utils import hdbscan_cluster


class PointGroup(BaseModel):
    __REQUIRED_DATA__ = [
        "pos",
    ]

    __REQUIRED_LABELS__ = list(PanopticLabels._fields)

    def __init__(self, option, model_type, dataset, modules):
        super(PointGroup, self).__init__(option)
        backbone_options = option.get("backbone", {"architecture": "unet"})
        self.Backbone = Minkowski(
            backbone_options.get("architecture", "unet"),
            input_nc=dataset.feature_dimension,
            num_layers=4,
            config=backbone_options.get("config", {}),
        )

        self._scorer_type = option.get("scorer_type", None)
        # cluster_voxel_size = option.get("cluster_voxel_size", 0.05)
        #TODO look at how to do back projection of GridSampling3D
        cluster_voxel_size = False
        if cluster_voxel_size:
            self._voxelizer = GridSampling3D(cluster_voxel_size, quantize_coords=True, mode="mean", return_inverse=True)
        else:
            self._voxelizer = None
        self.ScorerUnet = Minkowski("unet", input_nc=self.Backbone.output_nc, num_layers=4, config=option.scorer_unet)
        self.ScorerEncoder = Minkowski(
            "encoder", input_nc=self.Backbone.output_nc, num_layers=4, config=option.scorer_encoder
        )
        self.ScorerMLP = MLP([self.Backbone.output_nc, self.Backbone.output_nc, self.ScorerUnet.output_nc])
        self.ScorerHead = Seq().append(torch.nn.Linear(self.ScorerUnet.output_nc, 1)).append(torch.nn.Sigmoid())

        self.mask_supervise = option.get("mask_supervise", False)
        if self.mask_supervise:
            self.MaskScore = (
                Seq()
                .append(torch.nn.Linear(self.ScorerUnet.output_nc, self.ScorerUnet.output_nc))
                .append(torch.nn.ReLU())
                .append(torch.nn.Linear(self.ScorerUnet.output_nc, 1))
            )
        self.use_score_net = option.get("use_score_net", True)
        self.use_mask_filter_score_feature = option.get("use_mask_filter_score_feature", False)
        self.use_mask_filter_score_feature_start_epoch = option.get("use_mask_filter_score_feature_start_epoch", 200)
        self.mask_filter_score_feature_thre = option.get("mask_filter_score_feature_thre", 0.5)

        self.cal_iou_based_on_mask = option.get("cal_iou_based_on_mask", False)
        self.cal_iou_based_on_mask_start_epoch = option.get("cal_iou_based_on_mask_start_epoch", 200)

        self.Offset = Seq().append(MLP([self.Backbone.output_nc, self.Backbone.output_nc], bias=False))
        self.Offset.append(torch.nn.Linear(self.Backbone.output_nc, 3))

        self.Semantic = (
            Seq()
            .append(MLP([self.Backbone.output_nc, self.Backbone.output_nc], bias=False))
            .append(torch.nn.Linear(self.Backbone.output_nc, dataset.num_classes))
            .append(torch.nn.LogSoftmax(dim=-1))
        )
        self.loss_names = ["loss", "offset_norm_loss", "offset_dir_loss", "semantic_loss", "score_loss", "mask_loss"]
        stuff_classes = dataset.stuff_classes
        if is_list(stuff_classes):
            stuff_classes = torch.Tensor(stuff_classes).long()
        self._stuff_classes = torch.cat([torch.tensor([IGNORE_LABEL]), stuff_classes])

    def get_opt_mergeTh(self):
        """returns configuration"""
        if self.opt.block_merge_th:
            return self.opt.block_merge_th
        else:
            return 0.01
    
    def set_input(self, data, device):
        self.raw_pos = data.pos.to(device)
        self.input = data
        all_labels = {l: data[l].to(device) for l in self.__REQUIRED_LABELS__}
        self.labels = PanopticLabels(**all_labels)

    def forward(self, epoch=-1, **kwargs):
        # Backbone
        backbone_features = self.Backbone(self.input).x # [N, 16]

        # Semantic and offset heads
        semantic_logits = self.Semantic(backbone_features) # [N, 9]
        offset_logits = self.Offset(backbone_features) # [N, 3]

        # Grouping and scoring
        cluster_scores = None
        mask_scores = None
        all_clusters = None # list of clusters (point idx)
        cluster_type = None # 0 for cluster, 1 for vote
        if self.use_score_net: # and epoch > self.opt.prepare_epoch:
            if epoch > self.opt.prepare_epoch:   # Active by default epoch > -1: #
                if self.opt.cluster_type == 1:
                    all_clusters, cluster_type = self._cluster(semantic_logits, offset_logits)
                elif self.opt.cluster_type == 2:
                    all_clusters, cluster_type = self._cluster2(semantic_logits, offset_logits)
                if len(all_clusters):
                    cluster_scores, mask_scores = self._compute_score(epoch, all_clusters, backbone_features, semantic_logits)
        else:
            with torch.no_grad():
                if epoch % 1 == 0:
                    if self.opt.cluster_type == 1:
                        all_clusters, cluster_type = self._cluster(semantic_logits, offset_logits)
                    elif self.opt.cluster_type == 2:
                        all_clusters, cluster_type = self._cluster2(semantic_logits, offset_logits)
                
        self.output = PanopticResults(
            semantic_logits=semantic_logits,
            offset_logits=offset_logits,
            clusters=all_clusters,
            cluster_scores=cluster_scores,
            mask_scores=mask_scores,
            cluster_type=cluster_type,
        )

        # Sets visual data for debugging
        #with torch.no_grad():
            #self._dump_visuals(epoch)

    def _cluster(self, semantic_logits, offset_logits):
        """ Compute clusters from positions and votes """
        predicted_labels = torch.max(semantic_logits, 1)[1] # [N]
        clusters_votes = region_grow(
            self.raw_pos + offset_logits,
            predicted_labels,
            self.input.batch.to(self.device),
            ignore_labels=self._stuff_classes.to(self.device),
            radius=self.opt.cluster_radius_search,
            nsample=200,
            min_cluster_size=10
        )
        #clusters_votes = []

        all_clusters = clusters_votes
        all_clusters = [c.to(self.device) for c in all_clusters]
        cluster_type = torch.zeros(len(all_clusters), dtype=torch.uint8).to(self.device)
        return all_clusters, cluster_type
    
    def _cluster2(self, semantic_logits, offset_logits):
        """ Compute clusters from positions and votes """
        predicted_labels = torch.max(semantic_logits, 1)[1] # [N]
        clusters_pos = region_grow(
            self.raw_pos,
            predicted_labels,
            self.input.batch.to(self.device),
            ignore_labels=self._stuff_classes.to(self.device),
            radius=self.opt.cluster_radius_search,
            min_cluster_size=10
        )
        #clusters_pos = []
        clusters_votes = region_grow(
            self.raw_pos + offset_logits,
            predicted_labels,
            self.input.batch.to(self.device),
            ignore_labels=self._stuff_classes.to(self.device),
            radius=self.opt.cluster_radius_search,
            nsample=200,
            min_cluster_size=10
        )
        #clusters_votes = []

        all_clusters = clusters_pos + clusters_votes
        all_clusters = [c.to(self.device) for c in all_clusters]
        cluster_type = torch.zeros(len(all_clusters), dtype=torch.uint8).to(self.device)
        if len(clusters_pos):
            cluster_type[len(clusters_pos) :] = 1
        return all_clusters, cluster_type

    #Clustering based on shifted coordinates U original coordinates
    def _cluster3(self, semantic_logits, offset_logits):
        """ Compute clusters"""
        #remove stuff points
        N = semantic_logits.cpu().detach().numpy().shape[0]
        predicted_labels = torch.max(semantic_logits, 1)[1].cpu().detach().numpy() # [N]
        ind = np.arange(0, N)
        unique_predicted_labels = np.unique(predicted_labels)
        ignore_labels=self._stuff_classes.cpu().detach().numpy()
        label_mask = torch.ones(predicted_labels.size, dtype=torch.bool).cpu().detach().numpy()
        for l in unique_predicted_labels:
            if l in ignore_labels:
                # Build clusters for a given label (ignore other points)
                label_mask_l = predicted_labels == l
                label_mask[label_mask_l] = False
        local_ind = ind[label_mask]
        label_batch = self.input.batch[label_mask].cpu().detach().numpy()
        unique_in_batch = np.unique(label_batch)
        
        #Clustering based on original coordinates
        xyz_u = self.raw_pos[label_mask].cpu().detach().numpy()
        clusters_xyz, cluster_type_xyz = hdbscan_cluster.cluster_single(xyz_u, unique_in_batch, label_batch, local_ind, 0)
        #Clustering based on shifted coordinates
        shifted_cor = self.raw_pos + offset_logits
        shift_u = shifted_cor[label_mask].cpu().detach().numpy()
        clusters_embeds, cluster_type_embeds = hdbscan_cluster.cluster_single(shift_u, unique_in_batch, label_batch, local_ind, 1)

        all_clusters = []
        cluster_type = []
        all_clusters += clusters_xyz
        all_clusters += clusters_embeds
        cluster_type += cluster_type_xyz
        cluster_type += cluster_type_embeds
        all_clusters = [torch.tensor(c).to(self.device) for c in all_clusters]
        cluster_type = torch.tensor(cluster_type).to(self.device)
        return all_clusters, cluster_type

    def _compute_score(self, epoch, all_clusters, backbone_features, semantic_logits):
        """ Score the clusters """
        mask_scores = None
        if self._scorer_type: # unet
            # Assemble batches
            x = [] # backbone features
            coords = [] # input coords
            batch = [] 
            pos = []
            for i, cluster in enumerate(all_clusters):
                x.append(backbone_features[cluster])
                coords.append(self.input.coords[cluster])
                batch.append(i * torch.ones(cluster.shape[0]))
                pos.append(self.input.pos[cluster])
            batch_cluster = Data(x=torch.cat(x), coords=torch.cat(coords), batch=torch.cat(batch),)

            # Voxelise if required
            if self._voxelizer:
                batch_cluster.pos = torch.cat(pos)
                batch_cluster = batch_cluster.to(self.device)
                batch_cluster = self._voxelizer(batch_cluster)

            # Score
            batch_cluster = batch_cluster.to("cpu")
            if self._scorer_type == "MLP":
                score_backbone_out = self.ScorerMLP(batch_cluster.x.to(self.device))
                cluster_feats = scatter(
                    score_backbone_out, batch_cluster.batch.long().to(self.device), dim=0, reduce="max"
                )
            elif self._scorer_type == "encoder":
                score_backbone_out = self.ScorerEncoder(batch_cluster)
                cluster_feats = score_backbone_out.x
            else:
                score_backbone_out = self.ScorerUnet(batch_cluster)
                if self.mask_supervise:
                    mask_scores = self.MaskScore(score_backbone_out.x) # [point num of all proposals (voxelized), 1]
                    
                    if self.use_mask_filter_score_feature and epoch > self.use_mask_filter_score_feature_start_epoch:
                        mask_index_select = torch.ones_like(mask_scores)
                        mask_index_select[torch.sigmoid(mask_scores) < self.mask_filter_score_feature_thre] = 0.
                        score_backbone_out.x = score_backbone_out.x * mask_index_select
                    # mask_scores = mask_scores[batch_cluster.inverse_indices] # [point num of all proposals, 1]
                
                cluster_feats = scatter(
                    score_backbone_out.x, batch_cluster.batch.long().to(self.device), dim=0, reduce="max"
                ) # [num_cluster, 16]

            cluster_scores = self.ScorerHead(cluster_feats).squeeze(-1) # [num_cluster, 1]
            
        else:
            # Use semantic certainty as cluster confidence
            with torch.no_grad():
                cluster_semantic = []
                batch = []
                for i, cluster in enumerate(all_clusters):
                    cluster_semantic.append(semantic_logits[cluster, :])
                    batch.append(i * torch.ones(cluster.shape[0]))
                cluster_semantic = torch.cat(cluster_semantic)
                batch = torch.cat(batch)
                cluster_semantic = scatter(cluster_semantic, batch.long().to(self.device), dim=0, reduce="mean")
                cluster_scores = torch.max(torch.exp(cluster_semantic), 1)[0]
        return cluster_scores, mask_scores

    def _compute_loss(self, epoch):
        # Semantic loss
        self.semantic_loss = torch.nn.functional.nll_loss(
            self.output.semantic_logits, (self.labels.y).to(torch.int64), ignore_index=IGNORE_LABEL
        )
        self.loss = self.opt.loss_weights.semantic * self.semantic_loss

        # Offset loss
        self.input.instance_mask = self.input.instance_mask.to(self.device)
        self.input.vote_label = self.input.vote_label.to(self.device)
        offset_losses = offset_loss(
            self.output.offset_logits[self.input.instance_mask],
            self.input.vote_label[self.input.instance_mask],
            torch.sum(self.input.instance_mask),
        )
        for loss_name, loss in offset_losses.items():
            setattr(self, loss_name, loss)
            self.loss += self.opt.loss_weights[loss_name] * loss

        if self.output.mask_scores is not None:
            mask_scores_sigmoid = torch.sigmoid(self.output.mask_scores).squeeze()
        else:
            mask_scores_sigmoid = None
            
        # Calculate iou between each proposal and each GT instance
        if epoch > self.opt.prepare_epoch and self.use_score_net:
            if self.cal_iou_based_on_mask and (epoch > self.cal_iou_based_on_mask_start_epoch):
                ious = instance_ious(
                    self.output.clusters,
                    self.output.cluster_scores,
                    self.input.instance_labels.to(self.device),
                    self.input.batch.to(self.device),
                    mask_scores_sigmoid,
                    cal_iou_based_on_mask=True
                )
            else:
                ious = instance_ious(
                    self.output.clusters,
                    self.output.cluster_scores,
                    self.input.instance_labels.to(self.device),
                    self.input.batch.to(self.device),
                    mask_scores_sigmoid,
                    cal_iou_based_on_mask=False
                )
        # Score loss
        if self.output.cluster_scores is not None and self._scorer_type:
            self.score_loss = instance_iou_loss(
                ious,
                self.output.clusters,
                self.output.cluster_scores,
                self.input.instance_labels.to(self.device),
                self.input.batch.to(self.device),
                min_iou_threshold=self.opt.min_iou_threshold,
                max_iou_threshold=self.opt.max_iou_threshold,
            )
            self.loss += self.score_loss * self.opt.loss_weights["score_loss"]

        # Mask loss
        if self.output.mask_scores is not None and self.mask_supervise:
            self.mask_loss = mask_loss(
                ious,
                self.output.clusters,
                mask_scores_sigmoid,
                self.input.instance_labels.to(self.device),
                self.input.batch.to(self.device),
            )
            self.loss += self.mask_loss * self.opt.loss_weights["mask_loss"]

    def backward(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self._compute_loss(epoch)
        self.loss.backward()

    def _dump_visuals(self, epoch):
        if random.random() < self.opt.vizual_ratio:
            if not hasattr(self, "visual_count"):
                self.visual_count = 0
            data_visual = Data(
                pos=self.raw_pos, y=self.input.y, instance_labels=self.input.instance_labels, batch=self.input.batch
            )
            data_visual.semantic_pred = torch.max(self.output.semantic_logits, -1)[1]
            data_visual.vote = self.output.offset_logits
            nms_idx = self.output.get_instances()
            if self.output.clusters is not None:
                data_visual.clusters = [self.output.clusters[i].cpu() for i in nms_idx]
                data_visual.cluster_type = self.output.cluster_type[nms_idx]
            if not os.path.exists("viz"):
                os.mkdir("viz")
            torch.save(data_visual.to("cpu"), "viz/data_e%i_%i.pt" % (epoch, self.visual_count))
            self.visual_count += 1
