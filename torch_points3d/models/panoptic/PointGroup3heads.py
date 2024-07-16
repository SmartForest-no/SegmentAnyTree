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
from torch_points3d.core.losses import offset_loss, instance_iou_loss, mask_loss, instance_ious, discriminative_loss
from torch_points3d.core.data_transform import GridSampling3D
from .structure_3heads import PanopticLabels, PanopticResults
from torch_points3d.utils import hdbscan_cluster, meanshift_cluster
from torch_points3d.utils import is_list
from torch_points3d.utils import hdbscan_cluster
from os.path import exists, join
from .ply import read_ply, write_ply

class PointGroup3heads(BaseModel):
    __REQUIRED_DATA__ = [
        "pos",
    ]

    __REQUIRED_LABELS__ = list(PanopticLabels._fields)

    def __init__(self, option, model_type, dataset, modules):
        super(PointGroup3heads, self).__init__(option)
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


        self.Embed = Seq().append(MLP([self.Backbone.output_nc, self.Backbone.output_nc], bias=False))
        self.Embed.append(torch.nn.Linear(self.Backbone.output_nc, option.get("embed_dim", 5)))

        self.Semantic = (
            Seq()
            .append(MLP([self.Backbone.output_nc, self.Backbone.output_nc], bias=False))
            .append(torch.nn.Linear(self.Backbone.output_nc, dataset.num_classes))
        )
        self.LogSoftLay = (
            Seq()
            .append(torch.nn.LogSoftmax(dim=-1))
        )
        self.loss_names = ["loss", "offset_norm_loss", "offset_dir_loss",  "ins_loss", "ins_var_loss", "ins_dist_loss", "ins_reg_loss","semantic_loss", "score_loss", "mask_loss"]
        self.use_binary_loss = option.get("use_binary_loss", False)
        if self.use_binary_loss:
            self.BiSemantic = (
                Seq()
                .append(torch.nn.Linear(dataset.num_classes, 2))
                .append(torch.nn.LogSoftmax(dim=-1))
            )
            self.loss_names = ["loss", "offset_norm_loss", "offset_dir_loss",  "ins_loss", "ins_var_loss", "ins_dist_loss", "ins_reg_loss","semantic_loss", "score_loss", "mask_loss", "semantic_loss_bi"]
        
        stuff_classes = dataset.stuff_classes
        if is_list(stuff_classes):
            stuff_classes = torch.Tensor(stuff_classes).long()
        thing_classes = dataset.thing_classes
        if is_list(thing_classes):
            thing_classes = torch.Tensor(thing_classes).long()
        self._stuff_classes = torch.cat([torch.tensor([IGNORE_LABEL]), stuff_classes])
        self._thing_classes = thing_classes
        
        self._weight_per_point = option.get("weight_per_point", None)

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

        # Semantic, offset and embedding heads
        semantic_logits = self.Semantic(backbone_features) # [N, 9]
        semantic_logits = self.LogSoftLay(semantic_logits)
        bi_semantic_logits = None
        if self.use_binary_loss:
            bi_semantic_logits = self.BiSemantic(semantic_logits)
        offset_logits = self.Offset(backbone_features) # [N, 3]
        embed_logits = self.Embed(backbone_features) # [N, 5]

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
                elif self.opt.cluster_type == 3:
                    all_clusters, cluster_type = self._cluster3(semantic_logits, embed_logits)
                elif self.opt.cluster_type == 4:
                    all_clusters, cluster_type = self._cluster4(semantic_logits, embed_logits)
                elif self.opt.cluster_type == 5:
                    all_clusters, cluster_type = self._cluster5(semantic_logits, offset_logits, embed_logits)
                elif self.opt.cluster_type == 6:
                    all_clusters, cluster_type = self._cluster6(semantic_logits, offset_logits, embed_logits)
                elif self.opt.cluster_type == 7: #set1_classes5/set2/set3
                    all_clusters, cluster_type = self._cluster7(semantic_logits, offset_logits, embed_logits)
                if len(all_clusters):
                    cluster_scores, mask_scores = self._compute_score(epoch, all_clusters, backbone_features, semantic_logits)
                    #cluster_scores, mask_scores = self._compute_score_batch(epoch, all_clusters, cluster_type, backbone_features, semantic_logits)
                    #cluster_scores, mask_scores = self._compute_real_score(epoch, all_clusters, cluster_type, backbone_features, semantic_logits)
        else:
            with torch.no_grad():
                if epoch % 1 == 0:
                    if self.opt.cluster_type == 1:
                        all_clusters, cluster_type = self._cluster(semantic_logits, offset_logits)
                    elif self.opt.cluster_type == 2:
                        all_clusters, cluster_type = self._cluster2(semantic_logits, offset_logits)
                    elif self.opt.cluster_type == 3:
                        all_clusters, cluster_type = self._cluster3(semantic_logits, embed_logits)
                    elif self.opt.cluster_type == 4:
                        all_clusters, cluster_type = self._cluster4(semantic_logits, embed_logits)
                    elif self.opt.cluster_type == 5:
                        all_clusters, cluster_type = self._cluster5(semantic_logits, offset_logits, embed_logits)
                    elif self.opt.cluster_type == 6:
                        all_clusters, cluster_type = self._cluster6(semantic_logits, offset_logits, embed_logits)
                    elif self.opt.cluster_type == 7:
                        all_clusters, cluster_type = self._cluster7(semantic_logits, offset_logits, embed_logits)
                
        self.output = PanopticResults(
            semantic_logits=semantic_logits,
            bi_semantic_logits=bi_semantic_logits,
            offset_logits=offset_logits,
            embed_logits=embed_logits,
            clusters=all_clusters,
            cluster_scores=cluster_scores,
            mask_scores=mask_scores,
            cluster_type=cluster_type,
        )

        # Sets visual data for debugging
        #with torch.no_grad():
        #    self._dump_visuals(epoch)

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

    #clustering based on embedding features + meanshift
    def _cluster3(self, semantic_logits, embed_logits):
        """ Compute clusters"""
        #remove stuff points
        N = embed_logits.shape[0]  #.cpu().detach().numpy().shape[0]
        predicted_labels = torch.max(semantic_logits, 1)[1] #.cpu().detach().numpy() # [N]
        ind = torch.arange(0, N)
        unique_predicted_labels = torch.unique(predicted_labels) #np.unique(predicted_labels)
        ignore_labels=self._stuff_classes.to(self.device)  #.cpu().detach().numpy()
        label_mask = torch.ones(predicted_labels.shape[0], dtype=torch.bool) #.cpu().detach().numpy()
        for l in unique_predicted_labels:
            if l in ignore_labels:
                # Build clusters for a given label (ignore other points)
                label_mask_l = predicted_labels == l
                label_mask[label_mask_l] = False
        local_ind = ind[label_mask]
        label_batch = self.input.batch[label_mask]  #.cpu().detach().numpy()
        unique_in_batch = torch.unique(label_batch)
        
        #Clustering based on embeddings
        embeds_u = embed_logits[label_mask]  #.cpu().detach().numpy()
        
        #clusters_embed, cluster_type_embeds = hdbscan_cluster.cluster_single(embeds_u, unique_in_batch, label_batch, local_ind, 0)
        clusters_embed, cluster_type_embeds = meanshift_cluster.cluster_single(embeds_u, unique_in_batch, label_batch, local_ind, 0, self.opt.bandwidth)

        ###### Combine the two groups of clusters ######
        all_clusters = clusters_embed
        all_clusters = [c.to(self.device) for c in all_clusters]
        cluster_type = torch.zeros(len(all_clusters), dtype=torch.uint8).to(self.device)
        return all_clusters, cluster_type
    
    #clustering based on embedding features + meanshift U original coordinates + regiongrowing
    def _cluster4(self, semantic_logits, embed_logits):
        """ Compute clusters from positions and votes """

        ###### Cluster using original position with predicted semantic labels ######
        predicted_labels = torch.max(semantic_logits, 1)[1] # [N]
        clusters_pos = []
        clusters_pos = region_grow(
            self.raw_pos,
            predicted_labels,
            self.input.batch.to(self.device),
            ignore_labels=self._stuff_classes.to(self.device),
            radius=self.opt.cluster_radius_search,
            min_cluster_size=10
        )
        ###### Cluster using embedding without predicted semantic labels ######
        #remove stuff points
        N = embed_logits.shape[0]  #.cpu().detach().numpy().shape[0]
        ind = torch.arange(0, N)
        unique_predicted_labels = torch.unique(predicted_labels) #np.unique(predicted_labels)
        ignore_labels=self._stuff_classes.to(self.device)  #.cpu().detach().numpy()
        label_mask = torch.ones(predicted_labels.shape[0], dtype=torch.bool) #.cpu().detach().numpy()
        for l in unique_predicted_labels:
            if l in ignore_labels:
                # Build clusters for a given label (ignore other points)
                label_mask_l = predicted_labels == l
                label_mask[label_mask_l] = False
        local_ind = ind[label_mask]
        label_batch = self.input.batch[label_mask]  #.cpu().detach().numpy()
        unique_in_batch = torch.unique(label_batch)
        
        #Clustering based on embeddings
        embeds_u = embed_logits[label_mask]  #.cpu().detach().numpy()
        clusters_embed, cluster_type_embeds = meanshift_cluster.cluster_single(embeds_u, unique_in_batch, label_batch, local_ind, 1, self.opt.bandwidth)


        ###### Combine the two groups of clusters ######
        all_clusters = []
        cluster_type = []
        all_clusters = all_clusters + clusters_pos
        all_clusters = all_clusters + clusters_embed
        cluster_type = cluster_type + list(np.zeros(len(clusters_pos), dtype=np.uint8))
        cluster_type = cluster_type + cluster_type_embeds
        all_clusters = [c.clone().detach().to(self.device) for c in all_clusters]
        cluster_type = torch.tensor(cluster_type).to(self.device)
        return all_clusters, cluster_type
    
    #clustering based on embedding features + meanshift U shifted coordinates + regiongrowing
    def _cluster5(self, semantic_logits, offset_logits, embed_logits):
        """ Compute clusters from positions and votes """
        ###### Cluster using original position with predicted semantic labels ######
        predicted_labels = torch.max(semantic_logits, 1)[1] # [N]
        clusters_pos = []
        clusters_pos = region_grow(
            self.raw_pos + offset_logits,
            predicted_labels,
            self.input.batch.to(self.device),
            ignore_labels=self._stuff_classes.to(self.device),
            radius=self.opt.cluster_radius_search,
            nsample=200,
            min_cluster_size=10
        )
        ###### Cluster using embedding without predicted semantic labels ######
        #remove stuff points
        N = embed_logits.shape[0]  #.cpu().detach().numpy().shape[0]
        ind = torch.arange(0, N)
        unique_predicted_labels = torch.unique(predicted_labels) #np.unique(predicted_labels)
        ignore_labels=self._stuff_classes.to(self.device)  #.cpu().detach().numpy()
        label_mask = torch.ones(predicted_labels.shape[0], dtype=torch.bool) #.cpu().detach().numpy()
        for l in unique_predicted_labels:
            if l in ignore_labels:
                # Build clusters for a given label (ignore other points)
                label_mask_l = predicted_labels == l
                label_mask[label_mask_l] = False
        local_ind = ind[label_mask]
        label_batch = self.input.batch[label_mask]  #.cpu().detach().numpy()
        unique_in_batch = torch.unique(label_batch)
        
        #Clustering based on embeddings
        embeds_u = embed_logits[label_mask]  #.cpu().detach().numpy()
        clusters_embed, cluster_type_embeds = meanshift_cluster.cluster_single(embeds_u, unique_in_batch, label_batch, local_ind, 1, self.opt.bandwidth)


        ###### Combine the two groups of clusters ######
        all_clusters = []
        cluster_type = []
        all_clusters = all_clusters + clusters_pos
        all_clusters = all_clusters + clusters_embed
        cluster_type = cluster_type + list(np.zeros(len(clusters_pos), dtype=np.uint8))
        cluster_type = cluster_type + cluster_type_embeds
        all_clusters = [c.clone().detach().to(self.device) for c in all_clusters]
        cluster_type = torch.tensor(cluster_type).to(self.device)
        return all_clusters, cluster_type

    def _cluster6(self, semantic_logits, offset_logits, embed_logits):
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
        ###### Cluster using embedding without predicted semantic labels ######
        #remove stuff points
        N = embed_logits.shape[0]  #.cpu().detach().numpy().shape[0]
        ind = torch.arange(0, N)
        unique_predicted_labels = torch.unique(predicted_labels) #np.unique(predicted_labels)
        ignore_labels=self._stuff_classes.to(self.device)  #.cpu().detach().numpy()
        label_mask = torch.ones(predicted_labels.shape[0], dtype=torch.bool) #.cpu().detach().numpy()
        for l in unique_predicted_labels:
            if l in ignore_labels:
                # Build clusters for a given label (ignore other points)
                label_mask_l = predicted_labels == l
                label_mask[label_mask_l] = False
        local_ind = ind[label_mask]
        label_batch = self.input.batch[label_mask]  #.cpu().detach().numpy()
        unique_in_batch = torch.unique(label_batch)
        
        #Clustering based on embeddings
        embeds_u = embed_logits[label_mask]  #.cpu().detach().numpy()
        clusters_embed, cluster_type_embeds = meanshift_cluster.cluster_single(embeds_u, unique_in_batch, label_batch, local_ind, 2, self.opt.bandwidth)


        ###### Combine the two groups of clusters ######
        all_clusters = []
        cluster_type = []
        all_clusters = all_clusters + clusters_pos
        all_clusters = all_clusters + clusters_votes
        all_clusters = all_clusters + clusters_embed
        cluster_type = cluster_type + list(np.zeros(len(clusters_pos), dtype=np.uint8))
        cluster_type = cluster_type + list(np.ones(len(clusters_votes), dtype=np.uint8))
        cluster_type = cluster_type + cluster_type_embeds
        all_clusters = [c.clone().detach().to(self.device) for c in all_clusters]
        cluster_type = torch.tensor(cluster_type).to(self.device)

        return all_clusters, cluster_type


    def _cluster7(self, semantic_logits, offset_logits, embed_logits):
        """ Compute clusters from positions and votes """
        ###### Cluster using original position with predicted semantic labels ######
        predicted_labels = torch.max(semantic_logits, 1)[1] # [N]
        predicted_labels_copy = predicted_labels.detach().clone()
        #map the predicted semantic labels to the same label : trees
        min_thing_label = torch.min(self._thing_classes)
        for i in self._thing_classes:
            predicted_labels[predicted_labels_copy==i] = min_thing_label.to(self.device)
        clusters_pos = []
        clusters_pos = region_grow(
            self.raw_pos + offset_logits,
            predicted_labels,
            self.input.batch.to(self.device),
            ignore_labels=self._stuff_classes.to(self.device),
            radius=self.opt.cluster_radius_search,
            nsample=200,
            min_cluster_size=10
        )
        ###### Cluster using embedding without predicted semantic labels ######
        #remove stuff points
        N = embed_logits.shape[0]  #.cpu().detach().numpy().shape[0]
        ind = torch.arange(0, N)
        unique_predicted_labels = torch.unique(predicted_labels) #np.unique(predicted_labels)
        ignore_labels=self._stuff_classes.to(self.device)  #.cpu().detach().numpy()
        label_mask = torch.ones(predicted_labels.shape[0], dtype=torch.bool) #.cpu().detach().numpy()
        for l in unique_predicted_labels:
            if l in ignore_labels:
                # Build clusters for a given label (ignore other points)
                label_mask_l = predicted_labels == l
                label_mask[label_mask_l] = False
        local_ind = ind[label_mask]
        label_batch = self.input.batch[label_mask]  #.cpu().detach().numpy()
        unique_in_batch = torch.unique(label_batch)
        
        #Clustering based on embeddings
        embeds_u = embed_logits[label_mask]  #.cpu().detach().numpy()
        clusters_embed, cluster_type_embeds = meanshift_cluster.cluster_single(embeds_u, unique_in_batch, label_batch, local_ind, 1, self.opt.bandwidth)


        ###### Combine the two groups of clusters ######
        all_clusters = []
        cluster_type = []
        all_clusters = all_clusters + clusters_pos
        all_clusters = all_clusters + clusters_embed
        cluster_type = cluster_type + list(np.zeros(len(clusters_pos), dtype=np.uint8))
        cluster_type = cluster_type + cluster_type_embeds
        all_clusters = [c.clone().detach().to(self.device) for c in all_clusters]
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
 
    def _compute_score_batch(self, epoch, all_clusters, cluster_type, backbone_features, semantic_logits):
        """ Score the clusters """
        mask_scores = None
        cluster_scores = torch.zeros(len(all_clusters)).to(self.device)
        cluster_type_unique = torch.unique(cluster_type)
        for type_i in cluster_type_unique:
            type_mask_l = cluster_type == type_i
            type_mask_l = torch.where(type_mask_l)[0]
            if self._scorer_type: # unet
                # Assemble batches
                x = [] # backbone features
                coords = [] # input coords
                batch = [] 
                pos = []
                for i, mask_i in enumerate(type_mask_l):
                    cluster = all_clusters[mask_i]
                    #for i, cluster in enumerate(all_clusters):
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

                cluster_scores[type_mask_l] = self.ScorerHead(cluster_feats).squeeze(-1) # [num_cluster, 1]
                
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

    def _compute_real_score(self, epoch, all_clusters, cluster_type, backbone_features, semantic_logits):
        
        mask_scores = None
        cluster_scores = torch.zeros(len(all_clusters))
        if self.input.num_instances>0:
            ious = instance_ious(
                    all_clusters,
                    None,
                    self.input.instance_labels.to(self.device),
                    self.input.batch.to(self.device),
                    None,
                    cal_iou_based_on_mask=False
                )
            ious = ious.max(1)[0]
            min_iou_threshold = 0 #0.25
            max_iou_threshold = 1 #0.75
            lower_mask = ious < min_iou_threshold
            higher_mask = ious > max_iou_threshold
            middle_mask = torch.logical_and(torch.logical_not(lower_mask), torch.logical_not(higher_mask))
            assert torch.sum(lower_mask + higher_mask + middle_mask) == ious.shape[0]
            cluster_scores = torch.zeros_like(ious)
            iou_middle = ious[middle_mask]
            cluster_scores[higher_mask] = 1
            cluster_scores[middle_mask] = (iou_middle - min_iou_threshold) / (max_iou_threshold - min_iou_threshold)
        return cluster_scores, mask_scores
    
    def _compute_loss(self, epoch):
        # Semantic loss
        attr = getattr(self, 'weight_classes', None)
        if self._weight_per_point != None:
            if self._weight_per_point == 'height':
                # add weights for each point
                above_height = self.raw_pos[:,-1]-self.raw_pos[:,-1].min()
                sample_weight = above_height/(above_height.mean()+np.finfo(np.float64).eps)
                sample_weight = torch.exp(-sample_weight)
                #sample_weight = torch.empty((self.labels.y).to(torch.int64).size()).fill_(0.1)
                self.semantic_loss = torch.nn.functional.nll_loss(
                        self.output.semantic_logits, (self.labels.y).to(torch.int64), ignore_index=IGNORE_LABEL, reduction='none'
                    )
                self.semantic_loss = self.semantic_loss * sample_weight
                self.semantic_loss = self.semantic_loss.mean()
        elif attr is not None:
            self.semantic_loss = torch.nn.functional.nll_loss(
                self.output.semantic_logits, (self.labels.y).to(torch.int64), weight=self.weight_classes, ignore_index=IGNORE_LABEL
            )
        else:
            self.semantic_loss = torch.nn.functional.nll_loss(
                self.output.semantic_logits, (self.labels.y).to(torch.int64), ignore_index=IGNORE_LABEL
            )
        
        self.loss = self.opt.loss_weights.semantic * self.semantic_loss
            
        if self.use_binary_loss:
            bi_y = self.labels.y.detach().clone()
            bi_y[(bi_y[..., None] == self._stuff_classes[1:].to(self.device)).any(-1)] = 0 
            bi_y[(bi_y[..., None] == self._thing_classes.to(self.device)).any(-1)] = 1 
            #self.output.bi_semantic_logits = 0
            self.semantic_loss_bi = torch.nn.functional.nll_loss(
                    self.output.bi_semantic_logits, (bi_y).to(torch.int64), ignore_index=IGNORE_LABEL
                )
            self.loss += self.opt.loss_weights.semantic * self.semantic_loss_bi
        
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
        
        # Embed loss
        self.input.instance_labels = self.input.instance_labels.to(self.device)
        self.input.batch = self.input.batch.to(self.device)

        discriminative_losses = discriminative_loss(
            self.output.embed_logits[self.input.instance_mask],
            self.input.instance_labels[self.input.instance_mask],
            self.input.batch[self.input.instance_mask].to(self.device),
            self.opt.embed_dim
            )
        for loss_name, loss in discriminative_losses.items():
            setattr(self, loss_name, loss)
            if loss_name=="ins_loss":
                self.loss = self.loss + self.opt.loss_weights.embedding_loss * loss


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
                    self.input.instance_labels,
                    self.input.batch,
                    mask_scores_sigmoid,
                    cal_iou_based_on_mask=True
                )
            else:
                ious = instance_ious(
                    self.output.clusters,
                    self.output.cluster_scores,
                    self.input.instance_labels,
                    self.input.batch,
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
        if random.random(): # < self.opt.vizual_ratio:
            if not hasattr(self, "visual_count"):
                self.visual_count = 0
            data_visual = Data(
                pos=self.raw_pos, y=self.input.y, instance_labels=self.input.instance_labels, batch=self.input.batch, vote_label=self.labels.vote_label
            )
            data_visual.semantic_pred = torch.max(self.output.semantic_logits, -1)[1]
            data_visual.vote = self.output.offset_logits
            nms_idx = self.output.get_instances()
            if self.output.clusters is not None:
                data_visual.clusters = [self.output.clusters[i].cpu() for i in nms_idx]
                data_visual.cluster_type = self.output.cluster_type[nms_idx]
            if not os.path.exists("viz"):
                os.mkdir("viz")
            if not os.path.exists("viz/epoch_%i" % (epoch)):
                os.mkdir("viz/epoch_%i" % (epoch))
            #torch.save(data_visual.to("cpu"), "viz/data_e%i_%i.pt" % (epoch, self.visual_count))
            batch_size = torch.unique(data_visual.batch)
            for s in batch_size:
                print(s)
                batch_mask = data_visual.batch == s
                example_name='example_{:d}'.format(self.visual_count)
                val_name = join("viz", "epoch_"+str(epoch), example_name)
                write_ply(val_name,
                            [data_visual.pos[batch_mask].detach().cpu().numpy(), 
                            data_visual.y[batch_mask].detach().cpu().numpy().astype('int32'),
                            data_visual.instance_labels[batch_mask].detach().cpu().numpy().astype('int32'),
                            data_visual.vote_label[batch_mask].detach().cpu().numpy(),
                            data_visual.pos[batch_mask].detach().cpu().numpy()+data_visual.vote_label[batch_mask].detach().cpu().numpy()
                            ],
                            ['x', 'y', 'z', 'sem_label', 'ins_label','offset_x', 'offset_y', 'offset_z', 'center_x', 'center_y', 'center_z'])
                
            self.visual_count += 1
