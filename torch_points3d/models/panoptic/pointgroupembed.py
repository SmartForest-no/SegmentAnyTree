import torch
import os
from torch_points_kernels import region_grow
from torch_geometric.data import Data
from torch_scatter import scatter
import random

from sklearn.cluster import MeanShift
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.models.base_model import BaseModel
from torch_points3d.applications.minkowski import Minkowski
from torch_points3d.core.common_modules import Seq, MLP, FastBatchNorm1d
from torch_points3d.core.losses import offset_loss, instance_iou_loss, mask_loss, instance_ious, discriminative_loss
from torch_points3d.core.data_transform import GridSampling3D
from .structures_embed import PanopticLabels, PanopticResults
from torch_points3d.utils import is_list
#from hdbscan import HDBSCAN
#from cuml.cluster import HDBSCAN
import numpy as np
from sklearn.preprocessing import normalize
from numba import njit, jit
from multiprocessing import Pool
#import ray
from functools import  partial
from torch_points3d.utils import hdbscan_cluster, meanshift_cluster
#import cupy as cp


class PointGroupEmbed(BaseModel):
    __REQUIRED_DATA__ = [
        "pos",
    ]

    __REQUIRED_LABELS__ = list(PanopticLabels._fields)

    def __init__(self, option, model_type, dataset, modules):
        super(PointGroupEmbed, self).__init__(option)
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

        self.Semantic = (
            Seq()
            .append(MLP([self.Backbone.output_nc, self.Backbone.output_nc], bias=False))
            .append(torch.nn.Linear(self.Backbone.output_nc, dataset.num_classes))
            .append(torch.nn.LogSoftmax(dim=-1))
        )

        self.Embed = Seq().append(MLP([self.Backbone.output_nc, self.Backbone.output_nc], bias=False))
        self.Embed.append(torch.nn.Linear(self.Backbone.output_nc, option.get("embed_dim", 5)))

        self.loss_names = ["loss", "semantic_loss", "ins_loss", "ins_var_loss", "ins_dist_loss", "ins_reg_loss", "score_loss", "mask_loss"]
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
        embed_logits = self.Embed(backbone_features) # [N, 5]

        # Grouping and scoring
        cluster_scores = None
        mask_scores = None
        all_clusters = None # list of clusters (point idx)
        cluster_type = None # 0 for cluster, 1 for vote
        if self.use_score_net: # and epoch > self.opt.prepare_epoch:
            if epoch > self.opt.prepare_epoch:   # Active by default epoch > -1: #
                if self.opt.cluster_type == 1:
                    all_clusters, cluster_type = self._cluster(semantic_logits, embed_logits)
                elif self.opt.cluster_type == 2:
                    all_clusters, cluster_type = self._cluster2(semantic_logits, embed_logits)
                elif self.opt.cluster_type == 3:
                    all_clusters, cluster_type = self._cluster3(semantic_logits, embed_logits)
                elif self.opt.cluster_type == 4:
                    all_clusters, cluster_type = self._cluster4(semantic_logits, embed_logits)
                elif self.opt.cluster_type == 5:
                    all_clusters, cluster_type = self._cluster5(semantic_logits, embed_logits)
                elif self.opt.cluster_type == 6:
                    all_clusters, cluster_type = self._cluster6(semantic_logits, embed_logits)
                elif self.opt.cluster_type == 7:
                    all_clusters, cluster_type = self._cluster7(semantic_logits, embed_logits)
                elif self.opt.cluster_type == 8:
                    all_clusters, cluster_type = self._cluster8(semantic_logits, embed_logits)
                elif self.opt.cluster_type == 9:
                    all_clusters, cluster_type = self._cluster9(semantic_logits, embed_logits)
                elif self.opt.cluster_type == 10:
                    all_clusters, cluster_type = self._cluster10(semantic_logits, embed_logits)
                elif self.opt.cluster_type == 11:
                    all_clusters, cluster_type = self._cluster11(semantic_logits, embed_logits)
                elif self.opt.cluster_type == 12:
                    all_clusters, cluster_type = self._cluster12(semantic_logits, embed_logits)
                elif self.opt.cluster_type == 13:
                    all_clusters, cluster_type = self._cluster13(semantic_logits, embed_logits)
                elif self.opt.cluster_type == 14:
                    all_clusters, cluster_type = self._cluster14(semantic_logits, embed_logits)
                elif self.opt.cluster_type == 15:
                    all_clusters, cluster_type = self._cluster15(semantic_logits, embed_logits)
                elif self.opt.cluster_type == 16:
                    all_clusters, cluster_type = self._cluster16(semantic_logits, embed_logits)
                if len(all_clusters):
                    cluster_scores, mask_scores = self._compute_score(epoch, all_clusters, backbone_features, semantic_logits)
                    #cluster_scores, mask_scores = self._compute_score_batch(epoch, all_clusters, cluster_type, backbone_features, semantic_logits)
                    #cluster_scores, mask_scores = self._compute_real_score(epoch, all_clusters, cluster_type, backbone_features, semantic_logits)
        else:
            with torch.no_grad():
                if epoch % 1 == 0:
                    if self.opt.cluster_type == 1:
                        all_clusters, cluster_type = self._cluster(semantic_logits, embed_logits)
                    elif self.opt.cluster_type == 2:
                        all_clusters, cluster_type = self._cluster2(semantic_logits, embed_logits)
                    elif self.opt.cluster_type == 3:
                        all_clusters, cluster_type = self._cluster3(semantic_logits, embed_logits)
                    elif self.opt.cluster_type == 4:
                        all_clusters, cluster_type = self._cluster4(semantic_logits, embed_logits)
                    elif self.opt.cluster_type == 5:
                        all_clusters, cluster_type = self._cluster5(semantic_logits, embed_logits)
                    elif self.opt.cluster_type == 6:
                        all_clusters, cluster_type = self._cluster6(semantic_logits, embed_logits)
                    elif self.opt.cluster_type == 7:
                        all_clusters, cluster_type = self._cluster7(semantic_logits, embed_logits)
                    elif self.opt.cluster_type == 8:
                        all_clusters, cluster_type = self._cluster8(semantic_logits, embed_logits)
                    elif self.opt.cluster_type == 9:
                        all_clusters, cluster_type = self._cluster9(semantic_logits, embed_logits)
                    elif self.opt.cluster_type == 10:
                        all_clusters, cluster_type = self._cluster10(semantic_logits, embed_logits)
                    elif self.opt.cluster_type == 11:
                        all_clusters, cluster_type = self._cluster11(semantic_logits, embed_logits)
                    elif self.opt.cluster_type == 12:
                        all_clusters, cluster_type = self._cluster12(semantic_logits, embed_logits)
                    elif self.opt.cluster_type == 13:
                        all_clusters, cluster_type = self._cluster13(semantic_logits, embed_logits)
                    elif self.opt.cluster_type == 14:
                        all_clusters, cluster_type = self._cluster14(semantic_logits, embed_logits)
                    elif self.opt.cluster_type == 15:
                        all_clusters, cluster_type = self._cluster15(semantic_logits, embed_logits)
                    elif self.opt.cluster_type == 16:
                        all_clusters, cluster_type = self._cluster16(semantic_logits, embed_logits)
                    #if len(all_clusters):
                    #    cluster_scores, mask_scores = self._compute_score(epoch, all_clusters, backbone_features, semantic_logits)
        
        self.output = PanopticResults(
            semantic_logits=semantic_logits,
            embed_logits=embed_logits,
            clusters=all_clusters,
            cluster_scores=cluster_scores,
            mask_scores=mask_scores,
            cluster_type=cluster_type,
        )

        # Sets visual data for debugging
        #with torch.no_grad():
        #    self._dump_visuals(epoch, backbone_features)

    def meanshift_cluster(self, prediction, bandwidth):
        ms = MeanShift(bandwidth=bandwidth,bin_seeding=True, n_jobs=-1)
        #print ('Mean shift clustering, might take some time ...')
        ms.fit(prediction)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_ 	
        num_clusters = cluster_centers.shape[0]
         
        return num_clusters, torch.from_numpy(labels)

    #Clustering based on embeddings U original coordinates
    def _cluster(self, semantic_logits, embed_logits):
        """ Compute clusters"""
        #remove stuff points
        N = semantic_logits.shape[0]  #.cpu().detach().numpy().shape[0]
        predicted_labels = torch.max(semantic_logits, 1)[1] #.cpu().detach().numpy() # [N]
        ind = torch.arange(0, N)
        unique_predicted_labels = torch.unique(predicted_labels) #np.unique(predicted_labels)
        ignore_labels=self._stuff_classes.to(self.device)  #.cpu().detach().numpy()
        label_mask = torch.ones(predicted_labels.shape[0], dtype=torch.bool) #.cpu().detach().numpy()
        for l in unique_predicted_labels:
            if l in ignore_labels:
                # Build clusters for a given label (ignore other points)
                #label_mask_l = torch.where(x = l, x, 0.)
                label_mask_l = predicted_labels == l
                label_mask[label_mask_l] = False
        local_ind = ind[label_mask]
        label_batch = self.input.batch[label_mask]  #.cpu().detach().numpy()
        unique_in_batch = torch.unique(label_batch)
        
        #Clustering based on original coordinates
        xyz_u = self.raw_pos[label_mask]  #.cpu().detach().numpy()
        clusters_xyz, cluster_type_xyz = hdbscan_cluster.cluster_single(xyz_u, unique_in_batch, label_batch, local_ind, 0)
        #clusters_xyz=[]
        #cluster_type_xyz=[]
        #Clustering based on embeddings
        embeds_u = embed_logits[label_mask]  #.cpu().detach().numpy()
        clusters_embeds, cluster_type_embeds = hdbscan_cluster.cluster_single(embeds_u, unique_in_batch, label_batch, local_ind, 1)

        all_clusters = []
        cluster_type = []
        all_clusters = all_clusters + clusters_xyz
        all_clusters = all_clusters + clusters_embeds
        cluster_type = cluster_type + cluster_type_xyz
        cluster_type = cluster_type + cluster_type_embeds
        all_clusters = [c.clone().detach().to(self.device) for c in all_clusters]
        cluster_type = torch.tensor(cluster_type).to(self.device)
        return all_clusters, cluster_type
    
    #Clustering based on embeddings U other 9 randomly picked feature sets
    def _cluster2(self, semantic_logits, embed_logits):
        """ Compute clusters"""
        #remove stuff points
        N = semantic_logits.shape[0]  #.cpu().detach().numpy().
        predicted_labels = torch.max(semantic_logits, 1)[1]  #.cpu().detach().numpy() # [N]
        ind = torch.arange(0, N)
        unique_predicted_labels = torch.unique(predicted_labels)
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
        
        #9 cluster sets
        all_u = torch.cat((self.raw_pos[label_mask], embed_logits[label_mask]), 1)  #.cpu().detach().numpy()
        others_clusters, others_type = hdbscan_cluster.cluster_loop(all_u, unique_in_batch, label_batch, local_ind, 3, 5, 9)
        #Clustering based on embeddings
        embeds_u = embed_logits[label_mask]  #.cpu().detach().numpy()
        clusters_embeds, cluster_type_embeds = hdbscan_cluster.cluster_single(embeds_u, unique_in_batch, label_batch, local_ind, 9)

        all_clusters = []
        cluster_type = []
        all_clusters = all_clusters + others_clusters
        all_clusters = all_clusters + clusters_embeds
        cluster_type = cluster_type + others_type
        cluster_type = cluster_type + cluster_type_embeds
        all_clusters = [c.clone().detach().to(self.device) for c in all_clusters]
        cluster_type = torch.tensor(cluster_type).to(self.device)
        return all_clusters, cluster_type

    #original coordinates U other 9 randomly picked feature sets
    def _cluster3(self, semantic_logits, embed_logits):
        """ Compute clusters"""
        #remove stuff points
        N = semantic_logits.shape[0]  #.cpu().detach().numpy().shape[0]
        predicted_labels = torch.max(semantic_logits, 1)[1]  #.cpu().detach().numpy() # [N]
        ind = torch.arange(0, N)
        unique_predicted_labels = torch.unique(predicted_labels)
        ignore_labels=self._stuff_classes.to(self.device)  #.cpu().detach().numpy()
        label_mask = torch.ones(predicted_labels.shape[0], dtype=torch.bool)  #.cpu().detach().numpy()
        for l in unique_predicted_labels:
            if l in ignore_labels:
                # Build clusters for a given label (ignore other points)
                label_mask_l = predicted_labels == l
                label_mask[label_mask_l] = False
        local_ind = ind[label_mask]
        label_batch = self.input.batch[label_mask]  #.cpu().detach().numpy()
        unique_in_batch = torch.unique(label_batch)
        
        #9 cluster sets
        all_u = torch.cat((self.raw_pos[label_mask], embed_logits[label_mask]), 1)  #.cpu().detach().numpy()
        others_clusters, others_type = hdbscan_cluster.cluster_loop(all_u, unique_in_batch, label_batch, local_ind, 3, 5, 9)
        #Clustering based on original xyz
        xyz_u = self.raw_pos[label_mask]  #.cpu().detach().numpy()
        clusters_xyzs, cluster_type_xyzs = hdbscan_cluster.cluster_single(xyz_u, unique_in_batch, label_batch, local_ind, 9)

        all_clusters = []
        cluster_type = []
        all_clusters = all_clusters + others_clusters
        all_clusters = all_clusters + clusters_xyzs
        cluster_type = cluster_type + others_type
        cluster_type = cluster_type + cluster_type_xyzs
        all_clusters = [c.clone().detach().to(self.device) for c in all_clusters]
        cluster_type = torch.tensor(cluster_type).to(self.device)
        return all_clusters, cluster_type
    
    #Clustering based on embeddings U original coordinates U other 8 randomly picked feature sets
    def _cluster4(self, semantic_logits, embed_logits):
        """ Compute clusters"""
        #remove stuff points
        N = semantic_logits.shape[0]  #.cpu().detach().numpy().shape[0]
        predicted_labels = torch.max(semantic_logits, 1)[1]  #.cpu().detach().numpy() # [N]
        ind = torch.arange(0, N)
        unique_predicted_labels = torch.unique(predicted_labels)
        ignore_labels=self._stuff_classes.to(self.device)   #.cpu().detach().numpy()
        label_mask = torch.ones(predicted_labels.shape[0], dtype=torch.bool)  #.cpu().detach().numpy()
        for l in unique_predicted_labels:
            if l in ignore_labels:
                # Build clusters for a given label (ignore other points)
                label_mask_l = predicted_labels == l
                label_mask[label_mask_l] = False
        local_ind = ind[label_mask]
        label_batch = self.input.batch[label_mask]  #.cpu().detach().numpy()
        unique_in_batch = torch.unique(label_batch)
        
        #8 cluster sets
        all_u = torch.cat((self.raw_pos[label_mask], embed_logits[label_mask]), 1)  #.cpu().detach().numpy()
        others_clusters, others_type = hdbscan_cluster.cluster_loop(all_u, unique_in_batch, label_batch, local_ind, 3, 5, 8)
        #Clustering based on embeddings
        embeds_u = embed_logits[label_mask]   #.cpu().detach().numpy()
        clusters_embeds, cluster_type_embeds = hdbscan_cluster.cluster_single(embeds_u, unique_in_batch, label_batch, local_ind, 8)
        #Clustering based on original xyz
        xyz_u = self.raw_pos[label_mask]   #.cpu().detach().numpy()
        clusters_xyzs, cluster_type_xyzs = hdbscan_cluster.cluster_single(xyz_u, unique_in_batch, label_batch, local_ind, 9)

        all_clusters = []
        cluster_type = []
        all_clusters = all_clusters + others_clusters
        all_clusters = all_clusters + clusters_embeds
        all_clusters = all_clusters + clusters_xyzs
        cluster_type = cluster_type + others_type
        cluster_type = cluster_type + cluster_type_embeds
        cluster_type = cluster_type + cluster_type_xyzs
        all_clusters = [c.clone().detach().to(self.device) for c in all_clusters]
        cluster_type = torch.tensor(cluster_type).to(self.device)
        return all_clusters, cluster_type
    
    #10 randomly picked feature sets
    def _cluster5(self, semantic_logits, embed_logits):
        """ Compute clusters"""
        #remove stuff points
        N = semantic_logits.shape[0]  #.cpu().detach().numpy().shape[0]
        predicted_labels = torch.max(semantic_logits, 1)[1]  #.cpu().detach().numpy() # [N]
        ind = torch.arange(0, N)
        unique_predicted_labels = torch.unique(predicted_labels)
        ignore_labels=self._stuff_classes.to(self.device)  #.cpu().detach().numpy()
        label_mask = torch.ones(predicted_labels.shape[0], dtype=torch.bool)  #.cpu().detach().numpy()
        for l in unique_predicted_labels:
            if l in ignore_labels:
                # Build clusters for a given label (ignore other points)
                label_mask_l = predicted_labels == l
                label_mask[label_mask_l] = False
        local_ind = ind[label_mask]
        label_batch = self.input.batch[label_mask]  #.cpu().detach().numpy()
        unique_in_batch = torch.unique(label_batch)
        
        #10 cluster sets
        all_u = torch.cat((self.raw_pos[label_mask], embed_logits[label_mask]), 1)  #.cpu().detach().numpy()
        all_clusters, cluster_type = hdbscan_cluster.cluster_loop(all_u, unique_in_batch, label_batch, local_ind, 3, 5, 10)

        all_clusters = [c.clone().detach().to(self.device) for c in all_clusters]
        cluster_type = torch.tensor(cluster_type).to(self.device)
        return all_clusters, cluster_type
    
    #6 randomly picked feature sets from embeddings
    def _cluster6(self, semantic_logits, embed_logits):
        """ Compute clusters"""
        #remove stuff points
        N = semantic_logits.shape[0]
        predicted_labels = torch.max(semantic_logits, 1)[1] #.cpu().detach().numpy() # [N]
        ind = torch.arange(0, N)
        unique_predicted_labels = torch.unique(predicted_labels)
        ignore_labels=self._stuff_classes.to(self.device)  #.cpu().detach().numpy()
        label_mask = torch.ones(predicted_labels.shape[0], dtype=torch.bool)  #.cpu().detach().numpy()
        for l in unique_predicted_labels:
            if l in ignore_labels:
                # Build clusters for a given label (ignore other points)
                label_mask_l = predicted_labels == l
                label_mask[label_mask_l] = False
        local_ind = ind[label_mask]
        label_batch = self.input.batch[label_mask]   #.cpu().detach().numpy()
        unique_in_batch = torch.unique(label_batch)
        
        #6 cluster sets from embeddings
        all_u = embed_logits[label_mask]  #.cpu().detach().numpy()
        all_clusters, cluster_type = hdbscan_cluster.cluster_loop(all_u, unique_in_batch, label_batch, local_ind, 2, 5, 6)
        all_clusters = [c.clone().detach().to(self.device) for c in all_clusters]
        cluster_type = torch.tensor(cluster_type).to(self.device)
        return all_clusters, cluster_type   
    
    #original coordinates U other 9 randomly picked feature sets from embeddings
    def _cluster9(self, semantic_logits, embed_logits):
        """ Compute clusters"""
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
        clusters_embed, cluster_type_embeds = meanshift_cluster.cluster_loop(embeds_u, unique_in_batch, label_batch, local_ind, 3, 5, 10)

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
 
    #clustering based on embedding features + meanshift
    def _cluster7(self, semantic_logits, embed_logits):
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
    def _cluster8(self, semantic_logits, embed_logits):
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
        
        
    #6 randomly picked feature sets from embeddings (meanshift)
    def _cluster10(self, semantic_logits, embed_logits):
        """ Compute clusters"""
        #remove stuff points
        N = semantic_logits.shape[0]
        predicted_labels = torch.max(semantic_logits, 1)[1] #.cpu().detach().numpy() # [N]
        ind = torch.arange(0, N)
        unique_predicted_labels = torch.unique(predicted_labels)
        ignore_labels=self._stuff_classes.to(self.device)  #.cpu().detach().numpy()
        label_mask = torch.ones(predicted_labels.shape[0], dtype=torch.bool)  #.cpu().detach().numpy()
        for l in unique_predicted_labels:
            if l in ignore_labels:
                # Build clusters for a given label (ignore other points)
                label_mask_l = predicted_labels == l
                label_mask[label_mask_l] = False
        local_ind = ind[label_mask]
        label_batch = self.input.batch[label_mask]   #.cpu().detach().numpy()
        unique_in_batch = torch.unique(label_batch)
        
        #6 cluster sets from embeddings
        all_u = embed_logits[label_mask]  #.cpu().detach().numpy()
        all_clusters, cluster_type = meanshift_cluster.cluster_loop(all_u, unique_in_batch, label_batch, local_ind, 2, 5, 6)
        all_clusters = [c.clone().detach().to(self.device) for c in all_clusters]
        cluster_type = torch.tensor(cluster_type).to(self.device)
        return all_clusters, cluster_type  
    
    #6 randomly picked feature sets from embeddings
    def _cluster11(self, semantic_logits, embed_logits):
        """ Compute clusters"""
        #remove stuff points
        N = semantic_logits.shape[0]
        predicted_labels = torch.max(semantic_logits, 1)[1] #.cpu().detach().numpy() # [N]
        ind = torch.arange(0, N)
        unique_predicted_labels = torch.unique(predicted_labels)
        ignore_labels=self._stuff_classes.to(self.device)  #.cpu().detach().numpy()
        label_mask = torch.ones(predicted_labels.shape[0], dtype=torch.bool)  #.cpu().detach().numpy()
        for l in unique_predicted_labels:
            if l in ignore_labels:
                # Build clusters for a given label (ignore other points)
                label_mask_l = predicted_labels == l
                label_mask[label_mask_l] = False
        local_ind = ind[label_mask]
        label_batch = self.input.batch[label_mask]   #.cpu().detach().numpy()
        unique_in_batch = torch.unique(label_batch)
        
        #6 cluster sets from embeddings
        all_u = embed_logits[label_mask]  #.cpu().detach().numpy()
        all_clusters, cluster_type = hdbscan_cluster.cluster_loop_fixedD(all_u, unique_in_batch, label_batch, local_ind, 2, 5, 6)
        all_clusters = [c.clone().detach().to(self.device) for c in all_clusters]
        cluster_type = torch.tensor(cluster_type).to(self.device)
        return all_clusters, cluster_type   
    
    #clustering based on embedding features + meanshift U original coordinates + regiongrowing
    def _cluster12(self, semantic_logits, embed_logits):
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
        clusters_embed, cluster_type_embeds = meanshift_cluster.cluster_loop(embeds_u, unique_in_batch, label_batch, local_ind, 2, 5, 6)


        ###### Combine the two groups of clusters ######
        all_clusters = []
        cluster_type = []
        all_clusters = all_clusters + clusters_pos
        all_clusters = all_clusters + clusters_embed
        cluster_type = cluster_type + cluster_type_embeds
        cluster_type = cluster_type + list(np.ones(len(clusters_pos), dtype=np.uint8)*6)
        all_clusters = [c.clone().detach().to(self.device) for c in all_clusters]
        cluster_type = torch.tensor(cluster_type).to(self.device)
        return all_clusters, cluster_type
    
    #Clustering based on xyz U other 6 randomly picked feature sets
    def _cluster13(self, semantic_logits, embed_logits):
        """ Compute clusters"""
        #remove stuff points
        N = semantic_logits.shape[0]  #.cpu().detach().numpy().
        predicted_labels = torch.max(semantic_logits, 1)[1]  #.cpu().detach().numpy() # [N]
        ind = torch.arange(0, N)
        unique_predicted_labels = torch.unique(predicted_labels)
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
        
        #9 cluster sets
        all_u = embed_logits[label_mask]  #.cpu().detach().numpy()
        others_clusters, others_type = hdbscan_cluster.cluster_loop_fixedD(all_u, unique_in_batch, label_batch, local_ind, 2, 5, 6)
        #Clustering based on embeddings
        xyz_u = self.raw_pos[label_mask]  #.cpu().detach().numpy()
        clusters_xyz, cluster_type_xyz = hdbscan_cluster.cluster_single(xyz_u, unique_in_batch, label_batch, local_ind, 6)
        
        all_clusters = []
        cluster_type = []
        all_clusters = all_clusters + others_clusters
        all_clusters = all_clusters + clusters_xyz
        cluster_type = cluster_type + others_type
        cluster_type = cluster_type + cluster_type_xyz
        all_clusters = [c.clone().detach().to(self.device) for c in all_clusters]
        cluster_type = torch.tensor(cluster_type).to(self.device)
        return all_clusters, cluster_type
 
    #clustering based on embedding features + meanshift
    def _cluster14(self, semantic_logits, embed_logits):
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
        
        clusters_embed, cluster_type_embeds = hdbscan_cluster.cluster_single(embeds_u, unique_in_batch, label_batch, local_ind, 0)

        ###### Combine the two groups of clusters ######
        all_clusters = clusters_embed
        all_clusters = [c.to(self.device) for c in all_clusters]
        cluster_type = torch.zeros(len(all_clusters), dtype=torch.uint8).to(self.device)
        return all_clusters, cluster_type
    
    #Clustering based on xyz U other 6 randomly picked feature sets
    def _cluster15(self, semantic_logits, embed_logits):
        """ Compute clusters"""
        #remove stuff points
        N = semantic_logits.shape[0]  #.cpu().detach().numpy().
        predicted_labels = torch.max(semantic_logits, 1)[1]  #.cpu().detach().numpy() # [N]
        ind = torch.arange(0, N)
        unique_predicted_labels = torch.unique(predicted_labels)
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
        
        #9 cluster sets
        embeds_u = embed_logits[label_mask]  #.cpu().detach().numpy()
        others_clusters, others_type = meanshift_cluster.cluster_loop(embeds_u, unique_in_batch, label_batch, local_ind, 2, 5, 6)
        #Clustering based on embeddings
        
        clusters_xyz, cluster_type_xyz = hdbscan_cluster.cluster_single(embeds_u, unique_in_batch, label_batch, local_ind, 6)
        
        all_clusters = []
        cluster_type = []
        all_clusters = all_clusters + others_clusters
        all_clusters = all_clusters + clusters_xyz
        cluster_type = cluster_type + others_type
        cluster_type = cluster_type + cluster_type_xyz
        all_clusters = [c.clone().detach().to(self.device) for c in all_clusters]
        cluster_type = torch.tensor(cluster_type).to(self.device)
        return all_clusters, cluster_type  
    
    #Clustering based on xyz U other 6 randomly picked feature sets
    def _cluster16(self, semantic_logits, embed_logits):
        """ Compute clusters"""
        #remove stuff points
        N = semantic_logits.shape[0]  #.cpu().detach().numpy().
        predicted_labels = torch.max(semantic_logits, 1)[1]  #.cpu().detach().numpy() # [N]
        ind = torch.arange(0, N)
        unique_predicted_labels = torch.unique(predicted_labels)
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
        
        #9 cluster sets
        embeds_u = embed_logits[label_mask]  #.cpu().detach().numpy()
        
        clusters_xyz, cluster_type_xyz = hdbscan_cluster.cluster_loop(embeds_u, unique_in_batch, label_batch, local_ind, 2, 5, 6)
        
        others_clusters, others_type = meanshift_cluster.cluster_single(embeds_u, unique_in_batch, label_batch, local_ind, 6, self.opt.bandwidth)
        #Clustering based on embeddings
        
        all_clusters = []
        cluster_type = []
        all_clusters = all_clusters + others_clusters
        all_clusters = all_clusters + clusters_xyz
        cluster_type = cluster_type + others_type
        cluster_type = cluster_type + cluster_type_xyz
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
        self.semantic_loss = torch.nn.functional.nll_loss(
            self.output.semantic_logits, (self.labels.y).to(torch.int64), ignore_index=IGNORE_LABEL
        )
        self.loss = self.opt.loss_weights.semantic * self.semantic_loss

        # Embed loss
        self.input.instance_mask = self.input.instance_mask.to(self.device)
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
                self.input.instance_labels,
                self.input.batch,
                min_iou_threshold=self.opt.min_iou_threshold,
                max_iou_threshold=self.opt.max_iou_threshold,
            )
            self.loss = self.loss + self.score_loss * self.opt.loss_weights["score_loss"]

        # Mask loss
        if self.output.mask_scores is not None and self.mask_supervise:
            self.mask_loss = mask_loss(
                ious,
                self.output.clusters,
                mask_scores_sigmoid,
                self.input.instance_labels,
                self.input.batch,
            )
            self.loss = self.loss + self.mask_loss * self.opt.loss_weights["mask_loss"]

    def backward(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self._compute_loss(epoch)
        self.loss.backward()

    def _dump_visuals(self, epoch, backbone_features):
        #if random.random() < self.opt.vizual_ratio:
        if not hasattr(self, "visual_count"):
            self.visual_count = 0
        if not os.path.exists("val1"):
            os.mkdir("val1")
        data_visual = Data(
            pos=self.raw_pos, y=self.input.y, instance_labels=self.input.instance_labels  #, batch=self.input.batch
        )
        #data_visual.semantic_pred = torch.max(self.output.semantic_logits, -1)[1]
        data_visual.backbone_features = backbone_features
        data_visual.embed_features = self.output.embed_logits
        data_visual.semantic_logits = self.output.semantic_logits
        data_visual.coords = self.input.coords
        #data_visual.vote = self.output.offset_logits
        #nms_idx = self.output.get_instances()
        if len(self.output.clusters) and self.input.num_instances>0: #self.output.clusters is not None:
            #data_visual.clusters = [self.output.clusters[i].cpu() for i in nms_idx]
            #data_visual.cluster_type = self.output.cluster_type[nms_idx]
            data_visual.clusters = self.output.clusters
            data_visual.cluster_type = self.output.cluster_type
        
            ious = instance_ious(
                        self.output.clusters,
                        None,
                        self.input.instance_labels.to(self.device),
                        self.input.batch.to(self.device),
                        None,
                        cal_iou_based_on_mask=False
                    )
            ious = ious.max(1)[0]
            data_visual.score_gt = ious
            data_visual.score_pre = self.output.cluster_scores
            
            torch.save(data_visual.to("cpu"), "val1/data_e%i_%i.pt" % (epoch, self.visual_count))
            self.visual_count += 1
