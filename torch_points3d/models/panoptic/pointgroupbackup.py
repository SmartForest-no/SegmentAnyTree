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
from torch_points3d.utils.mean_shift_cos_gpu  import  MeanShiftCosine
from torch_points3d.utils.mean_shift_euc_gpu  import  MeanShiftEuc
import math
import numpy as np

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
        if epoch > self.opt.prepare_epoch:   # Active by default epoch > -1: #
            all_clusters, cluster_type = self._cluster(semantic_logits, embed_logits)
            if len(all_clusters):
                cluster_scores, mask_scores = self._compute_score(epoch, all_clusters, backbone_features, semantic_logits)

        self.output = PanopticResults(
            semantic_logits=semantic_logits,
            embed_logits=embed_logits,
            clusters=all_clusters,
            cluster_scores=cluster_scores,
            mask_scores=mask_scores,
            cluster_type=cluster_type,
        )

        # Sets visual data for debugging
        with torch.no_grad():
            self._dump_visuals(epoch)

    def meanshift_cluster(self, prediction, bandwidth):
        ms = MeanShift(bandwidth, bin_seeding=True, n_jobs=-1)
        #print ('Mean shift clustering, might take some time ...')
        ms.fit(prediction)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_ 	
        num_clusters = cluster_centers.shape[0]
         
        return num_clusters, torch.from_numpy(labels)
    
    def meanshift_gpu(self, prediction, bandwidth):
        #ms = MeanShiftCosine(bandwidth=0.3, cluster_all=True, GPU=True)
        ms = MeanShiftEuc(bandwidth=0.2, cluster_all=True, GPU=True)
        
        #print ('Mean shift clustering, might take some time ...')
        ms.fit(prediction)
        labels = ms.labels_	
        num_clusters = np.unique(labels).shape[0]
         
        return num_clusters, torch.from_numpy(labels)
    
    
    def meanshift_torch(self, prediction, bandwidth):
        def distance(x, X):
            return torch.sqrt(((x - X)**2).sum(1))
        def gaussian(dist, bandwidth):
            return torch.exp(-0.5 * ((dist / bandwidth))**2) / (bandwidth * torch.sqrt(torch.tensor(2 * math.pi).to(self.device)))
        def meanshift_step(X, bandwidth=0.6):
            for i, x in enumerate(X):
                dist = distance(x, X)
                weight = gaussian(dist, bandwidth)
                X[i] = (weight[:, None] * X).sum(0) / weight.sum()
            return X

        for it in range(300):
            prediction = meanshift_step(prediction)
        torch.unique(prediction,dim=0)
        
        return prediction
    
    def meanshift_torch2(self, prediction, bandwidth, bs=500, iteration=500):
        import math, numpy as np, operator, torch
        def gaussian(d, bw):
            return torch.exp(-0.5*((d/bw))**2) / (bw*math.sqrt(2*math.pi))
        def sum_sqz(a,axis): return a.sum(axis).squeeze(axis)
        def unit_prefix(x, n=1):
            for i in range(n): x = x.unsqueeze(0)
            return x
        def align(x, y, start_dim=2):
            xd, yd = x.dim(), y.dim()
            if xd > yd: y = unit_prefix(y, xd - yd)
            elif yd > xd: x = unit_prefix(x, yd - xd)

            xs, ys = list(x.size()), list(y.size())
            nd = len(ys)
            for i in range(start_dim, nd):
                td = nd-i-1
                if   ys[td]==1: ys[td] = xs[td]
                elif xs[td]==1: xs[td] = ys[td]
            return x.expand(*xs), y.expand(*ys)
        def aligned_op(x,y,f): return f(*align(x,y,0))
        def div(x, y): return aligned_op(x, y, operator.truediv)
        def add(x, y): return aligned_op(x, y, operator.add)
        def sub(x, y): return aligned_op(x, y, operator.sub)
        def mul(x, y): return aligned_op(x, y, operator.mul)
        def dist_b(a,b):
            return torch.sqrt((sub(a.unsqueeze(0),b.unsqueeze(1))**2).sum(2))
        
        data=prediction
        n = prediction.shape[0]
        X = torch.FloatTensor(np.copy(data)).cuda()
        for it in range(iteration):
            for i in range(0,n,bs):
                s = slice(i,min(n,i+bs))
                weight = gaussian(dist_b(X, X[s]), 2)
                num = sum_sqz(mul(weight, X), 1)
                X[s] = div(num, sum_sqz(weight, 1))
        return X

    def _cluster(self, semantic_logits, embed_logits):
        """ Compute clusters from positions and votes """

        ###### Cluster using original position with predicted semantic labels ######
        predicted_labels = torch.max(semantic_logits, 1)[1] # [N]
        clusters_pos = []
        #clusters_pos = region_grow(
        #    self.raw_pos,
        #    predicted_labels,
        #    self.input.batch.to(self.device),
        #    ignore_labels=self._stuff_classes.to(self.device),
        #    radius=self.opt.cluster_radius_search,
        #    min_cluster_size=10
        #)
        ###### Cluster using embedding without predicted semantic labels ######
        instance_num_embed = 0
        clusters_embed = []
        predicted_ins_labels_byEmbed = -1*torch.ones(predicted_labels.size(), dtype=torch.int64)
        ind = torch.arange(0, predicted_labels.shape[0])

        unique_predicted_labels = torch.unique(predicted_labels)
        ignore_labels=self._stuff_classes.to(self.device)
        label_mask = torch.ones(predicted_labels.size(), dtype=torch.bool).to(self.device)
        for l in unique_predicted_labels:
            if l in ignore_labels:
                # Build clusters for a given label (ignore other points)
                label_mask_l = predicted_labels == l
                label_mask[label_mask_l] = False
        local_ind = ind[label_mask]
        # Remap batch to a continuous sequence
        label_batch = self.input.batch[label_mask]
        unique_in_batch = torch.unique(label_batch)
        remaped_batch = torch.empty_like(label_batch)
        for new, old in enumerate(unique_in_batch):
            mask = label_batch == old
            remaped_batch[mask] = new
        embed_logits_logits_u = self.raw_pos[label_mask] #embed_logits[label_mask]
        predicted_labels_u = predicted_labels[label_mask]

        batch_size = torch.unique(remaped_batch)
        for s in batch_size:
            batch_mask = remaped_batch == s
            sampleInBatch_local_ind = local_ind[batch_mask]
            sample_embed_logits = embed_logits_logits_u[batch_mask]
            sample_predicted_labels = predicted_labels_u[batch_mask]
            #meanshift cluster for embeddings
            num_clusters_embed, pre_ins_labels_embed = self.meanshift_cluster(sample_embed_logits.detach().cpu(), self.opt.bandwidth)
            #num_clusters_embed, pre_ins_labels_embed = self.meanshift_torch(sample_embed_logits, self.opt.bandwidth)  
            #num_clusters_embed, pre_ins_labels_embed = self.meanshift_gpu(sample_embed_logits.detach().cpu(), self.opt.bandwidth)  
            predicted_ins_labels_byEmbed[sampleInBatch_local_ind]=pre_ins_labels_embed + instance_num_embed
            instance_num_embed += num_clusters_embed
        unique_preInslabels_embed = torch.unique(predicted_ins_labels_byEmbed)
        for l in unique_preInslabels_embed:
            if l == -1:
                continue
            label_mask = predicted_ins_labels_byEmbed == l
            local_ind = ind[label_mask]
            clusters_embed.append(local_ind)

        ###### Combine the two groups of clusters ######
        all_clusters = clusters_pos + clusters_embed
        all_clusters = [c.to(self.device) for c in all_clusters]
        cluster_type = torch.zeros(len(all_clusters), dtype=torch.uint8).to(self.device)
        if len(clusters_pos):
            cluster_type[len(clusters_pos) :] = 1
        return all_clusters, cluster_type
    
    def _cluster2(self, semantic_logits, embed_logits):
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
        clusters_embed = []
        clusters_embed = region_grow(
            embed_logits,
            predicted_labels,
            self.input.batch.to(self.device),
            ignore_labels=self._stuff_classes.to(self.device),
            radius=13,
            nsample=200,
            min_cluster_size=10
        )

        all_clusters = clusters_pos + clusters_embed
        all_clusters = [c.to(self.device) for c in all_clusters]
        cluster_type = torch.zeros(len(all_clusters), dtype=torch.uint8).to(self.device)
        if len(clusters_pos):
            cluster_type[len(clusters_pos) :] = 1
        return all_clusters, cluster_type
    
    def _cluster3(self, semantic_logits, embed_logits):
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
        
        # Remap batch to a continuous sequence
        label_batch = self.input.batch[label_mask].cpu().detach().numpy()
        unique_in_batch = np.unique(label_batch)
        #remaped_batch = torch.empty_like(label_batch)
        #for new, old in enumerate(unique_in_batch):
        #    mask = label_batch == old
        #    remaped_batch[mask] = new
        
        embed_logits_logits_u = torch.cat((self.raw_pos[label_mask], embed_logits[label_mask]), 1).cpu().detach().numpy()
        #embed_logits_logits_u = normalize(embed_logits_logits_u, axis=0)
        #torch.cat((self.raw_pos[label_mask], embed_logits[label_mask], semantic_logits[label_mask]), 1).cpu().detach().numpy()
        
        #predicted_labels_u = predicted_labels[label_mask]

        #batch_size = torch.unique(remaped_batch)
        #predicted_ins_labels_byEmbed = -1*torch.ones(N, dtype=torch.int64)
        #@ray.remote(num_returns=2)
        
        #all_clusters = []
        #cluster_type = []
        # Start Ray.
        #ray.init()
        #for loop_i in range(10):
        #    returns_2 = cluster_loop.remote(embed_logits_logits_u, unique_in_batch, label_batch, local_ind, loop_i)
        #    all_clusters_i, cluster_type_i = ray.get(returns_2)
        #    all_clusters += all_clusters_i
        #    cluster_type += cluster_type_i
            
        #all_clusters, cluster_type = hdbscan_cluster.cluster_loop(embed_logits_logits_u, unique_in_batch, label_batch, local_ind)
        
        #param = []
        #for i in range(0, 10):
        #    param.append(i)
        #pool_size = 10
        #with Pool(processes=pool_size) as pool:
        #    all_clusters, cluster_type = zip(*pool.map(partial(hdbscan_cluster.cluster_loop,embed_logits_logits_u=embed_logits_logits_u, unique_in_batch=unique_in_batch, label_batch=label_batch, local_ind=local_ind), param))
        
                #predicted_ins_labels_byEmbed[sampleInBatch_local_ind[mask_valid]]=pre_ins_labels_embed[mask_valid] + instance_num_embed
                #instance_num_embed += num_clusters_embed
        #unique_preInslabels_embed = torch.unique(predicted_ins_labels_byEmbed)
        #for l in unique_preInslabels_embed:
        #    if l == -1:
        #        continue
        #    label_mask = predicted_ins_labels_byEmbed == l
        #    local_ind = ind[label_mask]
        #    clusters_embed.append(local_ind)

        ###### Combine the two groups of clusters ######
        #all_clusters = clusters_pos + clusters_embed
        all_clusters = [torch.tensor(c).to(self.device) for c in all_clusters]
        #cluster_type = torch.cat(cluster_type)
        cluster_type = torch.tensor(cluster_type).to(self.device)
        #if len(clusters_pos):
        #    cluster_type[len(clusters_pos) :] = 1
        #pool.close()
        #pool.join()
        return all_clusters, cluster_type
    
    #@njit
    #@jit(nopython=True)
    #clustering based on embedding features + meanshift
    def _cluster7(self, semantic_logits, embed_logits):
        """ Compute clusters from positions and votes """

        ###### Cluster using original position with predicted semantic labels ######
        predicted_labels = torch.max(semantic_logits, 1)[1] # [N]
        #clusters_pos = []
        instance_num_embed = 0
        clusters_embed = []
        predicted_ins_labels_byEmbed = -1*torch.ones(predicted_labels.size(), dtype=torch.int64)
        ind = torch.arange(0, predicted_labels.shape[0])

        unique_predicted_labels = torch.unique(predicted_labels)
        ignore_labels=self._stuff_classes.to(self.device)
        label_mask = torch.ones(predicted_labels.size(), dtype=torch.bool).to(self.device)
        for l in unique_predicted_labels:
            if l in ignore_labels:
                # Build clusters for a given label (ignore other points)
                label_mask_l = predicted_labels == l
                label_mask[label_mask_l] = False
        local_ind = ind[label_mask]
        # Remap batch to a continuous sequence
        label_batch = self.input.batch[label_mask]
        unique_in_batch = torch.unique(label_batch)
        remaped_batch = torch.empty_like(label_batch)
        for new, old in enumerate(unique_in_batch):
            mask = label_batch == old
            remaped_batch[mask] = new
        embed_logits_logits_u = embed_logits[label_mask]  
        predicted_labels_u = predicted_labels[label_mask]

        batch_size = torch.unique(remaped_batch)
        for s in batch_size:
            batch_mask = remaped_batch == s
            sampleInBatch_local_ind = local_ind[batch_mask]
            sample_embed_logits = embed_logits_logits_u[batch_mask]
            sample_predicted_labels = predicted_labels_u[batch_mask]
            #meanshift cluster for embeddings
            num_clusters_embed, pre_ins_labels_embed = self.meanshift_cluster(sample_embed_logits.detach().cpu(), self.opt.bandwidth)  
            predicted_ins_labels_byEmbed[sampleInBatch_local_ind]=pre_ins_labels_embed + instance_num_embed
            instance_num_embed += num_clusters_embed
        unique_preInslabels_embed = torch.unique(predicted_ins_labels_byEmbed)
        for l in unique_preInslabels_embed:
            if l == -1:
                continue
            label_mask = predicted_ins_labels_byEmbed == l
            local_ind = ind[label_mask]
            clusters_embed.append(local_ind)

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
        instance_num_embed = 0
        clusters_embed = []
        predicted_ins_labels_byEmbed = -1*torch.ones(predicted_labels.size(), dtype=torch.int64)
        ind = torch.arange(0, predicted_labels.shape[0])

        unique_predicted_labels = torch.unique(predicted_labels)
        ignore_labels=self._stuff_classes.to(self.device)
        label_mask = torch.ones(predicted_labels.size(), dtype=torch.bool).to(self.device)
        for l in unique_predicted_labels:
            if l in ignore_labels:
                # Build clusters for a given label (ignore other points)
                label_mask_l = predicted_labels == l
                label_mask[label_mask_l] = False
        local_ind = ind[label_mask]
        # Remap batch to a continuous sequence
        label_batch = self.input.batch[label_mask]
        unique_in_batch = torch.unique(label_batch)
        remaped_batch = torch.empty_like(label_batch)
        for new, old in enumerate(unique_in_batch):
            mask = label_batch == old
            remaped_batch[mask] = new
        embed_logits_logits_u = embed_logits[label_mask]  
        predicted_labels_u = predicted_labels[label_mask]

        batch_size = torch.unique(remaped_batch)
        for s in batch_size:
            batch_mask = remaped_batch == s
            sampleInBatch_local_ind = local_ind[batch_mask]
            sample_embed_logits = embed_logits_logits_u[batch_mask]
            sample_predicted_labels = predicted_labels_u[batch_mask]
            #meanshift cluster for embeddings
            num_clusters_embed, pre_ins_labels_embed = self.meanshift_cluster(sample_embed_logits.detach().cpu(), self.opt.bandwidth)
            predicted_ins_labels_byEmbed[sampleInBatch_local_ind]=pre_ins_labels_embed + instance_num_embed
            instance_num_embed += num_clusters_embed
        unique_preInslabels_embed = torch.unique(predicted_ins_labels_byEmbed)
        for l in unique_preInslabels_embed:
            if l == -1:
                continue
            label_mask = predicted_ins_labels_byEmbed == l
            local_ind = ind[label_mask]
            clusters_embed.append(local_ind)

        ###### Combine the two groups of clusters ######
        all_clusters = clusters_pos + clusters_embed
        all_clusters = [c.to(self.device) for c in all_clusters]
        cluster_type = torch.zeros(len(all_clusters), dtype=torch.uint8).to(self.device)
        if len(clusters_pos):
            cluster_type[len(clusters_pos) :] = 1
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
                self.loss += self.opt.loss_weights.embedding_loss * loss

        if self.output.mask_scores is not None:
            mask_scores_sigmoid = torch.sigmoid(self.output.mask_scores).squeeze()
        else:
            mask_scores_sigmoid = None
            
        # Calculate iou between each proposal and each GT instance
        if epoch > self.opt.prepare_epoch:
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
            self.loss += self.score_loss * self.opt.loss_weights["score_loss"]

        # Mask loss
        if self.output.mask_scores is not None and self.mask_supervise:
            self.mask_loss = mask_loss(
                ious,
                self.output.clusters,
                mask_scores_sigmoid,
                self.input.instance_labels,
                self.input.batch,
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
