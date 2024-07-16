from typing import Any
import logging
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.nn import Sequential, Dropout, Linear
import torch.nn.functional as F
from torch import nn
import os
import numpy as np
#from .base import Segmentation_MP
from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.modules.KPConv import *
from torch_points3d.core.base_conv.partial_dense import *
from torch_points3d.core.common_modules import MultiHeadClassifier
from torch_points3d.models.base_model import BaseModel
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.datasets.multiscale_data import MultiScaleBatch
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from .structures_mine import PanopticLabels, PanopticResults
from torch_points3d.core.losses import offset_loss, discriminative_loss
import random
from sklearn.cluster import MeanShift
from torch_points3d.utils import is_list
from .ply import read_ply, write_ply
from os.path import exists, join
import time
from torch_points_kernels import region_grow
log = logging.getLogger(__name__)

time_for_offsetClustering = 0
time_for_embeddingClustering = 0
time_for_forwardPass = 0
count_for_inference = 0

class KPConvPaper(UnwrappedUnetBasedModel):
    __REQUIRED_LABELS__ = list(PanopticLabels._fields)
    def __init__(self, option, model_type, dataset, modules):
        # Extract parameters from the dataset
        self._num_classes = dataset.num_classes
        self._weight_classes = dataset.weight_classes
        self._use_category = getattr(option, "use_category", False)
        if self._use_category:
            if not dataset.class_to_segments:
                raise ValueError(
                    "The dataset needs to specify a class_to_segments property when using category information for segmentation"
                )
            self._class_to_seg = dataset.class_to_segments
            self._num_categories = len(self._class_to_seg)
            log.info("Using category information for the predictions with %i categories", self._num_categories)
        else:
            self._num_categories = 0

        # Assemble encoder / decoder
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)

        # Build final MLP
        cls_mlp_opt = option.mlp_cls
        ins_mlp_opt = option.mlp_ins
        offset_mlp_opt = option.mlp_offset
        if self._use_category:
            self.FC_layer = MultiHeadClassifier(
                last_mlp_opt.nn[0],
                self._class_to_seg,
                dropout_proba=last_mlp_opt.dropout,
                bn_momentum=last_mlp_opt.bn_momentum,
            )
        else:
            #semantic head
            in_feat = cls_mlp_opt.nn[0]
            self.Semantic = Sequential()
            for i in range(1, len(cls_mlp_opt.nn)):
                self.Semantic.add_module(
                    str(i),
                    Sequential(
                        *[
                            Linear(in_feat, cls_mlp_opt.nn[i], bias=False),
                            FastBatchNorm1d(cls_mlp_opt.nn[i], momentum=cls_mlp_opt.bn_momentum),
                            LeakyReLU(0.2),
                        ]
                    ),
                )
                in_feat = cls_mlp_opt.nn[i]

            if cls_mlp_opt.dropout:
                self.Semantic.add_module("Dropout", Dropout(p=cls_mlp_opt.dropout))

            self.Semantic.add_module("Class", Lin(in_feat, self._num_classes))
            self.Semantic.add_module("Softmax", nn.LogSoftmax(-1))
            
            #offset head
            in_feat2 = offset_mlp_opt.nn[0]
            self.Offset = Sequential()
            for i in range(1, len(offset_mlp_opt.nn)):
                self.Offset.add_module(
                    str(i),
                    Sequential(
                        *[
                            Linear(in_feat2, offset_mlp_opt.nn[i], bias=False),
                            FastBatchNorm1d(offset_mlp_opt.nn[i], momentum=offset_mlp_opt.bn_momentum),
                            LeakyReLU(0.2),
                        ]
                    ),
                )
                in_feat2 = offset_mlp_opt.nn[i]

            if offset_mlp_opt.dropout:
                self.Offset.add_module("Dropout", Dropout(p=offset_mlp_opt.dropout))

            self.Offset.add_module("Offset", Lin(in_feat2, 3))
            
            #embedding head
            in_feat3 = ins_mlp_opt.nn[0]
            self.Embedding = Sequential()
            for i in range(1, len(ins_mlp_opt.nn)):
                self.Embedding.add_module(
                    str(i),
                    Sequential(
                        *[
                            Linear(in_feat3, ins_mlp_opt.nn[i], bias=False),
                            FastBatchNorm1d(ins_mlp_opt.nn[i], momentum=ins_mlp_opt.bn_momentum),
                            LeakyReLU(0.2),
                        ]
                    ),
                )
                in_feat3 = ins_mlp_opt.nn[i]

            if ins_mlp_opt.dropout:
                self.Embedding.add_module("Dropout", Dropout(p=ins_mlp_opt.dropout))

            self.Embedding.add_module("Embedding", Lin(in_feat3, ins_mlp_opt.embed_dim))

        #self.embed_dim = ins_mlp_opt.embed_dim
        self.loss_names = ["loss", "offset_norm_loss", "offset_dir_loss", "semantic_loss", "ins_loss", "ins_var_loss", "ins_dist_loss", "ins_reg_loss"]
            
        self.lambda_reg = self.get_from_opt(option, ["loss_weights", "lambda_reg"])
        if self.lambda_reg:
            self.loss_names += ["loss_reg"]
        stuff_classes = dataset.stuff_classes
        if is_list(stuff_classes):
            stuff_classes = torch.Tensor(stuff_classes).long()
        self._stuff_classes = torch.cat([torch.tensor([IGNORE_LABEL]), stuff_classes])
        #self.visual_names = ["data_visual"]

    def get_opt_bandwidth(self):
        """returns configuration"""
        if self.opt.bandwidth:
            return self.opt.bandwidth
        else:
            return 0.6
            
    def get_opt_mergeTh(self):
        """returns configuration"""
        if self.opt.block_merge_th:
            return self.opt.block_merge_th
        else:
            return 0.01

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        data = data.to(device)
        #data.x = add_ones(data.pos, data.x, True)
        #print(data)
        self.raw_pos = data.pos
        if isinstance(data, MultiScaleBatch):
            self.pre_computed = data.multiscale
            self.upsample = data.upsample
            del data.upsample
            del data.multiscale
        else:
            self.upsample = None
            self.pre_computed = None

        self.input = data
        self.batch_idx = data.batch
        
        all_labels = {l: data[l] for l in self.__REQUIRED_LABELS__}  #.to(device) for l in self.__REQUIRED_LABELS__}
        self.labels = PanopticLabels(**all_labels)

    def forward(self, epoch=-1, step=-1, is_training=True, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        stack_down = []

        global time_for_forwardPass
        T1 = time.perf_counter()
        data = self.input
        for i in range(len(self.down_modules) - 1):
            data = self.down_modules[i](data, precomputed=self.pre_computed)
            stack_down.append(data)

        data = self.down_modules[-1](data, precomputed=self.pre_computed)
        innermost = False

        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append(data)
            data = self.inner_modules[0](data)
            innermost = True

        for i in range(len(self.up_modules)):
            if i == 0 and innermost:
                data = self.up_modules[i]((data, stack_down.pop()))
            else:
                data = self.up_modules[i]((data, stack_down.pop()), precomputed=self.upsample)

        last_feature = data.x
        # Semantic, embedding and offset heads
        semantic_logits = self.Semantic(last_feature)
        offset_logits = self.Offset(last_feature)
        embedding_logits = self.Embedding(last_feature)
        
        T2 = time.perf_counter()
        time_for_forwardPass += T2 - T1
        #print('time for forward pass:%sms' % ((T2 - T1)*1000))
        #print('total time for forward pass:%sms' % ((time_for_forwardPass)*1000))
        
        embed_clusters = None
        offset_clusters = None
        embed_pre = None
        offset_pre = None
        #print("epoch: {}".format(epoch))
        with torch.no_grad():
            if is_training:
                pass
                #if epoch % 30 == 0 and step % 50 == 0:
                #embed_clusters, offset_clusters, embed_pre, offset_pre = self._cluster(semantic_logits, embedding_logits, offset_logits)
            else:
                #embed_clusters, offset_clusters, embed_pre, offset_pre = self._cluster_2(semantic_logits, embedding_logits, offset_logits)
                embed_clusters, offset_clusters, embed_pre, offset_pre = self._cluster_3(semantic_logits, embedding_logits, offset_logits)
            

        self.output = PanopticResults(
            semantic_logits=semantic_logits,
            offset_logits=offset_logits,
            embedding_logits=embedding_logits,
            embed_clusters=embed_clusters,
            offset_clusters=offset_clusters,
            embed_pre=embed_pre,
            offset_pre=offset_pre,
        )
        
        if self.labels is not None:
            self.compute_loss()

        #self.data_visual = self.input
        #self.data_visual.pred = torch.max(self.output, -1)[1]
        #with torch.no_grad():
        #    if epoch % 1 == 0:
        #        self._dump_visuals_fortest(epoch)
        
        return self.output
    
    #clustering for points for each semantic classes separately
    def _cluster(self, semantic_logits, embedding_logits, offset_logits):
        """ Compute clusters from positions and votes """
        predicted_labels = torch.max(semantic_logits, 1)[1]
        batch = self.batch_idx #self.input.batch  #.to(self.device)
        unique_predicted_labels = torch.unique(predicted_labels)
        ignore_labels=self._stuff_classes.to(self.device)
        embed_clusters = []
        offset_clusters = []
        predicted_ins_labels_byEmbed = -1*torch.ones(predicted_labels.size(), dtype=torch.int64)
        predicted_ins_labels_byOffset = -1*torch.ones(predicted_labels.size(), dtype=torch.int64)
        ind = torch.arange(0, predicted_labels.shape[0])
        instance_num_embed = 0
        instance_num_offset = 0
        for l in unique_predicted_labels:
            if l in ignore_labels:
                continue
            # Build clusters for a given label (ignore other points)
            label_mask = predicted_labels == l
            local_ind = ind[label_mask]
            
            # Remap batch to a continuous sequence
            label_batch = batch[label_mask]
            unique_in_batch = torch.unique(label_batch)
            remaped_batch = torch.empty_like(label_batch)
            for new, old in enumerate(unique_in_batch):
                mask = label_batch == old
                remaped_batch[mask] = new
                
            embedding_logits_u = embedding_logits[label_mask]
            offset_logits_u = offset_logits[label_mask] + self.raw_pos[label_mask]
            
            batch_size = torch.unique(remaped_batch) #remaped_batch[-1] + 1
            for s in batch_size: #range(batch_size):
                batch_mask = remaped_batch == s
                sampleInBatch_local_ind = local_ind[batch_mask]
                sample_offset_logits = offset_logits_u[batch_mask]
                sample_embed_logits = embedding_logits_u[batch_mask]
                #meanshift cluster for offsets
                t_num_clusters, t_pre_ins_labels = self.meanshift_cluster(sample_offset_logits.detach().cpu(), self.opt.bandwidth)
                predicted_ins_labels_byOffset[sampleInBatch_local_ind]=t_pre_ins_labels + instance_num_offset
                instance_num_offset += t_num_clusters
                #meanshift cluster for embeddings
                t_num_clusters2, t_pre_ins_labels2 = self.meanshift_cluster(sample_embed_logits.detach().cpu(), self.opt.bandwidth)
                predicted_ins_labels_byEmbed[sampleInBatch_local_ind]=t_pre_ins_labels2 + instance_num_embed
                instance_num_embed += t_num_clusters2
        unique_preInslabels_embed = torch.unique(predicted_ins_labels_byEmbed)
        unique_preInslabels_offset = torch.unique(predicted_ins_labels_byOffset)
        for l in unique_preInslabels_embed:
            if l == -1:
                continue
            label_mask = predicted_ins_labels_byEmbed == l
            local_ind = ind[label_mask]
            embed_clusters.append(local_ind)
        for l in unique_preInslabels_offset:
            if l == -1:
                continue
            label_mask = predicted_ins_labels_byOffset == l
            local_ind = ind[label_mask]
            offset_clusters.append(local_ind)

        #all_clusters = embed_clusters + offset_clusters
        #all_clusters = [c.to(self.device) for c in all_clusters]

        return embed_clusters, offset_clusters, predicted_ins_labels_byEmbed, predicted_ins_labels_byOffset
        
    #clustering for all "thing" points by ignoring their semantic predictions
    def _cluster_2(self, semantic_logits, embedding_logits, offset_logits):
        """ Compute clusters from positions and votes """
        predicted_labels = torch.max(semantic_logits, 1)[1]
        batch = self.batch_idx #self.input.batch  #.to(self.device)
        unique_predicted_labels = torch.unique(predicted_labels)
        ignore_labels=self._stuff_classes.to(self.device)
        embed_clusters = []
        offset_clusters = []
        predicted_ins_labels_byEmbed = -1*torch.ones(predicted_labels.size(), dtype=torch.int64)
        predicted_ins_labels_byOffset = -1*torch.ones(predicted_labels.size(), dtype=torch.int64)
        ind = torch.arange(0, predicted_labels.shape[0])
        instance_num_embed = 0
        instance_num_offset = 0
        label_mask = torch.ones(predicted_labels.size(), dtype=torch.bool).to(self.device)
        for l in unique_predicted_labels:
            if l in ignore_labels:
                # Build clusters for a given label (ignore other points)
                label_mask_l = predicted_labels == l
                label_mask = label_mask^label_mask_l
        #for l in unique_predicted_labels:
            #if l in ignore_labels:
                #continue
        # Build clusters for a given label (ignore other points)
        #label_mask = predicted_labels == l
        local_ind = ind[label_mask]
        
        # Remap batch to a continuous sequence
        label_batch = batch[label_mask]
        unique_in_batch = torch.unique(label_batch)
        remaped_batch = torch.empty_like(label_batch)
        for new, old in enumerate(unique_in_batch):
            mask = label_batch == old
            remaped_batch[mask] = new
            
        embedding_logits_u = embedding_logits[label_mask]
        offset_logits_u = offset_logits[label_mask] + self.raw_pos[label_mask]
        
        batch_size = torch.unique(remaped_batch) #remaped_batch[-1] + 1
        for s in batch_size: #range(batch_size):
            batch_mask = remaped_batch == s
            sampleInBatch_local_ind = local_ind[batch_mask]
            sample_offset_logits = offset_logits_u[batch_mask]
            sample_embed_logits = embedding_logits_u[batch_mask]
            #meanshift cluster for offsets
            T1 = time.perf_counter()
            t_num_clusters, t_pre_ins_labels = self.meanshift_cluster(sample_offset_logits.detach().cpu(), self.opt.bandwidth)
            T2 =time.perf_counter()
            print('time:%sms' % ((T2 - T1)*1000))
            predicted_ins_labels_byOffset[sampleInBatch_local_ind]=t_pre_ins_labels + instance_num_offset
            instance_num_offset += t_num_clusters
            #meanshift cluster for embeddings
            t_num_clusters2, t_pre_ins_labels2 = self.meanshift_cluster(sample_embed_logits.detach().cpu(), self.opt.bandwidth)
            predicted_ins_labels_byEmbed[sampleInBatch_local_ind]=t_pre_ins_labels2 + instance_num_embed
            instance_num_embed += t_num_clusters2
        unique_preInslabels_embed = torch.unique(predicted_ins_labels_byEmbed)
        unique_preInslabels_offset = torch.unique(predicted_ins_labels_byOffset)
        for l in unique_preInslabels_embed:
            if l == -1:
                continue
            label_mask = predicted_ins_labels_byEmbed == l
            local_ind = ind[label_mask]
            embed_clusters.append(local_ind)
        for l in unique_preInslabels_offset:
            if l == -1:
                continue
            label_mask = predicted_ins_labels_byOffset == l
            local_ind = ind[label_mask]
            offset_clusters.append(local_ind)

        #all_clusters = embed_clusters + offset_clusters
        #all_clusters = [c.to(self.device) for c in all_clusters]

        return embed_clusters, offset_clusters, predicted_ins_labels_byEmbed, predicted_ins_labels_byOffset
        
    #clustering for all "thing" points by ignoring their semantic predictions (embeddings by meanshift, offsets by pointgroup)
    def _cluster_3(self, semantic_logits, embedding_logits, offset_logits):
        """ Compute clusters from positions and votes """
        global count_for_inference
        count_for_inference +=1 
        predicted_labels = torch.max(semantic_logits, 1)[1]
        batch = self.batch_idx #self.input.batch  #.to(self.device)
        unique_predicted_labels = torch.unique(predicted_labels)
        ignore_labels=self._stuff_classes.to(self.device)
        embed_clusters = []
        offset_clusters = []
        predicted_ins_labels_byEmbed = -1*torch.ones(predicted_labels.size(), dtype=torch.int64)
        predicted_ins_labels_byOffset = -1*torch.ones(predicted_labels.size(), dtype=torch.int64)
        ind = torch.arange(0, predicted_labels.shape[0])
        instance_num_embed = 0
        instance_num_offset = 0
        label_mask = torch.ones(predicted_labels.size(), dtype=torch.bool).to(self.device)
        for l in unique_predicted_labels:
            if l in ignore_labels:
                # Build clusters for a given label (ignore other points)
                label_mask_l = predicted_labels == l
                label_mask = label_mask^label_mask_l
        #for l in unique_predicted_labels:
            #if l in ignore_labels:
                #continue
        # Build clusters for a given label (ignore other points)
        #label_mask = predicted_labels == l
        local_ind = ind[label_mask]
        
        # Remap batch to a continuous sequence
        label_batch = batch[label_mask]
        unique_in_batch = torch.unique(label_batch)
        remaped_batch = torch.empty_like(label_batch)
        for new, old in enumerate(unique_in_batch):
            mask = label_batch == old
            remaped_batch[mask] = new
            
        embedding_logits_u = embedding_logits[label_mask]
        offset_logits_u = offset_logits[label_mask] + self.raw_pos[label_mask]
        predicted_labels_u = predicted_labels[label_mask]
        
        batch_size = torch.unique(remaped_batch) #remaped_batch[-1] + 1
        for s in batch_size: #range(batch_size):
            batch_mask = remaped_batch == s
            sampleInBatch_local_ind = local_ind[batch_mask]
            sample_offset_logits = offset_logits_u[batch_mask]
            sample_embed_logits = embedding_logits_u[batch_mask]
            sample_predicted_labels = predicted_labels_u[batch_mask]
            sample_batch = remaped_batch[batch_mask]
            #point grouping for offsets
            T1 = time.perf_counter()
            #t_num_clusters, t_pre_ins_labels = self.meanshift_cluster(sample_offset_logits.detach().cpu(), self.opt.bandwidth)
            t_num_clusters, t_pre_ins_labels = self.point_grouping(
                sample_offset_logits.to(self.device),
                sample_predicted_labels.to(self.device),
                sample_batch.to(self.device),
                ignore_labels=self._stuff_classes.to(self.device),
                nsample=200,
                radius=0.06  #0.06 for S3DIS  0.18 for NPM3D
            )
            T2 =time.perf_counter()
            #print('time for offsets clustering:%sms' % ((T2 - T1)*1000))
            global time_for_offsetClustering 
            time_for_offsetClustering += T2 - T1
            #print('total time for offsets clustering:%sms' % ((time_for_offsetClustering)*1000))
            #some points have no instance label (=-1)
            mask_valid = t_pre_ins_labels != -1
            predicted_ins_labels_byOffset[sampleInBatch_local_ind[mask_valid]]=t_pre_ins_labels[mask_valid] + instance_num_offset
            instance_num_offset += t_num_clusters
            #meanshift cluster for embeddings
            T1 = time.perf_counter()
            t_num_clusters2, t_pre_ins_labels2 = self.meanshift_cluster(sample_embed_logits.detach().cpu(), self.opt.bandwidth)
            T2 =time.perf_counter()
            #print('time for embed clustering:%sms' % ((T2 - T1)*1000))
            global time_for_embeddingClustering
            time_for_embeddingClustering += T2 - T1
            #print('total time for embed clustering:%sms' % ((time_for_embeddingClustering)*1000))
            predicted_ins_labels_byEmbed[sampleInBatch_local_ind]=t_pre_ins_labels2 + instance_num_embed
            instance_num_embed += t_num_clusters2
        unique_preInslabels_embed = torch.unique(predicted_ins_labels_byEmbed)
        unique_preInslabels_offset = torch.unique(predicted_ins_labels_byOffset)
        for l in unique_preInslabels_embed:
            if l == -1:
                continue
            label_mask = predicted_ins_labels_byEmbed == l
            local_ind = ind[label_mask]
            embed_clusters.append(local_ind)
        for l in unique_preInslabels_offset:
            if l == -1:
                continue
            label_mask = predicted_ins_labels_byOffset == l
            local_ind = ind[label_mask]
            offset_clusters.append(local_ind)

        #all_clusters = embed_clusters + offset_clusters
        #all_clusters = [c.to(self.device) for c in all_clusters]

        return embed_clusters, offset_clusters, predicted_ins_labels_byEmbed, predicted_ins_labels_byOffset
    
    def meanshift_cluster(self, prediction, bandwidth):
        ms = MeanShift(bandwidth, bin_seeding=True, n_jobs=-1)
        #print ('Mean shift clustering, might take some time ...')
        ms.fit(prediction)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_ 	
        num_clusters = cluster_centers.shape[0]
         
        return num_clusters, torch.from_numpy(labels)
       
    def point_grouping(self, pos, labels, batch, ignore_labels=[], nsample=300, radius=0.03):
        clusters_pos = region_grow(
                pos,
                labels,
                batch,
                ignore_labels=ignore_labels,
                radius=radius,
                nsample=nsample,
                min_cluster_size=32
            )
        #print(clusters_pos)
        predicted_ins_labels_byOffset = -1*torch.ones(labels.size(), dtype=torch.int64)
        
        for i, cluster in enumerate(clusters_pos):
            predicted_ins_labels_byOffset[cluster]=i
            
        num_clusters = len(clusters_pos)
        
        return num_clusters, predicted_ins_labels_byOffset
      
    def compute_loss(self):
        if self._weight_classes is not None:
            self._weight_classes = self._weight_classes.to(self.output.device)

        self.loss = 0

        # Get regularization on weights
        if self.lambda_reg:
            self.loss_reg = self.get_regularization_loss(regularizer_type="l2", lambda_reg=self.lambda_reg)
            self.loss += self.loss_reg

        # Collect internal losses and set them with self and them to self for later tracking
        #if self.lambda_internal_losses:
        #    self.loss += self.collect_internal_losses(lambda_weight=self.lambda_internal_losses)

        # Semantic loss
        self.semantic_loss = F.nll_loss(
            self.output.semantic_logits, (self.labels.y).to(torch.int64), ignore_index=IGNORE_LABEL
        )
        self.loss += self.opt.loss_weights.semantic * self.semantic_loss

        # Offset loss
        self.input.instance_mask = self.input.instance_mask.to(self.device)
        #print(torch.sum(self.input.instance_mask))
        #print(self.input.batch.size())
        if torch.sum(self.input.instance_mask)>1:
            self.input.vote_label = self.input.vote_label.to(self.device)
            offset_losses = offset_loss(
                self.output.offset_logits[self.input.instance_mask],
                self.input.vote_label[self.input.instance_mask],
                torch.sum(self.input.instance_mask),
            )
            for loss_name, loss in offset_losses.items():
                setattr(self, loss_name, loss)
                self.loss += self.opt.loss_weights[loss_name] * loss
                
            # Instance loss
            self.input.instance_labels = self.input.instance_labels.to(self.device)
            discriminative_losses = discriminative_loss(
                self.output.embedding_logits[self.input.instance_mask],
                self.input.instance_labels[self.input.instance_mask],
                self.input.batch[self.input.instance_mask].to(self.device),
                self.opt.mlp_ins.embed_dim
            )
            for loss_name, loss in discriminative_losses.items():
                setattr(self, loss_name, loss)
                if loss_name=="ins_loss":
                    self.loss += self.opt.loss_weights.embedding_loss * loss #discriminative_losses.items()[0]
            
    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss.backward()  # calculate gradients of network G w.r.t. loss_G
        
    def _dump_visuals_fortest(self, epoch):
        if 0<self.opt.vizual_ratio: #random.random() < self.opt.vizual_ratio:
            if not hasattr(self, "visual_count"):
                self.visual_count = 0
            if not os.path.exists("viz"):
                os.mkdir("viz")
            if not os.path.exists("viz/epoch_%i" % (epoch)):
                os.mkdir("viz/epoch_%i" % (epoch))
            #if self.visual_count%10!=0:
            #    return
            print("epoch:{}".format(epoch))
            data_visual = Data(
                pos=self.raw_pos, y=self.input.y, instance_labels=self.input.instance_labels, batch=self.input.batch
            )
            data_visual.semantic_pred = torch.max(self.output.semantic_logits, -1)[1]
            data_visual.embedding = self.output.embedding_logits
            data_visual.vote = self.output.offset_logits
            data_visual.semantic_prob = self.output.semantic_logits
            data_visual.input=self.input.x

            data_visual_fore = Data(
                pos=self.raw_pos[self.input.instance_mask], y=self.input.y[self.input.instance_mask], instance_labels=self.input.instance_labels[self.input.instance_mask], batch=self.input.batch[self.input.instance_mask],
                vote_label=self.labels.vote_label[self.input.instance_mask], pre_ins=self.output.embed_pre[self.input.instance_mask], pre_ins2=self.output.offset_pre[self.input.instance_mask],
                input=self.input.x[self.input.instance_mask]
            )
            data_visual_fore.vote = self.output.offset_logits[self.input.instance_mask]
            data_visual_fore.embedding = self.output.embedding_logits[self.input.instance_mask]
            
            batch_size = torch.unique(data_visual_fore.batch)
            for s in batch_size:
                print(s)
                
                batch_mask_com = data_visual.batch == s
                example_name='example_complete_{:d}'.format(self.visual_count)
                val_name = join("viz", "epoch_"+str(epoch), example_name)
                write_ply(val_name,
                            [data_visual.pos[batch_mask_com].detach().cpu().numpy(), 
                            data_visual.y[batch_mask_com].detach().cpu().numpy().astype('int32'),
                            data_visual.instance_labels[batch_mask_com].detach().cpu().numpy().astype('int32'),
                            data_visual.semantic_prob[batch_mask_com].detach().cpu().numpy(),
                            data_visual.embedding[batch_mask_com].detach().cpu().numpy(),
                            data_visual.vote[batch_mask_com].detach().cpu().numpy().astype('int32'),
                            data_visual.semantic_pred[batch_mask_com].detach().cpu().numpy().astype('int32'),
                            data_visual.input[batch_mask_com].detach().cpu().numpy(),
                            ],
                            #['x', 'y', 'z', 'sem_label', 'ins_label','offset_x', 'offset_y', 'offset_z', 'center_x', 'center_y', 'center_z','pre_ins_embed','pre_ins_offset', 'input_f1', 'input_f2', 'input_f3', 'input_f4', 'input_f5', 'input_f6', 'input_f7'])
                            ['x', 'y', 'z', 'sem_label', 'ins_label',
                            'sem_prob_1', 'sem_prob_2', 'sem_prob_3', 'sem_prob_4', 'sem_prob_5', 'sem_prob_6', 'sem_prob_7','sem_prob_8', 'sem_prob_9',
                            'embed_1', 'embed_2', 'embed_3', 'embed_4', 'embed_5',
                            'offset_x_pre', 'offset_y_pre', 'offset_z_pre','sem_pre_1',
                             'input_f1', 'input_f2', 'input_f3', 'input_f4'])
                
                
                
                batch_mask = data_visual_fore.batch == s
                example_name='example_{:d}'.format(self.visual_count)
                val_name = join("viz", "epoch_"+str(epoch), example_name)
                write_ply(val_name,
                            [data_visual_fore.pos[batch_mask].detach().cpu().numpy(), 
                            data_visual_fore.y[batch_mask].detach().cpu().numpy().astype('int32'),
                            data_visual_fore.instance_labels[batch_mask].detach().cpu().numpy().astype('int32'),
                            data_visual_fore.vote_label[batch_mask].detach().cpu().numpy(),
                            data_visual_fore.pos[batch_mask].detach().cpu().numpy()+data_visual_fore.vote_label[batch_mask].detach().cpu().numpy(),
                            data_visual_fore.pre_ins[batch_mask].detach().cpu().numpy().astype('int32'),
                            data_visual_fore.pre_ins2[batch_mask].detach().cpu().numpy().astype('int32'),
                            data_visual_fore.input[batch_mask].detach().cpu().numpy(),
                            ],
                            #['x', 'y', 'z', 'sem_label', 'ins_label','offset_x', 'offset_y', 'offset_z', 'center_x', 'center_y', 'center_z','pre_ins_embed','pre_ins_offset', 'input_f1', 'input_f2', 'input_f3', 'input_f4', 'input_f5', 'input_f6', 'input_f7'])
                            ['x', 'y', 'z', 'sem_label', 'ins_label','offset_x', 'offset_y', 'offset_z', 'center_x', 'center_y', 'center_z','pre_ins_embed','pre_ins_offset', 'input_f1', 'input_f2', 'input_f3', 'input_f4'])

                example_name='example_ins_{:d}'.format(self.visual_count)
                val_name = join("viz", "epoch_"+str(epoch), example_name)
            
                clustering = MeanShift(bandwidth=self.opt.bandwidth).fit(data_visual_fore.embedding[batch_mask].detach().cpu())                        
                pre_inslab = clustering.labels_
            
                write_ply(val_name,
                            [data_visual_fore.pos[batch_mask].detach().cpu().numpy(), 
                            data_visual_fore.embedding[batch_mask].detach().cpu().numpy(),
                            data_visual_fore.instance_labels[batch_mask].detach().cpu().numpy().astype('int32'),
                            pre_inslab.astype('int32'),
                            data_visual_fore.vote[batch_mask,0].detach().cpu().numpy(), 
                            data_visual_fore.vote_label[batch_mask,0].detach().cpu().numpy(), 
                            data_visual_fore.vote[batch_mask,1].detach().cpu().numpy(), 
                            data_visual_fore.vote_label[batch_mask,1].detach().cpu().numpy(), 
                            data_visual_fore.vote[batch_mask,2].detach().cpu().numpy(), 
                            data_visual_fore.vote_label[batch_mask,2].detach().cpu().numpy() 
                            ],
                            ['x', 'y', 'z', 'emb_feature_1', 'emb_feature_2', 'emb_feature_3', 'emb_feature_4', 'emb_feature_5', 'ins_label', 'pre_ins', 'offset_x', 'gt_offset_x', 'offset_y', 'gt_offset_y', 'offset_z', 'gt_offset_z'])
                example_name = 'example_shiftedCorPre_{:d}'.format(self.visual_count)
                val_name = join("viz", "epoch_"+str(epoch), example_name)

                clustering = MeanShift(bandwidth=self.opt.bandwidth).fit((data_visual_fore.pos[batch_mask].detach().cpu()+data_visual_fore.vote[batch_mask].detach().cpu()))                    
                pre_inslab = clustering.labels_
                write_ply(val_name,
                            [data_visual_fore.pos[batch_mask].detach().cpu().numpy()+data_visual_fore.vote[batch_mask].detach().cpu().numpy(),
                             pre_inslab.astype('int32')], 
                            ['shifted_x_pre', 'shifted_y_pre', 'shifted_z_pre', 'pre_ins'])
                example_name = 'example_shiftedCorGT_{:d}'.format(self.visual_count)
                val_name = join("viz", "epoch_"+str(epoch), example_name)
                write_ply(val_name,
                            [data_visual_fore.pos[batch_mask].detach().cpu().numpy()+data_visual_fore.vote_label[batch_mask].detach().cpu().numpy(),
                             data_visual_fore.instance_labels[batch_mask].detach().cpu().numpy().astype('int32')],  
                            ['shifted_x_gt', 'shifted_y_gt', 'shifted_z_gt', 'ins_label'])
                self.visual_count += 1
    def _dump_visuals(self, epoch):
        if 0<self.opt.vizual_ratio: #random.random() < self.opt.vizual_ratio:
            if not hasattr(self, "visual_count"):
                self.visual_count = 0
            if not os.path.exists("viz"):
                os.mkdir("viz")
            if not os.path.exists("viz/epoch_%i" % (epoch)):
                os.mkdir("viz/epoch_%i" % (epoch))
            #if self.visual_count%10!=0:
            #    return
            print("epoch:{}".format(epoch))
            data_visual = Data(
                pos=self.raw_pos, y=self.input.y, instance_labels=self.input.instance_labels, batch=self.input.batch
            )
            data_visual.semantic_pred = torch.max(self.output.semantic_logits, -1)[1]

            data_visual_fore = Data(
                pos=self.input.pos[self.input.instance_mask], y=self.input.y[self.input.instance_mask], instance_labels=self.input.instance_labels[self.input.instance_mask], batch=self.input.batch[self.input.instance_mask],
                vote_label=self.labels.vote_label[self.input.instance_mask],
                input=self.input.x[self.input.instance_mask]
            )
            data_visual_fore.vote = self.output.offset_logits[self.input.instance_mask]
            data_visual_fore.embedding = self.output.embedding_logits[self.input.instance_mask]
            
            batch_size = torch.unique(data_visual_fore.batch)
            for s in batch_size:
                print(s)
                batch_mask = data_visual_fore.batch == s
                example_name='example_{:d}'.format(self.visual_count)
                val_name = join("viz", "epoch_"+str(epoch), example_name)
                write_ply(val_name,
                            [data_visual_fore.pos[batch_mask].detach().cpu().numpy(), 
                            data_visual_fore.y[batch_mask].detach().cpu().numpy().astype('int32'),
                            data_visual_fore.instance_labels[batch_mask].detach().cpu().numpy().astype('int32'),
                            data_visual_fore.vote_label[batch_mask].detach().cpu().numpy(),
                            data_visual_fore.pos[batch_mask].detach().cpu().numpy()+data_visual_fore.vote_label[batch_mask].detach().cpu().numpy(),
                            data_visual_fore.input[batch_mask].detach().cpu().numpy(),
                            ],
                            ['x', 'y', 'z', 'sem_label', 'ins_label','offset_x', 'offset_y', 'offset_z', 'center_x', 'center_y', 'center_z', 
                            'input_f1', 'input_f2', 'input_f3', 'input_f4'])

                #if s>-1:
                #    self.visual_count += 1
                #    continue
                example_name='example_ins_{:d}'.format(self.visual_count)
                val_name = join("viz", "epoch_"+str(epoch), example_name)
            
                clustering = MeanShift(bandwidth=self.opt.bandwidth).fit(data_visual_fore.embedding[batch_mask].detach().cpu())                        
                pre_inslab = clustering.labels_
            
                write_ply(val_name,
                            [data_visual_fore.pos[batch_mask].detach().cpu().numpy(), 
                            data_visual_fore.embedding[batch_mask].detach().cpu().numpy(),
                            data_visual_fore.instance_labels[batch_mask].detach().cpu().numpy().astype('int32'),
                            pre_inslab.astype('int32'),
                            data_visual_fore.vote[batch_mask,0].detach().cpu().numpy(), 
                            data_visual_fore.vote_label[batch_mask,0].detach().cpu().numpy(), 
                            data_visual_fore.vote[batch_mask,1].detach().cpu().numpy(), 
                            data_visual_fore.vote_label[batch_mask,1].detach().cpu().numpy(), 
                            data_visual_fore.vote[batch_mask,2].detach().cpu().numpy(), 
                            data_visual_fore.vote_label[batch_mask,2].detach().cpu().numpy() 
                            ],
                            ['x', 'y', 'z', 'emb_feature_1', 'emb_feature_2', 'emb_feature_3', 'emb_feature_4', 'emb_feature_5', 'ins_label', 'pre_ins', 'offset_x', 'gt_offset_x', 'offset_y', 'gt_offset_y', 'offset_z', 'gt_offset_z'])
                example_name = 'example_shiftedCorPre_{:d}'.format(self.visual_count)
                val_name = join("viz", "epoch_"+str(epoch), example_name)

                clustering = MeanShift(bandwidth=self.opt.bandwidth).fit((data_visual_fore.pos[batch_mask].detach().cpu()+data_visual_fore.vote[batch_mask].detach().cpu()))                    
                pre_inslab = clustering.labels_
                write_ply(val_name,
                            [data_visual_fore.pos[batch_mask].detach().cpu().numpy()+data_visual_fore.vote[batch_mask].detach().cpu().numpy(),
                             pre_inslab.astype('int32')], 
                            ['shifted_x_pre', 'shifted_y_pre', 'shifted_z_pre', 'pre_ins'])
                example_name = 'example_shiftedCorGT_{:d}'.format(self.visual_count)
                val_name = join("viz", "epoch_"+str(epoch), example_name)
                write_ply(val_name,
                            [data_visual_fore.pos[batch_mask].detach().cpu().numpy()+data_visual_fore.vote_label[batch_mask].detach().cpu().numpy(),
                             data_visual_fore.instance_labels[batch_mask].detach().cpu().numpy().astype('int32')],  
                            ['shifted_x_gt', 'shifted_y_gt', 'shifted_z_gt', 'ins_label'])
                self.visual_count += 1
            