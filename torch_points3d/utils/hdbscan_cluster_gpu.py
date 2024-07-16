import numpy as np
from sklearn.preprocessing import normalize
from cuml.cluster import HDBSCAN
import torch
import cupy as cp
import cuml
import time
#from hdbscan import HDBSCAN

def hdbscan_cluster(prediction):
    clusterer = HDBSCAN(min_samples=5, verbose=0) #core_dist_n_jobs=1)
    ary = cp.asarray(prediction.data)
    clusterer.fit(ary)
    #labels = clusterer.fit_predict(ary)
    labels = clusterer.labels_
    
    labels = torch.as_tensor(labels, device='cuda')
    num_clusters = clusterer.labels_.max()
        
    return num_clusters, labels
        
def cluster_loop(embed_logits_logits_u, unique_in_batch, label_batch, local_ind, low, high, loop_num):
    #torch.save(embed_logits_logits_u, 'embed_logits_logits_u.pt')
    #torch.save(unique_in_batch, 'unique_in_batch.pt')
    #torch.save(label_batch, 'label_batch.pt')
    #torch.save(local_ind, 'local_ind.pt')
    t = time.time()
    all_clusters = []
    cluster_type = []
    pick_num  = np.random.randint(low=low,high=high+1,size=loop_num)
    for loop_i in range(loop_num):
        #feature_choose = np.random.choice(embed_logits_logits_u.shape[-1], pick_num[loop_i], replace=False)
        feature_choose = torch.multinomial(torch.ones(embed_logits_logits_u.shape[-1]), pick_num[loop_i], replacement=False, out=None)
        #print('type %d feature choose:' % (loop_i))
        #print(feature_choose)
        embed_logits_logits_typei = embed_logits_logits_u[:,feature_choose]
        for s in unique_in_batch:
            batch_mask = label_batch == s
            if torch.sum(batch_mask)>5:
                sampleInBatch_local_ind = local_ind[batch_mask]
                sample_embed_logits = embed_logits_logits_typei[batch_mask]
                #sample_predicted_labels = predicted_labels_u[batch_mask]
                #meanshift cluster for embeddings
                #num_clusters_embed, pre_ins_labels_embed = self.meanshift_cluster(sample_embed_logits.detach().cpu(), self.opt.bandwidth)
                #hdbscan
                sample_embed_logits = torch.nn.functional.normalize(sample_embed_logits, dim=0) #normalize(sample_embed_logits, axis=0)
                num_clusters_embed, pre_ins_labels_embed = hdbscan_cluster(sample_embed_logits)
                #if return label is -1
                unique_preInslabels = torch.unique(pre_ins_labels_embed)
                for l in unique_preInslabels:
                    if l == -1:
                        continue
                    label_mask_l = pre_ins_labels_embed == l
                    all_clusters.append(sampleInBatch_local_ind[label_mask_l])
                    cluster_type.append(loop_i)
    print("total time",time.time()-t)
    return all_clusters, cluster_type

def cluster_single(embed_logits_logits_u, unique_in_batch, label_batch, local_ind, type):
    
    all_clusters = []
    cluster_type = []
    for s in unique_in_batch:
        batch_mask = label_batch == s
        if torch.sum(batch_mask)>3:
            sampleInBatch_local_ind = local_ind[batch_mask]
            sample_embed_logits = embed_logits_logits_u[batch_mask]
            #hdbscan
            sample_embed_logits = torch.nn.functional.normalize(sample_embed_logits, dim=0)
            #normalize(sample_embed_logits, axis=0)
            num_clusters_embed, pre_ins_labels_embed = hdbscan_cluster(sample_embed_logits)
            unique_preInslabels = torch.unique(pre_ins_labels_embed)
            for l in unique_preInslabels:
                if l == -1:
                    continue
                label_mask_l = pre_ins_labels_embed == l
                all_clusters.append(sampleInBatch_local_ind[label_mask_l])
                cluster_type.append(type)
    return all_clusters, cluster_type

