import torch
from typing import List
from torch_points_kernels import instance_iou
from torch_scatter import scatter
import numba

def offset_loss(pred_offsets, gt_offsets, total_instance_points):
    """ Computes the L1 norm between prediction and ground truth and
    also computes cosine similarity between both vectors.
    see https://arxiv.org/pdf/2004.01658.pdf equations 2 and 3
    """
    pt_diff = pred_offsets - gt_offsets
    pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)
    offset_norm_loss = torch.sum(pt_dist) / (total_instance_points + 1e-6)

    gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # (N), float
    gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
    pred_offsets_norm = torch.norm(pred_offsets, p=2, dim=1)
    pred_offsets_ = pred_offsets / (pred_offsets_norm.unsqueeze(-1) + 1e-8)
    direction_diff = -(gt_offsets_ * pred_offsets_).sum(-1)  # (N)
    offset_dir_loss = torch.sum(direction_diff) / (total_instance_points + 1e-6)

    return {"offset_norm_loss": offset_norm_loss, "offset_dir_loss": offset_dir_loss}

def instance_ious(
    predicted_clusters: List[torch.Tensor],
    cluster_scores: torch.Tensor,
    instance_labels: torch.Tensor,
    batch: torch.Tensor,
    mask_scores_sigmoid: torch.Tensor,
    cal_iou_based_on_mask: bool
    ):

    #assert len(predicted_clusters) == cluster_scores.shape[0]
    
    if not cal_iou_based_on_mask:
        ious = instance_iou(predicted_clusters, instance_labels, batch)
    else:
        assert mask_scores_sigmoid is not None
        if batch is None:
            batch = torch.zeros_like(instance_labels)
        # Gather number of gt instances per batch and size of those instances
        gt_instance_sizes = []
        num_gt_instances = []
        batch_size = batch[-1] + 1
        for s in range(batch_size):
            batch_mask = batch == s
            sample_gt_instances = instance_labels[batch_mask]
            sample_num_gt_instances = torch.max(sample_gt_instances).item()
            num_gt_instances.append(sample_num_gt_instances)
            for instance_id in range(1, sample_num_gt_instances + 1):
                gt_instance_sizes.append(torch.sum(sample_gt_instances == instance_id))
        gt_instance_sizes = torch.stack(gt_instance_sizes)
        num_gt_instances = torch.tensor(num_gt_instances)

        # Instance offset when flatten
        instance_offsets = [0]
        cum_offset = 0
        for instance in predicted_clusters:
            cum_offset += instance.shape[0]
            instance_offsets.append(cum_offset)

        # Compute ious
        proposal_idx = torch.cat(predicted_clusters)

        num_proposed_instances = len(instance_offsets) - 1
        ious = torch.zeros((num_proposed_instances, num_gt_instances.sum()))
        offset_num_gt_instances = torch.cat([torch.tensor([0]), num_gt_instances.cumsum(0)])
        for proposed_instance in range(num_proposed_instances):

            instance = proposal_idx[instance_offsets[proposed_instance] : instance_offsets[proposed_instance + 1]]
            mask_score = mask_scores_sigmoid[instance_offsets[proposed_instance] : instance_offsets[proposed_instance + 1]]
            proposal_total = (mask_score > 0.5).sum()

            sample_idx = batch[instance[0]]
            gt_count_offset = offset_num_gt_instances[sample_idx]
            sample_instance_count = num_gt_instances[sample_idx]
            for instance_id in numba.prange(1, sample_instance_count + 1):
                # intersection = 0
                # for i, idx in enumerate(instance):
                #     if(mask_score[i] > 0.5):
                #         if instance_labels[idx] == instance_id:
                #             intersection += 1
                intersection = torch.logical_and(mask_score > 0.5, instance_labels[instance] == instance_id).sum()
                ious[proposed_instance, gt_count_offset + instance_id - 1] = intersection / float(
                    proposal_total + gt_instance_sizes[gt_count_offset + instance_id - 1] - intersection + 1e-5
                    )
        if instance_labels.is_cuda:
            ious = ious.cuda()
    return ious

def instance_iou_loss(
    ious: torch.Tensor,
    predicted_clusters: List[torch.Tensor],
    cluster_scores: torch.Tensor,
    instance_labels: torch.Tensor,
    batch: torch.Tensor,
    min_iou_threshold=0.25,
    max_iou_threshold=0.75,
):
    """ Loss that promotes higher scores for clusters with higher instance iou,
    see https://arxiv.org/pdf/2004.01658.pdf equation (7)
    """
    assert len(predicted_clusters) == cluster_scores.shape[0]
    ious = ious.max(1)[0]
    lower_mask = ious < min_iou_threshold
    higher_mask = ious > max_iou_threshold
    middle_mask = torch.logical_and(torch.logical_not(lower_mask), torch.logical_not(higher_mask))
    assert torch.sum(lower_mask + higher_mask + middle_mask) == ious.shape[0]
    shat = torch.zeros_like(ious)
    iou_middle = ious[middle_mask]
    shat[higher_mask] = 1
    shat[middle_mask] = (iou_middle - min_iou_threshold) / (max_iou_threshold - min_iou_threshold)
    return torch.nn.functional.binary_cross_entropy(cluster_scores, shat)
    
#soft group paper
def instance_iou_loss2(
    ious: torch.Tensor,
    predicted_clusters: List[torch.Tensor],
    cluster_scores: torch.Tensor,
    instance_labels: torch.Tensor,
    batch: torch.Tensor,
    min_iou_threshold=0.5,
):
    assert len(predicted_clusters) == cluster_scores.shape[0]
    ious = instance_iou(predicted_clusters, instance_labels, batch).max(1)[0]
    lower_mask = ious < min_iou_threshold
    higher_mask = ious >= min_iou_threshold
    assert torch.sum(lower_mask + higher_mask) == ious.shape[0]
    shat = ious[higher_mask]
    cluster_scores_count = cluster_scores[higher_mask]
    if not shat.size()[0]:
        return 0
    return torch.nn.functional.mse_loss(cluster_scores_count, shat)
    

def instance_iou_loss_focal(
    ious: torch.Tensor,
    predicted_clusters: List[torch.Tensor],
    cluster_scores: torch.Tensor,
    instance_labels: torch.Tensor,
    batch: torch.Tensor,
    min_iou_threshold=0.25,
    max_iou_threshold=0.75,
):
    """ Loss that promotes higher scores for clusters with higher instance iou,
    see https://arxiv.org/pdf/2004.01658.pdf equation (7)
    """
    assert len(predicted_clusters) == cluster_scores.shape[0]
    ious = ious.max(1)[0]
    
    return torch.nn.functional.binary_cross_entropy(cluster_scores, ious)
    


def mask_loss(
    ious: torch.Tensor,
    predicted_clusters: List[torch.Tensor],
    mask_scores_sigmoid: torch.Tensor,
    instance_labels: torch.Tensor,
    batch: torch.Tensor
):

    # for each proposal, get max IOU and its corresponding GT instance index
    max_ious, max_iou_idxes = ious.max(1)
    max_iou_idxes += 1

    # count the number of GT instance in each batch (sphere)
    gt_instance_sizes = []
    num_gt_instances = []
    batch_size = batch[-1] + 1
    for s in range(batch_size):
        batch_mask = batch == s
        sample_gt_instances = instance_labels[batch_mask]
        sample_num_gt_instances = torch.max(sample_gt_instances).item()
        num_gt_instances.append(sample_num_gt_instances)
    num_gt_instances = torch.tensor(num_gt_instances)

    # count the number of GT instance by considering all batches as a whole
    offset_num_gt_instances = torch.cat([torch.tensor([0]), num_gt_instances.cumsum(0)])
    if instance_labels.is_cuda:
        offset_num_gt_instances = offset_num_gt_instances.cuda()

    # for each proposal, calculate a mask
    mask_labels = []
    for i, pred_cluster in enumerate(predicted_clusters):
        mask_label =  -torch.ones(pred_cluster.shape[0])
        if max_ious[i] > 0.5:
            max_iou_idx = max_iou_idxes[i] - offset_num_gt_instances[max_iou_idxes[i] > offset_num_gt_instances.cuda()][-1]
            positive_mask = instance_labels[pred_cluster] == max_iou_idx
            mask_label[positive_mask] = 1
            mask_label[~positive_mask] = 0
        mask_labels.append(mask_label)

    mask_labels = torch.cat(mask_labels)
    if instance_labels.is_cuda:
        mask_labels = mask_labels.cuda()
    mask_label_weight = (mask_labels != -1).float()
    mask_labels[mask_labels==-1.] = 0.5

    return torch.nn.functional.binary_cross_entropy(mask_scores_sigmoid, mask_labels, weight=mask_label_weight)

def discriminative_loss(
    embedding_logits: torch.Tensor,
    instance_labels: torch.Tensor,
    batch: torch.Tensor,
    feature_dim,
):
    loss = []
    loss_var = []
    loss_dist = []
    loss_reg = []
    batch_size = torch.unique(batch) #batch[-1] + 1
    for s in batch_size: #range(batch_size):
        batch_mask = batch == s
        sample_gt_instances = instance_labels[batch_mask]
        sample_embed_logits = embedding_logits[batch_mask]
        sample_loss, sample_loss_var, sample_loss_dist, sample_loss_reg = discriminative_loss_single(sample_embed_logits, sample_gt_instances, feature_dim)
        loss.append(sample_loss)
        loss_var.append(sample_loss_var)
        loss_dist.append(sample_loss_dist)
        loss_reg.append(sample_loss_reg)
    loss = torch.stack(loss)
    loss_var = torch.stack(loss_var)
    loss_dist = torch.stack(loss_dist)
    loss_reg = torch.stack(loss_reg)
    return {"ins_loss": torch.mean(loss), "ins_var_loss": torch.mean(loss_var), "ins_dist_loss": torch.mean(loss_dist), "ins_reg_loss": torch.mean(loss_reg)}
    #return torch.mean(loss), torch.mean(loss_var), torch.mean(loss_dist), torch.mean(loss_reg)
    
def discriminative_loss_single(
    prediction,
    correct_label,
    feature_dim,
    delta_v = 0.5,
    delta_d = 1.5,
    param_var = 1.,
    param_dist = 1.,
    param_reg = 0.001,
):

    ''' Discriminative loss for a single prediction/label pair.
    :param prediction: inference of network
    :param correct_label: instance label
    :feature_dim: feature dimension of prediction
    :param delta_v: cutoff variance distance
    :param delta_d: curoff cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    '''
    ### Reshape so pixels are aligned along a vector
    reshaped_pred = torch.reshape(prediction, (-1, feature_dim))
    ### Count instances
    unique_labels, unique_id, counts = torch.unique(correct_label, return_inverse=True, return_counts=True)
    #counts = tf.cast(counts, tf.float32)
    
    num_instances = unique_labels.size()
    #segmented_sum = tf.unsorted_segment_sum(reshaped_pred, unique_id, num_instances)
    #not sure
    segmented_sum = scatter(reshaped_pred, unique_id, dim=0, reduce="sum")

    mu = torch.div(segmented_sum, (torch.reshape(counts, (-1, 1)) + 1e-8 ))
    unique_id_t = unique_id.unsqueeze(1)
    
    unique_id_t = unique_id_t.expand(unique_id_t.size()[0], mu.size()[-1])
    mu_expand = torch.gather(mu, 0, unique_id_t)

    ### Calculate l_var
    #distance = tf.norm(tf.subtract(mu_expand, reshaped_pred), axis=1)
    #tmp_distance = tf.subtract(reshaped_pred, mu_expand)
    tmp_distance = reshaped_pred - mu_expand
    distance = torch.norm(tmp_distance, p=1, dim=1)
    distance = torch.subtract(distance, delta_v)
    distance = torch.clip(distance, min=0.)
    distance = torch.square(distance)
    l_var = scatter(distance, unique_id, dim=0, reduce="sum")
    l_var = torch.div(l_var, counts + 1e-8)
    l_var = torch.sum(l_var)
    l_var = torch.div(l_var, float(num_instances[0]))

    ### Calculate l_dist

    # Get distance for each pair of clusters like this:
    #   mu_1 - mu_1
    #   mu_2 - mu_1
    #   mu_3 - mu_1
    #   mu_1 - mu_2
    #   mu_2 - mu_2
    #   mu_3 - mu_2
    #   mu_1 - mu_3
    #   mu_2 - mu_3
    #   mu_3 - mu_3

    mu_interleaved_rep = mu.repeat(num_instances[0], 1)
    mu_band_rep = mu.repeat(1, num_instances[0])
    mu_band_rep = torch.reshape(mu_band_rep, (num_instances[0] * num_instances[0], feature_dim))

    mu_diff = torch.subtract(mu_band_rep, mu_interleaved_rep)
    # Filter out zeros from same cluster subtraction
    eye = torch.eye(num_instances[0])
    #zero = torch.zeros(1, dtype=torch.float32)
    diff_cluster_mask = torch.eq(eye, 0)
    diff_cluster_mask = torch.reshape(diff_cluster_mask, (-1,))
    mu_diff_bool = mu_diff[diff_cluster_mask]
    #intermediate_tensor = tf.reduce_sum(tf.abs(mu_diff),axis=1)
    #zero_vector = tf.zeros(1, dtype=tf.float32)
    #bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
    #mu_diff_bool = tf.boolean_mask(mu_diff, bool_mask)

    mu_norm = torch.norm(mu_diff_bool, p=1, dim=1)
    mu_norm = torch.subtract(torch.mul(delta_d, 2.0), mu_norm)
    mu_norm = torch.clip(mu_norm, min=0.)
    mu_norm = torch.square(mu_norm)

    l_dist = torch.mean(mu_norm)
    
    if num_instances[0]==1:
        l_dist = torch.tensor(0).cuda()
    ### Calculate l_reg
    l_reg = torch.mean(torch.norm(mu, p=1, dim=1))

    if num_instances[0]==0:
        l_var = torch.tensor(0).cuda()
        l_dist = torch.tensor(0).cuda()
        l_reg = torch.tensor(0).cuda()
    
    param_scale = 1.
    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg

    loss = param_scale * (l_var + l_dist + l_reg)

    #if torch.is_tensor(loss):
    #    loss = loss.item()
    #if torch.is_tensor(l_var):
    #    l_var = l_var.item()
    #if torch.is_tensor(l_dist):
    #    l_dist = l_dist.item()
    #if torch.is_tensor(l_reg):
    #    l_reg = l_reg.item()

    return loss, l_var, l_dist, l_reg