import torch
import numpy as np
from typing import NamedTuple, List

class PanopticResults(NamedTuple):
    semantic_logits: torch.Tensor
    offset_logits: torch.Tensor
    embedding_logits: torch.Tensor
    embed_clusters: List[torch.Tensor] # Each item contains the list of indices in the cluster
    offset_clusters: List[torch.Tensor] # Each item contains the list of indices in the cluster
    embed_pre: torch.Tensor
    offset_pre: torch.Tensor

class PanopticLabels(NamedTuple):
    center_label: torch.Tensor
    y: torch.Tensor
    num_instances: torch.Tensor
    instance_labels: torch.Tensor
    instance_mask: torch.Tensor
    vote_label: torch.Tensor
