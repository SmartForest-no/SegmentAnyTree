# Author Mengyang Zhao <Mengyang.Zhao@tufts.edu>

import math
import operator

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import exp, sqrt

def euc_batch(a, b):
    result = sqrt(((b[None,:] - a[:,None]) ** 2).sum(2))
    #pdist = torch.nn.PairwiseDistance(p=2)
    #result = pdist(a, b)
    return result

    #num = a@b.T
    #denom = torch.norm(a, dim=1).reshape(-1, 1) * torch.norm(b, dim=1)
    #return num / denom

def get_weight(sim, bandwidth):

    thr = 1-bandwidth
    #max = torch.tensor(1.0e+10).double().cuda()
    max = torch.tensor(1.0).double().cuda()
    min = torch.tensor(0.0).double().cuda()
    #dis=torch.where(sim>thr, 1-sim, max)
    dis=torch.where(sim>thr, max, min)

    return dis

def gaussian(dist, bandwidth):
    return exp(-0.5 * ((dist / bandwidth))**2) / (bandwidth * math.sqrt(2 * math.pi))

def meanshift_torch(data, seed , bandwidth, max_iter=300):

    stop_thresh = 1e-3 * bandwidth
    iter=0

    X = torch.from_numpy(np.copy(data)).double().cuda()
    S = torch.from_numpy(np.copy(seed)).double().cuda()
    B = torch.tensor(bandwidth).double().cuda()
    
    while True:
        #cosine = cos_batch(S, X)

        weight = gaussian(euc_batch(S, X),B)

        #torch.where(distances>(1-bandwidth))
        #weight = gaussian(distances, B)
        num = (weight[:, :, None] * X).sum(dim=1)
        S_old = S
        S = num / weight.sum(1)[:, None]
        #cosine2 = torch.norm(S - S_old, dim=1).mean()
        iter+=1

        if (torch.norm(S - S_old, dim=1).mean() < stop_thresh or iter == max_iter):
            break
    
    p_num=[]
    for line in weight:
        p_num.append(line[line==1].size()[0])

    my_mean = S.cpu().numpy()

    return my_mean, p_num