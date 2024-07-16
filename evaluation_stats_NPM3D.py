from pathlib import Path
import glob
from collections import defaultdict
from plyfile import PlyData, PlyElement
import numpy as np
from scipy import stats
from torch_points3d.modules.KPConv.plyutils import read_ply

#This file produces stats about the total average F1 score, the average F1 score per forest region, and packs all F1 score within a forest region together
#and save these stats in a file called "Eval_F1_per_region"
if __name__ == '__main__':
    #initialization
    NUM_CLASSES = 10  
    NUM_CLASSES_count = 9 
    # class index for instance segmenatation
    ins_classcount = [3,4,5,7,8,9]  
    # class index for stuff segmentation
    stuff_classcount = [1,2,6]  
    # class index for semantic segmenatation
    sem_classcount = [1,2,3,4,5,6,7,8,9]  
    #class index for semantic classes with gt points
    sem_classcount_have = []
    
    # Initialize...
    LOG_FOUT = open('/path/to/your/output/folder/evaluation_total.txt', 'a')  # save evaluation file with name output_file_name

    def log_string(out_str):
        LOG_FOUT.write(out_str + '\n')
        LOG_FOUT.flush()
        print(out_str)
        
    # acc and macc
    true_positive_classes = np.zeros(NUM_CLASSES)
    positive_classes = np.zeros(NUM_CLASSES)
    gt_classes = np.zeros(NUM_CLASSES)

    # precision & recall
    total_gt_ins = np.zeros(NUM_CLASSES)
    at = 0.5
    tpsins = [[] for itmp in range(NUM_CLASSES)]
    fpsins = [[] for itmp in range(NUM_CLASSES)]
    IoU_Tp = np.zeros(NUM_CLASSES)
    IoU_Mc = np.zeros(NUM_CLASSES)
    # mucov and mwcov
    all_mean_cov = [[] for itmp in range(NUM_CLASSES)]
    all_mean_weighted_cov = [[] for itmp in range(NUM_CLASSES)]
    
    #TO ADAPT: test_sem_list is the list of semantic prediction files 
    test_sem_path = '/path/to/project/PanopticSegForLargeScalePointCloud/outputs/your_output_folder/your_output_folder-PointGroup-PAPER-20230307_152720/eval/2023-03-14_09-35-55'
    test_sem_list = sorted(glob.glob(test_sem_path + '/Semantic_results_forEval*.ply', recursive=False), key=lambda name:int(name[26+len(test_sem_path):-4]))
    #TO ADAPT: test_ins_list is the list of instance prediction files 
    test_ins_list = sorted(glob.glob(test_sem_path + '/Instance_results_forEval*.ply', recursive=False), key=lambda name:int(name[26+len(test_sem_path):-4]))
    #glob.glob(test_sem_path + '/Instance_Embed_results_forEval*.ply', recursive=False)

    for test_sem_i, test_ins_i in zip(test_sem_list, test_ins_list):
        sem_data = PlyData(text=True).read(test_sem_i)
        ins_data = PlyData(text=True).read(test_ins_i)
        
        sem_pre_i = sem_data._get_elements()[0]._get_data()["preds"]+1
        sem_gt_i = sem_data._get_elements()[0]._get_data()["gt"]+1
        ins_pre_i_ori = ins_data._get_elements()[0]._get_data()["preds"]
        ins_gt_i_ori = ins_data._get_elements()[0]._get_data()["gt"]
        
        pred_sem_complete = sem_pre_i
        gt_sem_complete = sem_gt_i
        pred_ins_complete = ins_pre_i_ori
        gt_ins_complete = ins_gt_i_ori
        
        
        idxc = ((gt_sem_complete!=0) & (gt_sem_complete!=1) & (gt_sem_complete!=2) &  (gt_sem_complete!=6)) | ((pred_sem_complete!=0) & (pred_sem_complete!=1) & (pred_sem_complete!=2) &  (pred_sem_complete!=6))
        pred_ins = pred_ins_complete[idxc]
        gt_ins = gt_ins_complete[idxc]
        pred_sem = pred_sem_complete[idxc]
        gt_sem = gt_sem_complete[idxc]

        # pn semantic mIoU
        for j in range(gt_sem_complete.shape[0]):
            gt_l = int(gt_sem_complete[j])
            pred_l = int(pred_sem_complete[j])
            gt_classes[gt_l] += 1
            positive_classes[pred_l] += 1
            true_positive_classes[gt_l] += int(gt_l == pred_l)

        # instance
        un = np.unique(pred_ins)
        pts_in_pred = [[] for itmp in range(NUM_CLASSES)]
        for ig, g in enumerate(un):  # each object in prediction
            if g == -1:
                continue
            tmp = (pred_ins == g)
            sem_seg_i = int(stats.mode(pred_sem[tmp])[0])
            pts_in_pred[sem_seg_i] += [tmp]
    
        un = np.unique(gt_ins)
        pts_in_gt = [[] for itmp in range(NUM_CLASSES)]
        for ig, g in enumerate(un):
            if g == -1:
                continue
            tmp = (gt_ins == g)
            sem_seg_i = int(stats.mode(gt_sem[tmp])[0])
            pts_in_gt[sem_seg_i] += [tmp]
        
        # instance mucov & mwcov
        for i_sem in range(NUM_CLASSES):
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
                num_ins_gt_point = np.sum(ins_gt)
                num_gt_point += num_ins_gt_point
                for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                    union = (ins_pred | ins_gt)
                    intersect = (ins_pred & ins_gt)
                    iou = float(np.sum(intersect)) / np.sum(union)

                    if iou > ovmax:
                        ovmax = iou
                        ipmax = ip

                sum_cov += ovmax
                mean_weighted_cov += ovmax * num_ins_gt_point

            if len(pts_in_gt[i_sem]) != 0:
                mean_cov = sum_cov / len(pts_in_gt[i_sem])
                all_mean_cov[i_sem].append(mean_cov)

                mean_weighted_cov /= num_gt_point
                all_mean_weighted_cov[i_sem].append(mean_weighted_cov)
        
        
        # instance precision & recall
        for i_sem in range(NUM_CLASSES):
            if not pts_in_pred[i_sem]:
                continue
            IoU_Tp_per = 0
            IoU_Mc_per = 0
            tp = [0.] * len(pts_in_pred[i_sem])
            fp = [0.] * len(pts_in_pred[i_sem])
            if pts_in_gt[i_sem]:
                total_gt_ins[i_sem] += len(pts_in_gt[i_sem]) 
            #gtflag = np.zeros(len(pts_in_gt[i_sem]))
            #total_gt_ins[i_sem] += len(pts_in_gt[i_sem])

            for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                ovmax = -1.
                if not pts_in_gt[i_sem]:
                    fp[ip] = 1
                    continue
                for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                    union = (ins_pred | ins_gt)
                    intersect = (ins_pred & ins_gt)
                    iou = float(np.sum(intersect)) / np.sum(union)

                    if iou > ovmax:
                        ovmax = iou
                        #igmax = ig

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
    
    
    # semantic results
    iou_list = []
    for i in range(NUM_CLASSES):
        if gt_classes[i] > 0:
            sem_classcount_have.append(i)
            iou = true_positive_classes[i] / float(gt_classes[i] + positive_classes[i] - true_positive_classes[i])
        else:
            iou = 0.0
        iou_list.append(iou)
        
    set1 = set(sem_classcount)
    set2 = set(sem_classcount_have)
    set3 = set1 & set2
    sem_classcount_final = list(set3)
    
    set1 = set(stuff_classcount)
    set2 = set(sem_classcount_have)
    set3 = set1 & set2
    stuff_classcount_final = list(set3)
        
    log_string('Semantic Segmentation oAcc: {}'.format(sum(true_positive_classes)/float(sum(positive_classes))))
    #log_string('Semantic Segmentation Acc: {}'.format(true_positive_classes / gt_classes))
    log_string('Semantic Segmentation mAcc: {}'.format(np.mean(true_positive_classes[sem_classcount_final] / gt_classes[sem_classcount_final])))
    log_string('Semantic Segmentation IoU: {}'.format(iou_list))
    log_string('Semantic Segmentation mIoU: {}'.format(1.*sum(iou_list)/len(sem_classcount_final)))
    log_string('  ')     
    
    MUCov = np.zeros(NUM_CLASSES)
    MWCov = np.zeros(NUM_CLASSES)
    for i_sem in range(NUM_CLASSES):
        MUCov[i_sem] = np.mean(all_mean_cov[i_sem])
        MWCov[i_sem] = np.mean(all_mean_weighted_cov[i_sem])

    precision = np.zeros(NUM_CLASSES)
    recall = np.zeros(NUM_CLASSES)
    RQ = np.zeros(NUM_CLASSES)
    SQ = np.zeros(NUM_CLASSES)
    PQ = np.zeros(NUM_CLASSES)
    PQStar = np.zeros(NUM_CLASSES)
    set1 = set(ins_classcount)
    set2 = set(sem_classcount_have)
    set3 = set1 & set2
    ins_classcount_final = list(set3)
    
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
        if (total_gt_ins[i_sem])==0:
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

    ############################################
    ######  RQ, SQ, PQ, PQ_star for stuff ######
    ############################################
    for i_sem in stuff_classcount:
        if iou_list[i_sem] >= 0.5:
            RQ[i_sem] = 1
            SQ[i_sem] = iou_list[i_sem]
        else:
            RQ[i_sem] = 0
            SQ[i_sem] = 0
        PQ[i_sem] = SQ[i_sem] * RQ[i_sem]
        PQStar[i_sem] = iou_list[i_sem]

    if np.mean(precision[ins_classcount_final])+np.mean(recall[ins_classcount_final])==0:
        F1_score = 0.0
    else:
        F1_score = (2*np.mean(precision[ins_classcount_final])*np.mean(recall[ins_classcount_final]))/(np.mean(precision[ins_classcount_final])+np.mean(recall[ins_classcount_final]))
   
    # instance results
    log_string('Instance Segmentation:')
    log_string('Instance Segmentation MUCov: {}'.format(MUCov[ins_classcount]))
    log_string('Instance Segmentation mMUCov: {}'.format(np.mean(MUCov[ins_classcount_final])))
    log_string('Instance Segmentation MWCov: {}'.format(MWCov[ins_classcount]))
    log_string('Instance Segmentation mMWCov: {}'.format(np.mean(MWCov[ins_classcount_final])))
    log_string('Instance Segmentation Precision: {}'.format(precision[ins_classcount]))
    log_string('Instance Segmentation mPrecision: {}'.format(np.mean(precision[ins_classcount_final])))
    log_string('Instance Segmentation Recall: {}'.format(recall[ins_classcount]))
    log_string('Instance Segmentation mRecall: {}'.format(np.mean(recall[ins_classcount_final])))
    log_string('Instance Segmentation F1 score: {}'.format(F1_score))
    log_string('Instance Segmentation RQ: {}'.format(RQ[sem_classcount]))
    log_string('Instance Segmentation meanRQ: {}'.format(np.mean(RQ[sem_classcount_final])))
    log_string('Instance Segmentation SQ: {}'.format(SQ[sem_classcount]))
    log_string('Instance Segmentation meanSQ: {}'.format(np.mean(SQ[sem_classcount_final])))
    log_string('Instance Segmentation PQ: {}'.format(PQ[sem_classcount]))
    log_string('Instance Segmentation meanPQ: {}'.format(np.mean(PQ[sem_classcount_final])))
    log_string('Instance Segmentation PQ star: {}'.format(PQStar[sem_classcount]))
    log_string('Instance Segmentation mean PQ star: {}'.format(np.mean(PQStar[sem_classcount_final])))
    log_string('Instance Segmentation RQ (things): {}'.format(RQ[ins_classcount]))
    log_string('Instance Segmentation meanRQ (things): {}'.format(np.mean(RQ[ins_classcount_final])))
    log_string('Instance Segmentation SQ (things): {}'.format(SQ[ins_classcount]))
    log_string('Instance Segmentation meanSQ (things): {}'.format(np.mean(SQ[ins_classcount_final])))
    log_string('Instance Segmentation PQ (things): {}'.format(PQ[ins_classcount]))
    log_string('Instance Segmentation meanPQ (things): {}'.format(np.mean(PQ[ins_classcount_final])))
    log_string('Instance Segmentation RQ (stuff): {}'.format(RQ[stuff_classcount]))
    log_string('Instance Segmentation meanRQ (stuff): {}'.format(np.mean(RQ[stuff_classcount_final])))
    log_string('Instance Segmentation SQ (stuff): {}'.format(SQ[stuff_classcount]))
    log_string('Instance Segmentation meanSQ (stuff): {}'.format(np.mean(SQ[stuff_classcount_final])))
    log_string('Instance Segmentation PQ (stuff): {}'.format(PQ[stuff_classcount]))
    log_string('Instance Segmentation meanPQ (stuff): {}'.format(np.mean(PQ[stuff_classcount_final])))