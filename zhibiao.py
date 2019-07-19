# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:43:28 2019

@author: SuperLee
"""
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#import seaborn as sn
import matplotlib.pyplot as plt


def Kappa(y_true,y_pred,normalize, labels):
    
    '''
    some details : https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    #sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    '''
    
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    cm = confusion_matrix(y_true_f,y_pred_f,labels)
    
    N = len(y_true_f)
    sum_ci = 0
    for i in range(cm.shape[0]):
        ai = np.sum(cm[i,:])
        bi = np.sum(cm[:,i])
        ci = ai * bi
        sum_ci += ci
        
    Pe = sum_ci/(N*N)
    Po = np.sum(cm.diagonal())/ N
    kappa = (Po - Pe)/(1 - Pe)
        
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    color_map = dict(zip([157,208,238,241],['Ship','Sea','Road','Land']))  #for muticlasses
#    color_map = dict(zip([0,255],['Background','Ship']))
        
    f,ax = plt.subplots(figsize =(8,8))
    '''
    color options: cmap
        https://blog.csdn.net/baishuiniyaonulia/article/details/81416649
    '''
    im = ax.imshow(cm, interpolation='nearest', cmap = plt.cm.Wistia)
    ax.figure.colorbar(im, ax=ax)
    
    class_name = [color_map[i] for i in labels]
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels = class_name, yticklabels = class_name,
           title = "Confusion Matrix of Segmention result",
           ylabel='True label',
           xlabel='Predicted label') 
    fmt = '.4f' if normalize else '10d'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    f.savefig(r'H:\ship_segmentation_multiclass\firstjob\IS\first_exp\result\IS_vote_1024_512.jpg',bbox_inches = 'tight',dpi=300)
    error_mat = pd.DataFrame(cm)
    
    return error_mat, kappa
        

def missing_alarm(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    TP = np.sum(y_pred_f * y_true_f)
    miss_alarm = 1 - (TP / np.sum(y_true_f))
    return miss_alarm

def false_alarm(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    TP = np.sum(y_true_f * y_pred_f)
    FP = 1 - (TP / np.sum(y_pred_f))
    return FP

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection ) / (np.sum(y_true_f) + np.sum(y_pred_f))

'''
Add some classical metrics for image segmentation 
'''

def pixel_accuracy(eval_segm, gt_segm):

    '''
    sum_i(nii) / sum_i(t_i)
    '''
    check_size(eval_segm,gt_segm)
    cl, n_cl = extract_classes(gt_segm)
    eval_mask,gt_mask = extract_both_masks(eval_segm,gt_segm,cl,n_cl)
    
    sum_nii = 0
    sum_t_i = 0
    for i,c in enumerate(cl):
        curr_eval_mask = eval_mask[i,:,:]
        curr_gt_mask = gt_mask[i,:,:]
        sum_nii += np.sum(np.logical_and(curr_eval_mask,curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)
    
    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else :
        pixel_accuracy_ = sum_nii / sum_t_i

    return pixel_accuracy_ 

def mean_accuracy(eval_segm, gt_segm):

    '''
    (1/n_cl)*sum_i(n_ii/t_i)
    '''
    check_size(eval_segm,gt_segm)
    cl,n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    accuracy = list([0])*n_cl

    for i,c in enumerate(cl):
        curr_eval_mask = eval_mask[i,:,:]
        curr_gt_mask = gt_mask[i,:,:]
        n_ii = np.sum(np.logical_and(curr_eval_mask,curr_gt_mask))
       
        t_i = np.sum(curr_eval_mask)   #

        if (t_i != 0):
            accuracy[i] = n_ii / t_i
    
    mean_accuracy_ = np.mean(accuracy) 
    return mean_accuracy_ ,accuracy

def mean_IOU(eval_segm,gt_segm):
    '''
    (1/n_cl)*sum_i(n_ii/t_i +sum_j(n_ji)-n_ii)
    '''
    check_size(eval_segm,gt_segm)
    cl, n_cl = union_classes(eval_segm,gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl,n_cl)
    
    IOU = list([0])*n_cl
    for i,c in enumerate(cl):
        curr_eval_mask = eval_mask[i,:,:]
        curr_gt_mask = gt_mask[i,:,:]
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue
        n_ii = np.sum(np.logical_and(curr_eval_mask,curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IOU[i] = n_ii /(t_i + n_ij - n_ii)

    mean_IOU_ = np.sum(IOU) / n_cl_gt
    
    return mean_IOU_ , IOU

def mean_dice(eval_segm,gt_segm):
    '''
    1/n_cl *sum_i(2*n_ii / (t_i + n_ij))
    '''
    check_size(eval_segm,gt_segm)
    cl, n_cl = union_classes(eval_segm,gt_segm)
    _,n_cl_gt = extract_classes(gt_segm)
    eval_mask,gt_mask = extract_both_masks(eval_segm,gt_segm,cl,n_cl)

    Dice = list([0])*n_cl
    for i,c in enumerate(cl):
        curr_eval_mask = eval_mask[i,:,:]
        curr_gt_mask = gt_mask[i,:,:]

        if (np.sum(curr_eval_mask)== 0) or (np.sum(curr_gt_mask) == 0):
            continue
        n_ii = np.sum(np.logical_and(curr_eval_mask,curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        Dice[i] = (2*n_ii)/(t_i + n_ij)

    mean_dice_ = np.sum(Dice) / n_cl_gt
    return mean_dice_ , Dice

def mean_false_alarm(eval_segm,gt_segm):

    '''
    1/n_cl *sum_i(1 - (n_ii)/n_ij)
    '''
    check_size(eval_segm,gt_segm)
    cl, n_cl = union_classes(eval_segm,gt_segm)
    _,n_cl_gt = extract_classes(gt_segm)
    eval_mask,gt_mask = extract_both_masks(eval_segm,gt_segm,cl,n_cl)
    False_alarm = list([0])*n_cl
    for i,c in enumerate(cl):
        curr_eval_mask = eval_mask[i,:,:]
        curr_gt_mask = gt_mask[i,:,:]

        if (np.sum(curr_eval_mask) == 0):
            continue
        n_ii = np.sum(np.logical_and(curr_eval_mask,curr_gt_mask))
        n_ij = np.sum(curr_eval_mask)    #虚警的计算公式仍需要确定

        False_alarm[i] = 1 - (n_ii/n_ij)
    
    mean_false_alarm_ = np.sum(False_alarm) / n_cl_gt
   
    return mean_false_alarm_ , False_alarm

def mean_miss_alarm(eval_segm,gt_segm):
    '''
    1/n_cl *sum_i(1- (n_ii)/t_i)
    '''
    check_size(eval_segm,gt_segm)
    cl, n_cl = union_classes(eval_segm,gt_segm)
    _,n_cl_gt = extract_classes(gt_segm)
    eval_mask,gt_mask = extract_both_masks(eval_segm,gt_segm,cl,n_cl)
    Miss_alarm = list([0])*n_cl
    for i,c in enumerate(cl):
        curr_eval_mask = eval_mask[i,:,:]
        curr_gt_mask = gt_mask[i,:,:]
        if (np.sum(curr_gt_mask) == 0):
            continue
        n_ii = np.sum(np.logical_and(curr_eval_mask,curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        Miss_alarm[i] = 1 - (n_ii / t_i)
    
    mean_miss_alarm_ = np.sum(Miss_alarm) / n_cl_gt
    return mean_miss_alarm_ , Miss_alarm

def frequency_weighted_IOU(eval_segm, gt_segm):
    '''
    sum_k(t_k)^(-1)*sum_i((t_i*n_ii)/(t_i + sum_j(n_ji)- n_ii))
    '''
    check_size(eval_segm, gt_segm)
    cl, n_cl = union_classes(eval_segm,gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm,gt_segm, cl, n_cl)
    frequency_weighted_IOU = list([0])*n_cl

    for i,c in enumerate(cl):
        curr_eval_mask = eval_mask[i,:,:]
        curr_gt_mask = gt_mask[i,:,:]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask,curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IOU[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)
    
    sum_k_t_k = get_pixel_area(eval_segm)
    frequency_weighted_IOU_ = np.sum(frequency_weighted_IOU) / sum_k_t_k

    return frequency_weighted_IOU_ ,frequency_weighted_IOU
  

'''
Auxiliary functions used during evaluation
'''

def get_pixel_area(segm):
    
    return segm.shape[0]*segm.shape[1]

def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise
    return height, width

def check_size(eval_segm,gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):

        raise EvalSegErr("DiffDim: Different dimension of metrics!")


def extract_masks(segm, cl, n_cl):
    h,w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))
    for i,c in enumerate(cl):
        masks[i,:,:] = segm == c

    return masks

def extract_both_masks(eval_segm,gt_segm,cl,n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)
    # np.union1d : compute the uinon of two sets
    cl = np.union1d(eval_cl,gt_cl) 
    n_cl = len(cl)

    return cl, n_cl


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


'''
Exceptions
'''
class EvalSegErr(Exception):

    def __init__(self, value):
        self.value = value
    
    def __str__(self):
        return repr(self.value)


if __name__ == '__main__':

    '''
    The eval_segm and gt_segm should be read as single channel
    '''
    mask_true = cv2.imread(r'H:\ship_segmentation_multiclass\firstjob\IS\first_exp\raw_data/IS_label.tif',2)
    mask_pred = cv2.imread(r'H:\ship_segmentation_multiclass\firstjob\IS\first_exp\result\IS_test_0717_patchsize1024_stride512_vote.tif',2)
    zhibiao_save_path = r'H:\ship_segmentation_multiclass\firstjob\IS\first_exp\result\zhibiao_1024_512_718vote.csv'
    
    cl = np.unique(mask_true).tolist()
    error_mat, kappa = Kappa(y_true = mask_true,y_pred = mask_pred,normalize = True,labels = cl)
    
    t = classification_report(y_true = mask_true.flatten(),y_pred = mask_pred.flatten(),target_names=['Ship','Sea','Land'])
#    t = classification_report(y_true = mask_true.flatten(),y_pred = mask_pred.flatten(),target_names=['Background','Ship'])
    print(t)
    
    Mean_dice, Dice = mean_dice(eval_segm = mask_pred, gt_segm = mask_true)
    Mean_false_alarm, False_alarm = mean_false_alarm(eval_segm = mask_pred, gt_segm = mask_true)
    Mean_miss_alarm, miss_alarm = mean_miss_alarm(eval_segm = mask_pred, gt_segm = mask_true)

    Pixel_Accuracy = pixel_accuracy(eval_segm = mask_pred,gt_segm = mask_true)
    Mean_Accuracy, Accuracy_list = mean_accuracy(eval_segm = mask_pred,gt_segm = mask_true)
    Mean_IOU, IOU_list = mean_IOU(eval_segm = mask_pred,gt_segm = mask_true)
    Frequency_Weighted_IOU, Frequency_Weighted_IOU_list = frequency_weighted_IOU(eval_segm = mask_pred,gt_segm = mask_true)

    # write the evaluatins into csv file
    index_name = ['Ship','Sea','Land']
#    index_name = ['Background','Ship']
    zhibiao = pd.DataFrame(index = index_name)
    zhibiao['Dice'] = Dice
    zhibiao['False_alarm'] = False_alarm
    zhibiao['miss_alarm'] = miss_alarm
    zhibiao['Accuracy'] = Accuracy_list
    zhibiao['IOU'] = IOU_list
    zhibiao['Frequency_Weighted_IOU'] = Frequency_Weighted_IOU
    zhibiao['Kappa'] = kappa
    
    zhibiao.to_csv(zhibiao_save_path)




 


    
    