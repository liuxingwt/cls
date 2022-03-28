"""
Binary classification result evaluation functions.
ROC curve reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
interpolation reference: https://numpy.org/doc/stable/reference/generated/numpy.interp.html
"""
import os 
import glob
import numpy as np
from numpy import interp  
from easydict import EasyDict
from sklearn.metrics import roc_curve, auc, average_precision_score, confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d

list_fpr = [0.1, 0.05, 0.01, 0.001, 0.0001]
list_tpr = [0.9, 0.95, 0.99, 0.995, 0.999]
list_thresh = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def eval_roc2(score_file, threshold=None):
    print('Evaluating Score File: ', score_file)
    labels, scores = get_scores(score_file)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1) 
    result_line = []
    # interpolate tpr
    print("index    TPR          FPR          Thresh")
    for i in range(len(list_tpr)):
        print("%3d"%(i+1),  "%10.4f"%list_tpr[i],  "%14.8f"%interp(list_tpr[i], tpr, fpr),  "%14.8f"%interp(list_tpr[i], tpr, thresholds))
    # interpolate fpr
    print("index    FPR          TPR          Thresh")
    for i in range(len(list_fpr)):
        print("%3d"%(i+1),  "%10.4f"%list_fpr[i],  "%14.8f"%interp(list_fpr[i], fpr, tpr),  "%14.8f"%interp(list_fpr[i], fpr, thresholds))
    # interpolate threshold
    print("index   Thresh        FPR          TPR")
    for i in range(len(list_thresh)):
        print("%3d"%(i+1),  "%10.4f"%list_thresh[i], "%14.8f"%interp(list_thresh[i], thresholds, fpr, period=1), "%14.8f"%interp(list_thresh[i], thresholds, tpr, period=1)) 
    return
    

def get_scores(path_txt, isreverse = False):
    with open(path_txt, 'r', encoding='utf-8') as fid:
        lines = fid.readlines() 
        # print('Samples num: ', len(lines))

    test_scores = []    
    test_labels = []
    for i in range(len(lines)):        
        line = lines[i]
        num_split = line.strip().split(' ')
        if len(num_split)==2:
            image_name = num_split[0]
            score_str = num_split[-1]
            # set label by image namee
            if "fake" in image_name or "spoof" in image_name:
                    label = "0"
            elif "genuine" in image_name or "real" in image_name:
                label = "1"
            else:
                pass
        elif len(num_split)>=3:
            image_name = num_split[0]
            label = num_split[1]
            score_str = num_split[-1]
        else:
            print('error score content!')
            exit(1)

        if label == "0":
            test_labels += [0]
            test_scores += [float(score_str)]
        elif label == "1":
            test_labels += [1]
            test_scores += [float(score_str)]
        else:
            print('error image name!')
            exit(1)
    print('Effective Test Samples:', len(test_scores))
    test_labels = np.array(test_labels)
    test_scores = np.array(test_scores)    
    
    print('Positive  Test Samples:', np.sum(test_labels==1))
    print('Negative  Test Samples:', np.sum(test_labels==0))
    return test_labels, test_scores



def eval_roc(score_file=None, y_trues=None, y_preds=None, threshold=None):
    print('Evaluating Score File: ', score_file)
    if score_file:
        y_trues, y_preds = get_scores(score_file)
    
    metrics = EasyDict()
    fpr, tpr, thresholds = roc_curve(y_trues, y_preds)
    metrics.AUC = auc(fpr, tpr)
    
    metrics.EER = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    metrics.Thre = float(interp1d(fpr, thresholds)(metrics.EER))

    if threshold == 'best':
        _, best_metrics = find_best_threshold(y_trues, y_preds)
        return best_metrics

    elif threshold == 'auto':
        threshold = metrics.Thre

    prediction = (np.array(y_preds) > threshold).astype(int)
    
    res = confusion_matrix(y_trues, prediction)
    TP, FN = res[0, :]
    FP, TN = res[1, :]
    metrics.TPR = float(TP / (TP + FN))
    metrics.FPR = float(FP / (FP + TN))
    metrics.ACC = (TP + TN) / len(y_trues)
    metrics.APCER = float(FP / (TN + FP))
    metrics.BPCER = float(FN / (FN + TP))
    metrics.ACER = (metrics.APCER + metrics.BPCER) / 2
    print("threshold: ", threshold)
    print("metrics.TPR: ", metrics.TPR)
    print("metrics.FPR: ", metrics.FPR)
    print("metrics.ACC: ", metrics.ACC)
    print("metrics.APCER: ", metrics.APCER)
    print("metrics.BPCER: ", metrics.BPCER)
    print("metrics.ACER: ", metrics.ACER)
    return metrics
    
    
def find_best_threshold(y_trues, y_preds):
    best_thre = 0.5
    best_metrics = None
    candidate_thres = list(np.unique(np.sort(y_preds)))
    for thre in candidate_thres:
        metrics = eval_roc(y_trues=y_trues, y_preds=y_preds, threshold=thre)
        if best_metrics is None:
            best_metrics = metrics
            best_thre = thre
        elif metrics.ACER < best_metrics.ACER:
            best_metrics = metrics
            best_thre = thre
    return best_thre, best_metrics
 
    
    
    
if __name__ == '__main__':
    # The line format of score_file is: image_path  label ouput 
    score_file = "/home/projects/bpd_project/src/classifier_v5_fas/testing-results/out_2022_0328_1.txt"
    eval_roc(score_file=score_file, threshold='auto')
