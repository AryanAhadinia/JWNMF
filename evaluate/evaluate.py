import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_curve


def ground_truth_for_eval(gt, ob):
    return gt[ob != 1]


def predicted_scores_for_eval(pr_scores, ob):
    return pr_scores[ob != 1]


def get_best_f1(gt, pr_scores):
    precision, recall, thresholds = precision_recall_curve(gt, pr_scores)
    f1_scores = 2 * recall * precision / (recall + precision)
    f1_scores = np.nan_to_num(f1_scores)
    index = np.argmax(f1_scores)
    return precision[index], recall[index], f1_scores[index], thresholds[index]


def auc_score(gt, pr_scores):
    return roc_auc_score(gt, pr_scores)


def accuracy(gt, pr_binary):
    return accuracy_score(gt, pr_binary)


def sre(gt, pr_binary):
    return np.linalg.norm(gt) / np.linalg.norm(gt - pr_binary)


def mcc(gt, pr_binary):
    return matthews_corrcoef(gt, pr_binary)


def evaluate(gt, pr_scores):
    precision, recall, f1_score, threshold = get_best_f1(gt, pr_scores)
    pr_binary = pr_scores > threshold
    return {
        "best_f1": f1_score,
        "best_threshold": threshold,
        "precision": precision,
        "recall": recall,
        "auc": auc_score(gt, pr_scores),
        "accuracy": accuracy(gt, pr_binary),
        "sre": sre(gt, pr_binary),
        "mcc": mcc(gt, pr_binary),
    }


def evaluate_cascade(gt, pr_scores):
    precision, recall, f1_score, threshold = get_best_f1(gt, pr_scores)
    pr_binary = pr_scores > threshold
    return {
        "best_f1": f1_score,
        "best_threshold": threshold,
        "precision": precision,
        "recall": recall,
        "auc": auc_score(gt, pr_scores),
        "accuracy": accuracy(gt, pr_binary),
    }
