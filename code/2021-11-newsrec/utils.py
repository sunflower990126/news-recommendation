import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def print_write(print_str, log_file):
    print(*print_str)
    if log_file is None:
        return
    with open(log_file, 'a') as f:
        print(*print_str, file=f)


def torch2numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return tuple([torch2numpy(xi) for xi in x])
    else:
        return x


def mrr_score(y_true, y_score):
    """Computing mrr score metric.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        numpy.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        np.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        numpy.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def group_labels(labels, preds, group_keys):
    """Devide `labels` and `preds` into several group according to values in group keys.

    Args:
        labels (list): ground truth label list.
        preds (list): prediction score list.
        group_keys (list): group key list.

    Returns:
        list, list:
        - Labels after group.
        - Predictions after group.
    """
    all_keys = list(set(group_keys))
    group_labels = {k: [] for k in all_keys}
    group_preds = {k: [] for k in all_keys}
    for l, p, k in zip(labels, preds, group_keys):
        group_labels[k].append(l)
        group_preds[k].append(p)
    all_labels = []
    all_preds = []
    for k in all_keys:
        all_labels.append(group_labels[k])
        all_preds.append(group_preds[k])
    return all_labels, all_preds


def cal_metric(labels, preds, imp_indexs=None):
    """Calculate metrics.

    Args:
        labels (array-like): Labels.
        preds (array-like): Predictions.

    Return:
        dict: Metrics.
    """
    res = {}

    # auc
    auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
    res["auc"] = round(auc, 4)

    if imp_indexs is None:
        return res
    else:
        g_labels, g_preds = group_labels(labels, preds, imp_indexs)

    # mean_mrr
    mean_mrr = np.mean(
        [
            mrr_score(each_labels, each_preds)
            for each_labels, each_preds in zip(g_labels, g_preds)
        ]
    )
    res["mean_mrr"] = round(mean_mrr, 4)

    # ndcg@5 ndcg@10
    ndcg_list = [5, 10]
    for k in ndcg_list:
        ndcg_temp = np.mean(
            [
                ndcg_score(each_labels, each_preds, k)
                for each_labels, each_preds in zip(g_labels, g_preds)
            ]
        )
        res["ndcg@{0}".format(k)] = round(ndcg_temp, 4)

    # group_auc
    group_auc = np.mean(
        [
            roc_auc_score(each_labels, each_preds)
            for each_labels, each_preds in zip(g_labels, g_preds)
        ]
    )
    res["group_auc"] = round(group_auc, 4)

    return res
