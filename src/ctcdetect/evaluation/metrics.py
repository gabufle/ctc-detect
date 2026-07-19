"""Evaluation metrics for CTC-Detect.

Provides metrics computation for evaluating CTC detection predictions against ground truth.
"""


import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute evaluation metrics from ground truth and predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels (0 = non-CTC, 1 = CTC).
    y_scores : np.ndarray
        Predicted CTC probabilities.
    threshold : float
        Binary classification threshold (default 0.5).

    Returns
    -------
    dict
        Dictionary containing:
        - auroc, auprc, f1, sensitivity, specificity, ppv, npv
        - confusion_matrix (2x2 numpy array)
        - tp, fp, tn, fn (int)
        - threshold (float)
        - n_total, n_positive, n_negative (int)
        - prevalence (float)
        - y_true, y_scores (original arrays, for plotting)
    """
    y_pred = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    n_total = len(y_true)
    n_positive = int(y_true.sum())
    n_negative = n_total - n_positive
    prevalence = n_positive / n_total if n_total > 0 else 0.0

    # Compute curve data for plotting
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "confusion_matrix": cm,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": threshold,
        "n_total": n_total,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "prevalence": prevalence,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision_curve": precision,
        "recall_curve": recall,
        "pr_thresholds": pr_thresholds,
        "y_true": y_true,
        "y_scores": y_scores,
    }


__all__ = ["compute_metrics"]
