"""Evaluation utilities for CTC-Detect.

Provides metrics computation, report generation, and plotting
for evaluating CTC detection predictions against ground truth.
"""

from pathlib import Path
from typing import Optional

import numpy as np


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
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        confusion_matrix,
        precision_recall_curve,
        roc_curve,
    )

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


def generate_eval_report(metrics: dict, output_path: Path) -> None:
    """Generate a text evaluation report.

    Parameters
    ----------
    metrics : dict
        Output from ``compute_metrics``.
    output_path : Path
        Directory to write ``eval_report.txt``.
    """
    report_lines = [
        "=" * 60,
        "CTC-DETECT EVALUATION REPORT",
        "=" * 60,
        "",
        f"Total cells evaluated: {metrics['n_total']}",
        f"Ground truth CTCs: {metrics['n_positive']} ({metrics['prevalence']*100:.1f}%)",
        f"Ground truth non-CTCs: {metrics['n_negative']} ({(1-metrics['prevalence'])*100:.1f}%)",
        "",
        f"Threshold: {metrics['threshold']}",
        "",
        "Metrics:",
        f"  AUROC:        {metrics['auroc']:.4f}",
        f"  AUPRC:        {metrics['auprc']:.4f}",
        f"  F1:           {metrics['f1']:.4f}",
        f"  Sensitivity:  {metrics['sensitivity']:.4f}",
        f"  Specificity:  {metrics['specificity']:.4f}",
        f"  PPV:          {metrics['ppv']:.4f}",
        f"  NPV:          {metrics['npv']:.4f}",
        "",
        f"Confusion Matrix (threshold={metrics['threshold']}):",
        "                 Predicted",
        "                 non-CTC    CTC",
        f"  Actual non-CTC  {metrics['tn']:6d}  {metrics['fp']:6d}",
        f"  Actual CTC      {metrics['fn']:6d}  {metrics['tp']:6d}",
        "",
        f"  TP: {metrics['tp']}  FP: {metrics['fp']}",
        f"  FN: {metrics['fn']}  TN: {metrics['tn']}",
        "",
        "=" * 60,
    ]
    report_file = output_path / "eval_report.txt"
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines) + "\n")


def generate_eval_html_report(metrics: dict, output_path: Path) -> None:
    """Generate an HTML evaluation report.

    Parameters
    ----------
    metrics : dict
        Output from ``compute_metrics``.
    output_path : Path
        Directory to write ``eval_report.html``.
    """
    m = metrics
    prev_pct = m['prevalence'] * 100
    non_prev_pct = (1 - m['prevalence']) * 100

    html = (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        '<title>CTC-Detect Evaluation Report</title>\n'
        '<style>\n'
        'body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'
        ' max-width: 800px; margin: 40px auto; padding: 0 20px; color: #333; }\n'
        'h1 { color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 8px; }\n'
        'h2 { color: #444; margin-top: 32px; }\n'
        'table { border-collapse: collapse; width: 100%; margin: 16px 0; }\n'
        'th, td { border: 1px solid #ddd; padding: 10px 14px; text-align: left; }\n'
        'th { background: #f8f9fa; font-weight: 600; }\n'
        'tr:nth-child(even) { background: #fafafa; }\n'
        '.value { font-weight: bold; color: #1a73e8; font-size: 1.2em; }\n'
        '.cm-cell { text-align: center; font-size: 1.3em; font-weight: bold; }\n'
        '.cm-header { background: #e8f0fe; }\n'
        '.img-container { text-align: center; margin: 20px 0; }\n'
        '.img-container img { max-width: 100%; border: 1px solid #ddd; border-radius: 8px; }\n'
        '</style>\n'
        '</head>\n'
        '<body>\n'
        '<h1>CTC-Detect Evaluation Report</h1>\n'
        '\n'
        '<h2>Dataset Summary</h2>\n'
        '<table>\n'
        '  <tr><th>Metric</th><th>Value</th></tr>\n'
        f'  <tr><td>Total cells</td><td class="value">{m["n_total"]}</td></tr>\n'
        f'  <tr><td>Ground truth CTCs</td><td>{m["n_positive"]} ({prev_pct:.1f}%)</td></tr>\n'
        f'  <tr><td>Ground truth non-CTCs</td><td>{m["n_negative"]} ({non_prev_pct:.1f}%)</td></tr>\n'
        f'  <tr><td>Threshold</td><td>{m["threshold"]}</td></tr>\n'
        '</table>\n'
        '\n'
        '<h2>Performance Metrics</h2>\n'
        '<table>\n'
        '  <tr><th>Metric</th><th>Value</th></tr>\n'
        f'  <tr><td>AUROC</td><td class="value">{m["auroc"]:.4f}</td></tr>\n'
        f'  <tr><td>AUPRC</td><td class="value">{m["auprc"]:.4f}</td></tr>\n'
        f'  <tr><td>F1 Score</td><td>{m["f1"]:.4f}</td></tr>\n'
        f'  <tr><td>Sensitivity (Recall)</td><td>{m["sensitivity"]:.4f}</td></tr>\n'
        f'  <tr><td>Specificity</td><td>{m["specificity"]:.4f}</td></tr>\n'
        f'  <tr><td>PPV (Precision)</td><td>{m["ppv"]:.4f}</td></tr>\n'
        f'  <tr><td>NPV</td><td>{m["npv"]:.4f}</td></tr>\n'
        '</table>\n'
        '\n'
        f'<h2>Confusion Matrix (threshold={m["threshold"]})</h2>\n'
        '<table>\n'
        '  <tr><th></th><th class="cm-header">Predicted non-CTC</th><th class="cm-header">Predicted CTC</th></tr>\n'
        f'  <tr><th class="cm-header">Actual non-CTC</th><td class="cm-cell">{m["tn"]}</td><td class="cm-cell">{m["fp"]}</td></tr>\n'
        f'  <tr><th class="cm-header">Actual CTC</th><td class="cm-cell">{m["fn"]}</td><td class="cm-cell">{m["tp"]}</td></tr>\n'
        '</table>\n'
        '\n'
        '<h2>ROC Curve</h2>\n'
        '<div class="img-container">\n'
        '  <img src="roc.png" alt="ROC Curve">\n'
        '</div>\n'
        '\n'
        '<h2>Precision-Recall Curve</h2>\n'
        '<div class="img-container">\n'
        '  <img src="pr.png" alt="Precision-Recall Curve">\n'
        '</div>\n'
        '\n'
        '</body>\n'
        '</html>'
    )

    html_file = output_path / "eval_report.html"
    with open(html_file, "w") as f:
        f.write(html)


def plot_roc_pr(metrics: dict, output_path: Path) -> None:
    """Generate ROC and PR curve PNG plots.

    Parameters
    ----------
    metrics : dict
        Output from ``compute_metrics`` (must contain curve data).
    output_path : Path
        Directory to write ``roc.png`` and ``pr.png``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(metrics["fpr"], metrics["tpr"], "b-", linewidth=2,
            label=f"AUROC = {metrics['auroc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.fill_between(metrics["fpr"], metrics["tpr"], alpha=0.1, color="blue")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    roc_path = output_path / "roc.png"
    fig.savefig(str(roc_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PR curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(metrics["recall_curve"], metrics["precision_curve"],
            "r-", linewidth=2, label=f"AUPRC = {metrics['auprc']:.4f}")
    baseline = metrics["prevalence"]
    ax.axhline(y=baseline, color="k", linestyle="--", alpha=0.5,
               label=f"Baseline = {baseline:.4f}")
    ax.fill_between(metrics["recall_curve"], metrics["precision_curve"],
                    alpha=0.1, color="red")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    pr_path = output_path / "pr.png"
    fig.savefig(str(pr_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
