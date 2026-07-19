"""Evaluation package for CTC-Detect.

Provides metrics computation, report generation, and plotting
for evaluating CTC detection predictions against ground truth.
"""

from ctcdetect.evaluation.metrics import compute_metrics
from ctcdetect.evaluation.plots import (
    generate_umap,
    plot_roc_pr,
    plot_score_distribution,
)
from ctcdetect.evaluation.reports import (
    generate_eval_html_report,
    generate_eval_report,
    generate_html_report,
    generate_report,
)

__all__ = [
    # Metrics
    "compute_metrics",
    # Plots
    "generate_umap",
    "plot_roc_pr",
    "plot_score_distribution",
    # Reports
    "generate_report",
    "generate_html_report",
    "generate_eval_report",
    "generate_eval_html_report",
]
