"""Tests for the evaluation module."""

import numpy as np
import pytest

from ctcdetect.evaluate import (
    compute_metrics,
    generate_eval_report,
    generate_eval_html_report,
    plot_roc_pr,
)


def test_compute_metrics_perfect():
    """Perfect predictions should yield AUROC=1.0 and AUPRC=1.0."""
    rng = np.random.default_rng(42)
    n = 100
    y_true = np.array([0] * 50 + [1] * 50)
    # Perfect scores: all negatives get low score, all positives get high score
    y_scores = np.array(
        list(rng.uniform(0.0, 0.4, size=50)) +
        list(rng.uniform(0.6, 1.0, size=50))
    )

    metrics = compute_metrics(y_true, y_scores, threshold=0.5)
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0


def test_compute_metrics_random():
    """Random predictions should yield AUROC approximately 0.5."""
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_scores = rng.uniform(0, 1, size=n)

    metrics = compute_metrics(y_true, y_scores, threshold=0.5)
    # AUROC should be roughly 0.5 for random predictions
    assert 0.3 <= metrics["auroc"] <= 0.7


def test_compute_metrics_all_positive():
    """All positive labels — edge case for specificity."""
    rng = np.random.default_rng(42)
    n = 50
    y_true = np.ones(n, dtype=int)
    y_scores = rng.uniform(0, 1, size=n)

    metrics = compute_metrics(y_true, y_scores, threshold=0.5)
    assert metrics["n_positive"] == n
    assert metrics["n_negative"] == 0
    # All are positive, so TN=0, FP=0
    assert metrics["tn"] == 0
    assert metrics["fp"] == 0
    # TP + FN = n (all actual positives are either predicted positive or negative)
    assert metrics["tp"] + metrics["fn"] == n


def test_generate_eval_report(temp_output_dir):
    """Text eval report file should be created."""
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_scores)

    generate_eval_report(metrics, temp_output_dir)

    report_file = temp_output_dir / "eval_report.txt"
    assert report_file.exists()
    content = report_file.read_text()
    assert "CTC-DETECT EVALUATION REPORT" in content
    assert "AUROC" in content


def test_generate_eval_html_report(temp_output_dir):
    """HTML eval report file should be created."""
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_scores)

    generate_eval_html_report(metrics, temp_output_dir)

    html_file = temp_output_dir / "eval_report.html"
    assert html_file.exists()
    content = html_file.read_text()
    assert "<html" in content.lower()
    assert "AUROC" in content


def test_plot_roc_pr(temp_output_dir):
    """ROC and PR PNG files should be created."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_scores)

    plot_roc_pr(metrics, temp_output_dir)

    assert (temp_output_dir / "roc.png").exists()
    assert (temp_output_dir / "pr.png").exists()
