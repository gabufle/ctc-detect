"""Tests for standalone eval_report scaling helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import numpy as np
import pandas as pd

from ctcdetect.scripts.eval_report import choose_umap_indices, collect_group_confusion


def test_choose_umap_indices_caps_rows_and_keeps_error_strata():
    """Large UMAP inputs should be sampled without dropping error regions."""
    y_true = np.array([0] * 900 + [1] * 100)
    y_pred = y_true.copy()
    y_pred[10:20] = 1
    y_pred[930:940] = 0

    rng = np.random.default_rng(42)
    idx = choose_umap_indices(y_true, y_pred, max_points=120, rng=rng)

    assert len(idx) <= 120
    sampled_pairs = set(zip(y_true[idx], y_pred[idx], strict=True))
    assert (0, 1) in sampled_pairs
    assert (1, 0) in sampled_pairs


def test_collect_group_confusion_caps_plots_but_keeps_all_group_metrics(tmp_path):
    """Many groups should not create unbounded PNGs, but JSON metrics stay complete."""
    rows = []
    for group_idx in range(8):
        for cell_idx in range(6):
            true_label = cell_idx % 2
            rows.append(
                {
                    "barcode": f"cell_{group_idx}_{cell_idx}",
                    "true_label": true_label,
                    "predicted_label": true_label,
                    "cancer_type": f"group_{group_idx}",
                }
            )
    merged = pd.DataFrame(rows)

    group_metrics = collect_group_confusion(
        merged,
        "cancer_type",
        tmp_path,
        max_group_plots=3,
        min_group_size=2,
    )

    plotted = [metrics for metrics in group_metrics.values() if metrics["plot"]]
    pngs = sorted((tmp_path / "confusion_by_group").glob("*.png"))

    assert len(group_metrics) == 8
    assert len(plotted) == 3
    assert len(pngs) == 3
    assert all("raw" in metrics for metrics in group_metrics.values())
