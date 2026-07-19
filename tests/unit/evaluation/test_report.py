"""Tests for report generation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ctcdetect.evaluation.reports import generate_report, generate_html_report


@pytest.fixture
def predictions_csv(tmp_path):
    """Create a predictions CSV for report generation tests."""
    n = 50
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "barcode": [f"barcode_{i:03d}" for i in range(n)],
        "ctc_probability": rng.uniform(0, 1, size=n),
        "predicted_label": rng.integers(0, 2, size=n),
        "uncertain": rng.choice([True, False], size=n),
    })
    csv_path = tmp_path / "predictions.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_generate_report(predictions_csv, temp_output_dir):
    """Text summary file should be created with correct content."""
    generate_report(predictions_csv, temp_output_dir, threshold=0.5)

    report_file = temp_output_dir / "summary.txt"
    assert report_file.exists()

    content = report_file.read_text()
    assert "CTC-DETECT SUMMARY REPORT" in content
    assert "Total cells analyzed" in content
    assert "CTC Probability Score Statistics" in content


def test_generate_html_report(predictions_csv, temp_output_dir):
    """HTML file should be created with correct content."""
    generate_html_report(predictions_csv, temp_output_dir, threshold=0.5)

    html_file = temp_output_dir / "summary.html"
    assert html_file.exists()

    content = html_file.read_text()
    assert "<html" in content.lower()
    assert "CTC-Detect Summary Report" in content
    assert "Total Cells" in content
