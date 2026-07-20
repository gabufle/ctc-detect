"""Tests for the CTC-Detect CLI commands."""


from typer.testing import CliRunner

from ctcdetect.cli.app import app

runner = CliRunner()


def test_info_command():
    """Verify info command runs without error."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "Version" in result.output or "CTC-Detect" in result.output


def test_validate_command_csv(sample_csv_data):
    """Validate a CSV file."""
    result = runner.invoke(app, ["validate", "--input", str(sample_csv_data)])
    assert result.exit_code == 0
    assert "✓" in result.output or "good" in result.output.lower()


def test_validate_command_missing_file():
    """Validate with non-existent file should exit with code 1."""
    result = runner.invoke(app, ["validate", "--input", "/nonexistent/path/data.csv"])
    assert result.exit_code == 1


def test_evaluate_command_no_ground_truth(sample_predictions_csv, temp_output_dir):
    """Evaluate with predictions only (no ground truth)."""
    result = runner.invoke(app, [
        "evaluate",
        "--predictions", str(sample_predictions_csv),
        "--output", str(temp_output_dir),
    ])
    assert result.exit_code == 0
    assert (temp_output_dir / "score_distribution.png").exists()


def test_evaluate_command_with_ground_truth(
    sample_predictions_csv, sample_ground_truth_csv, temp_output_dir
):
    """Evaluate with predictions + ground truth."""
    result = runner.invoke(app, [
        "evaluate",
        "--predictions", str(sample_predictions_csv),
        "--ground-truth", str(sample_ground_truth_csv),
        "--output", str(temp_output_dir),
    ])
    assert result.exit_code == 0
    assert (temp_output_dir / "eval_report.txt").exists()
    assert (temp_output_dir / "eval_report.html").exists()
    assert (temp_output_dir / "roc.png").exists()
    assert (temp_output_dir / "pr.png").exists()


def test_evaluate_missing_file():
    """Evaluate with missing predictions file should exit with code 1."""
    result = runner.invoke(app, [
        "evaluate",
        "--predictions", "/nonexistent/predictions.csv",
    ])
    assert result.exit_code == 1


def test_model_download_invalid_version():
    """Download with invalid version should produce an error."""
    result = runner.invoke(app, [
        "model", "download", "--version", "nonexistent_version_xyz",
    ])
    assert result.exit_code == 1


def test_model_list_command():
    """Model list command should run without error."""
    result = runner.invoke(app, ["model", "list"])
    assert result.exit_code == 0


def test_evaluate_missing_columns(temp_output_dir):
    """Evaluate with predictions CSV missing required columns should exit with code 1."""
    import pandas as pd
    # Create a CSV with wrong columns
    bad_csv = temp_output_dir / "bad_predictions.csv"
    pd.DataFrame({"foo": [1, 2], "bar": [3, 4]}).to_csv(bad_csv, index=False)

    result = runner.invoke(app, [
        "evaluate",
        "--predictions", str(bad_csv),
    ])
    assert result.exit_code == 1


def test_evaluate_ground_truth_missing_columns(temp_output_dir):
    """Evaluate with ground truth CSV missing required columns should exit with code 1."""
    import pandas as pd

    # Create valid predictions
    pred_csv = temp_output_dir / "predictions.csv"
    pd.DataFrame({
        "barcode": ["a", "b"],
        "ctc_probability": [0.8, 0.2],
        "predicted_label": [1, 0],
        "uncertain": [False, False],
    }).to_csv(pred_csv, index=False)

    # Create bad ground truth (missing true_label)
    bad_gt = temp_output_dir / "bad_gt.csv"
    pd.DataFrame({"barcode": ["a", "b"], "wrong_col": [1, 0]}).to_csv(bad_gt, index=False)

    result = runner.invoke(app, [
        "evaluate",
        "--predictions", str(pred_csv),
        "--ground-truth", str(bad_gt),
        "--output", str(temp_output_dir),
    ])
    assert result.exit_code == 1


def test_evaluate_no_matching_barcodes(temp_output_dir):
    """Evaluate with non-matching barcodes between pred and gt should exit with code 1."""
    import pandas as pd

    pred_csv = temp_output_dir / "predictions.csv"
    pd.DataFrame({
        "barcode": ["a", "b"],
        "ctc_probability": [0.8, 0.2],
        "predicted_label": [1, 0],
        "uncertain": [False, False],
    }).to_csv(pred_csv, index=False)

    gt_csv = temp_output_dir / "ground_truth.csv"
    pd.DataFrame({
        "barcode": ["x", "y"],
        "true_label": [1, 0],
    }).to_csv(gt_csv, index=False)

    result = runner.invoke(app, [
        "evaluate",
        "--predictions", str(pred_csv),
        "--ground-truth", str(gt_csv),
        "--output", str(temp_output_dir),
    ])
    assert result.exit_code == 1
