"""Tests for main.py run and batch commands with mocking."""

from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

import pytest
from typer.testing import CliRunner

from ctcdetect.cli.app import app

runner = CliRunner()


def test_run_command_with_mock_model(tmp_path):
    """Run command with mocked model should succeed."""
    import pandas as pd
    import numpy as np
    csv_path = tmp_path / "input.csv"
    df = pd.DataFrame(
        np.ones((5, 3), dtype=np.float32),
        index=[f"gene_{i}" for i in range(5)],
        columns=[f"cell_{i}" for i in range(3)],
    )
    df.to_csv(csv_path)
    
    output_path = tmp_path / "output"
    
    with patch("ctcdetect.detect.run_detection") as mock_run:
        result = runner.invoke(app, [
            "run",
            "--input", str(csv_path),
            "--output", str(output_path),
        ])
    
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_run_command_with_threshold_and_skip_umap(tmp_path):
    """Run command with --threshold and --skip-umap should pass them through."""
    import pandas as pd
    import numpy as np
    csv_path = tmp_path / "input.csv"
    df = pd.DataFrame(
        np.ones((5, 3), dtype=np.float32),
        index=[f"gene_{i}" for i in range(5)],
        columns=[f"cell_{i}" for i in range(3)],
    )
    df.to_csv(csv_path)
    
    output_path = tmp_path / "output"
    
    with patch("ctcdetect.detect.run_detection") as mock_run:
        result = runner.invoke(app, [
            "run",
            "--input", str(csv_path),
            "--output", str(output_path),
            "--threshold", "0.7",
            "--skip-umap",
        ])
    
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_batch_command_with_mock(tmp_path):
    """Batch command with mocked run_detection should process multiple samples."""
    import pandas as pd
    import numpy as np
    
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    for sample_name in ["sample_A", "sample_B"]:
        sample_dir = input_dir / sample_name
        sample_dir.mkdir()
        csv_path = sample_dir / "data.csv"
        df = pd.DataFrame(
            np.ones((5, 3), dtype=np.float32),
            index=[f"gene_{i}" for i in range(5)],
            columns=[f"cell_{i}" for i in range(3)],
        )
        df.to_csv(csv_path)
    
    output_dir = tmp_path / "output"
    
    with patch("ctcdetect.detect.run_detection") as mock_run:
        result = runner.invoke(app, [
            "batch",
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
        ])
    
    assert result.exit_code == 0
    assert mock_run.call_count == 2


def test_batch_command_no_subdirs(tmp_path):
    """Batch with no subdirectories should exit with error."""
    input_dir = tmp_path / "empty_input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    
    result = runner.invoke(app, [
        "batch",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
    ])
    
    # Should fail with no subdirectories
    assert result.exit_code == 1


def test_batch_command_with_mock_errors(tmp_path):
    """Batch command should report failure if any sample fails."""
    import pandas as pd
    import numpy as np
    
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    for sample_name in ["good_sample", "bad_sample"]:
        sample_dir = input_dir / sample_name
        sample_dir.mkdir()
        csv_path = sample_dir / "data.csv"
        df = pd.DataFrame(
            np.ones((5, 3), dtype=np.float32),
            index=[f"gene_{i}" for i in range(5)],
            columns=[f"cell_{i}" for i in range(3)],
        )
        df.to_csv(csv_path)
    
    output_dir = tmp_path / "output"
    
    with patch("ctcdetect.detect.run_detection") as mock_run:
        mock_run.side_effect = [None, RuntimeError("Model not found")]
        
        result = runner.invoke(app, [
            "batch",
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
        ])
    
    # Should exit with 1 because one sample failed
    assert result.exit_code == 1
    assert mock_run.call_count == 2
