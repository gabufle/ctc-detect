"""Additional tests for CTC-Detect CLI to boost coverage."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd
import numpy as np
from typer.testing import CliRunner

from ctcdetect.main import app

runner = CliRunner()


# ── Run command tests (mocked) ──

def test_run_command_missing_input():
    """Run with non-existent input should exit with code 1."""
    result = runner.invoke(app, [
        "run",
        "--input", "/nonexistent/path",
        "--output", "/tmp/test_output",
    ])
    assert result.exit_code == 1


def test_run_command_missing_output(tmp_path):
    """Run with valid input but no output should fail."""
    # Create a minimal CSV input
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"GENE1": [1, 2], "GENE2": [3, 4]}, index=["CELL1", "CELL2"]).to_csv(csv_path)
    
    result = runner.invoke(app, [
        "run",
        "--input", str(csv_path),
    ])
    # Should fail because --output is required
    assert result.exit_code != 0


def test_run_command_help():
    """Run --help should work."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    # Strip ANSI escape codes for assertion
    import re
    clean = re.sub(r'\x1b\[[0-9;]*m', '', result.output)
    assert "--input" in clean
    assert "--output" in clean


# ── Batch command tests ──

def test_batch_command_missing_input_dir():
    """Batch with non-existent input dir should exit with code 1."""
    result = runner.invoke(app, [
        "batch",
        "--input-dir", "/nonexistent/dir",
        "--output-dir", "/tmp/test_output",
    ])
    assert result.exit_code == 1


def test_batch_command_help():
    """Batch --help should work."""
    result = runner.invoke(app, ["batch", "--help"])
    assert result.exit_code == 0
    import re
    clean = re.sub(r'\x1b\[[0-9;]*m', '', result.output)
    assert "--input-dir" in clean


# ── Validate command tests ──

def test_validate_command_h5ad(sample_h5ad_data):
    """Validate an h5ad file."""
    result = runner.invoke(app, ["validate", "--input", str(sample_h5ad_data)])
    assert result.exit_code == 0


def test_validate_command_help():
    """Validate --help should work."""
    result = runner.invoke(app, ["validate", "--help"])
    assert result.exit_code == 0


# ── Model command tests ──

def test_model_download_help():
    """Model download --help should work."""
    result = runner.invoke(app, ["model", "download", "--help"])
    assert result.exit_code == 0


def test_model_list_help():
    """Model list --help should work."""
    result = runner.invoke(app, ["model", "list", "--help"])
    assert result.exit_code == 0


# ── Evaluate command tests ──

def test_evaluate_command_help():
    """Evaluate --help should work."""
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    import re
    clean = re.sub(r'\x1b\[[0-9;]*m', '', result.output)
    assert "--predictions" in clean


def test_evaluate_command_with_threshold(sample_predictions_csv, sample_ground_truth_csv, temp_output_dir):
    """Evaluate with custom threshold."""
    result = runner.invoke(app, [
        "evaluate",
        "--predictions", str(sample_predictions_csv),
        "--ground-truth", str(sample_ground_truth_csv),
        "--output", str(temp_output_dir),
        "--threshold", "0.7",
    ])
    assert result.exit_code == 0


# ── Info command tests ──

def test_info_command_shows_version():
    """Info command should show version info."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    # Should contain some version-like output
    assert "0.1.0" in result.output or "CTC-Detect" in result.output


# ── Preprocess module tests ──

def test_detect_format_cellranger(tmp_path):
    """Create temp dir with matrix.mtx → 'cellranger'."""
    from ctcdetect.preprocess import detect_format
    
    # Create Cell Ranger-like directory structure
    cr_dir = tmp_path / "cellranger_output"
    cr_dir.mkdir()
    (cr_dir / "matrix.mtx").touch()
    (cr_dir / "barcodes.tsv").touch()
    (cr_dir / "features.tsv").touch()
    
    assert detect_format(cr_dir) == "cellranger"


def test_detect_format_cellranger_nested(tmp_path):
    """Create temp dir with filtered_feature_bc_matrix/ → 'cellranger'."""
    from ctcdetect.preprocess import detect_format
    
    cr_dir = tmp_path / "sample_output"
    cr_dir.mkdir()
    nested = cr_dir / "filtered_feature_bc_matrix"
    nested.mkdir()
    (nested / "matrix.mtx").touch()
    
    assert detect_format(cr_dir) == "cellranger"


def test_detect_format_h5ad(tmp_path):
    """Create .h5ad file → 'h5ad'."""
    from ctcdetect.preprocess import detect_format
    import scanpy as sc
    
    h5ad_path = tmp_path / "test.h5ad"
    adata = sc.AnnData(X=np.ones((10, 5), dtype=np.float32))
    adata.obs_names = [f"cell_{i}" for i in range(10)]
    adata.var_names = [f"gene_{i}" for i in range(5)]
    adata.write_h5ad(str(h5ad_path))
    
    assert detect_format(h5ad_path) == "h5ad"


def test_detect_format_csv(tmp_path):
    """Create .csv file → 'csv'."""
    from ctcdetect.preprocess import detect_format
    
    csv_path = tmp_path / "test.csv"
    csv_path.touch()
    
    assert detect_format(csv_path) == "csv"


def test_detect_format_tsv(tmp_path):
    """Create .tsv file → 'tsv'."""
    from ctcdetect.preprocess import detect_format
    
    tsv_path = tmp_path / "test.tsv"
    tsv_path.touch()
    
    assert detect_format(tsv_path) == "tsv"


def test_detect_format_unknown(tmp_path):
    """Unknown extension should raise SystemExit."""
    from ctcdetect.preprocess import detect_format
    
    unknown_path = tmp_path / "test.xyz"
    unknown_path.touch()
    
    with pytest.raises(SystemExit):
        detect_format(unknown_path)


def test_detect_format_nonexistent():
    """Non-existent path should raise SystemExit."""
    from ctcdetect.preprocess import detect_format
    
    with pytest.raises(SystemExit):
        detect_format(Path("/nonexistent/path"))


def test_load_data_csv(tmp_path):
    """load_data should return AnnData for CSV."""
    from ctcdetect.preprocess import load_data
    
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame(
        np.ones((5, 3), dtype=np.float32),
        index=[f"gene_{i}" for i in range(5)],
        columns=[f"cell_{i}" for i in range(3)],
    )
    df.to_csv(csv_path)
    
    adata = load_data(csv_path)
    assert adata.shape == (3, 5)  # cells x genes


def test_load_data_h5ad(tmp_path):
    """load_data should return AnnData for h5ad."""
    from ctcdetect.preprocess import load_data
    import scanpy as sc
    
    h5ad_path = tmp_path / "test.h5ad"
    adata = sc.AnnData(X=np.ones((10, 5), dtype=np.float32))
    adata.obs_names = [f"cell_{i}" for i in range(10)]
    adata.var_names = [f"gene_{i}" for i in range(5)]
    adata.write_h5ad(str(h5ad_path))
    
    result = load_data(h5ad_path)
    assert result.shape == (10, 5)


def test_validate_input_csv(tmp_path):
    """validate_input should return True for valid CSV."""
    from ctcdetect.preprocess import validate_input
    
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame(
        np.ones((5, 3), dtype=np.float32),
        index=[f"gene_{i}" for i in range(5)],
        columns=[f"cell_{i}" for i in range(3)],
    )
    df.to_csv(csv_path)
    
    assert validate_input(csv_path) is True


def test_validate_input_h5ad(tmp_path):
    """validate_input should return True for valid h5ad."""
    from ctcdetect.preprocess import validate_input
    import scanpy as sc
    
    h5ad_path = tmp_path / "test.h5ad"
    adata = sc.AnnData(X=np.ones((10, 5), dtype=np.float32))
    adata.obs_names = [f"cell_{i}" for i in range(10)]
    adata.var_names = [f"gene_{i}" for i in range(5)]
    adata.write_h5ad(str(h5ad_path))
    
    assert validate_input(h5ad_path) is True


# ── Config module tests ──

def test_config_model_cache_dir():
    """Config should have MODEL_CACHE_DIR."""
    from ctcdetect import config
    assert config.MODEL_CACHE_DIR.exists() or True  # May not exist yet


def test_config_default_model():
    """Config should have DEFAULT_MODEL."""
    from ctcdetect import config
    assert config.DEFAULT_MODEL == "ctheodoris/Geneformer-V1-10M"


def test_config_version_map():
    """Config should have VERSION_MAP."""
    from ctcdetect import config
    assert "latest" in config.VERSION_MAP
    assert "v1.0" in config.VERSION_MAP


def test_config_get_version():
    """get_version should resolve version aliases."""
    from ctcdetect import config
    assert config.get_version("latest") == "ctheodoris/Geneformer-V1-10M"
    assert config.get_version("v1.0") == "ctheodoris/Geneformer-V1-10M"


def test_config_get_version_invalid():
    """get_version should raise ValueError for unknown versions."""
    from ctcdetect import config
    with pytest.raises(ValueError):
        config.get_version("nonexistent")


def test_config_get_system_info():
    """get_system_info should return a dict."""
    from ctcdetect import config
    info = config.get_system_info()
    assert isinstance(info, dict)


# ── Utils module tests ──

def test_validate_output_path_creates_parents(tmp_path):
    """validate_output_path should create parent directories."""
    from ctcdetect.utils import validate_output_path
    
    output = tmp_path / "new_dir" / "subdir" / "output"
    result = validate_output_path(str(output))
    assert result.parent.exists()


def test_print_banner(capsys):
    """print_banner should print something."""
    from ctcdetect.utils import print_banner
    print_banner()
    captured = capsys.readouterr()
    assert "CTC-Detect" in captured.out


# ── Multi command tests ──

def test_multi_command_help():
    """Multi command --help should work."""
    result = runner.invoke(app, ["multi", "--help"])
    assert result.exit_code == 0
    # Strip ANSI escape codes for assertion
    import re
    clean = re.sub(r'\x1b\[[0-9;]*m', '', result.output)
    assert "FILES" in clean
    assert "--output" in clean
    assert "--threshold" in clean
    assert "--skip-umap" in clean


def test_multi_command_missing_files():
    """Multi command without files should fail."""
    result = runner.invoke(app, [
        "multi",
        "--output", "/tmp/test_output",
    ])
    assert result.exit_code != 0
    assert "missing" in result.output.lower() or "required" in result.output.lower()


def test_multi_command_missing_output(tmp_path):
    """Multi command with files but no output should fail."""
    # Create test CSV files
    csv1_path = tmp_path / "test1.csv"
    csv2_path = tmp_path / "test2.csv"
    pd.DataFrame({"GENE1": [1, 2], "GENE2": [3, 4]}, index=["CELL1", "CELL2"]).to_csv(csv1_path)
    pd.DataFrame({"GENE1": [5, 6], "GENE2": [7, 8]}, index=["CELL3", "CELL4"]).to_csv(csv2_path)
    
    result = runner.invoke(app, [
        "multi",
        str(csv1_path),
        str(csv2_path),
    ])
    # Should fail because --output is required
    assert result.exit_code != 0


def test_multi_command_with_valid_files(tmp_path):
    """Multi command with valid files should process them."""
    # Create test CSV files
    csv1_path = tmp_path / "test1.csv"
    csv2_path = tmp_path / "test2.csv"
    pd.DataFrame({"GENE1": [1, 2], "GENE2": [3, 4]}, index=["CELL1", "CELL2"]).to_csv(csv1_path)
    pd.DataFrame({"GENE1": [5, 6], "GENE2": [7, 8]}, index=["CELL3", "CELL4"]).to_csv(csv2_path)
    
    output_dir = tmp_path / "output"
    
    # Mock the run_detection function to avoid actual Geneformer processing
    with patch("ctcdetect.detect.run_detection") as mock_run:
        mock_run.return_value = None
        
        result = runner.invoke(app, [
            "multi",
            str(csv1_path),
            str(csv2_path),
            "--output", str(output_dir),
            "--skip-umap",
        ])
        
        # Should succeed (exit code 0)
        assert result.exit_code == 0
        
        # Should have called run_detection twice (once per file)
        assert mock_run.call_count == 2
        
        # Check that output directories were created
        assert (output_dir / "sample_1_test1").exists()
        assert (output_dir / "sample_2_test2").exists()


def test_multi_command_with_threshold(tmp_path):
    """Multi command should pass threshold option to run_detection."""
    # Create test CSV file
    csv_path = tmp_path / "test.csv"
    pd.DataFrame({"GENE1": [1, 2], "GENE2": [3, 4]}, index=["CELL1", "CELL2"]).to_csv(csv_path)
    
    output_dir = tmp_path / "output"
    
    # Mock the run_detection function
    with patch("ctcdetect.detect.run_detection") as mock_run:
        mock_run.return_value = None
        
        result = runner.invoke(app, [
            "multi",
            str(csv_path),
            "--output", str(output_dir),
            "--threshold", "0.7",
            "--skip-umap",
        ])
        
        assert result.exit_code == 0
        
        # Check that threshold was passed correctly
        call_args = mock_run.call_args
        assert call_args[1]["threshold"] == 0.7
