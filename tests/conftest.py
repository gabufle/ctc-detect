"""Shared test fixtures for CTC-Detect test suite."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scanpy as sc


@pytest.fixture
def sample_csv_data(tmp_path):
    """Create a temporary CSV file with a cell count matrix (genes x cells).

    Returns a Path to a CSV with 50 genes x 20 cells.
    """
    n_genes = 50
    n_cells = 20
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    cell_barcodes = [f"CELL{i:03d}" for i in range(n_cells)]

    # Create a sparse-ish count matrix
    rng = np.random.default_rng(42)
    data = rng.poisson(lam=2, size=(n_genes, n_cells))

    df = pd.DataFrame(data, index=gene_names, columns=cell_barcodes)
    csv_path = tmp_path / "test_counts.csv"
    df.to_csv(csv_path)
    return csv_path


@pytest.fixture
def sample_h5ad_data(tmp_path):
    """Create a temporary h5ad file with scanpy.

    Returns a Path to an h5ad file with 50 cells x 30 genes.
    """
    n_obs = 50
    n_vars = 30
    rng = np.random.default_rng(42)

    X = rng.poisson(lam=3, size=(n_obs, n_vars)).astype(np.float32)
    obs_names = [f"barcode_{i:03d}" for i in range(n_obs)]
    var_names = [f"gene_{i}" for i in range(n_vars)]

    adata = sc.AnnData(X=X)
    adata.obs_names = obs_names
    adata.var_names = var_names

    h5ad_path = tmp_path / "test_data.h5ad"
    adata.write_h5ad(h5ad_path)
    return h5ad_path


@pytest.fixture
def sample_predictions_csv(tmp_path):
    """Create a temporary predictions CSV file.

    Contains columns: barcode, ctc_probability, predicted_label, uncertain.
    """
    n = 100
    rng = np.random.default_rng(42)

    barcodes = [f"barcode_{i:03d}" for i in range(n)]
    probs = rng.uniform(0, 1, size=n)
    labels = (probs >= 0.5).astype(int)
    uncertain = probs < 0.4  # some are uncertain

    df = pd.DataFrame({
        "barcode": barcodes,
        "ctc_probability": probs,
        "predicted_label": labels,
        "uncertain": uncertain,
    })
    csv_path = tmp_path / "predictions.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_ground_truth_csv(tmp_path):
    """Create a temporary ground truth CSV file.

    Contains columns: barcode, true_label.
    """
    n = 100
    rng = np.random.default_rng(42)

    barcodes = [f"barcode_{i:03d}" for i in range(n)]
    labels = rng.integers(0, 2, size=n)

    df = pd.DataFrame({
        "barcode": barcodes,
        "true_label": labels,
    })
    csv_path = tmp_path / "ground_truth.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    out = tmp_path / "outputs"
    out.mkdir(exist_ok=True)
    return out
