"""Tests for input format detection and data loading."""


import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from ctcdetect.core.preprocess import SUPPORTED_FORMATS, detect_format, load_data, validate_input
from ctcdetect.exceptions import InputError


def test_detect_format_cellranger(tmp_path):
    """A directory containing matrix.mtx should be detected as cellranger."""
    (tmp_path / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n"
        "3 2 2\n"
        "1 1 5\n"
        "2 2 3\n"
    )
    fmt = detect_format(tmp_path)
    assert fmt == "cellranger"


def test_detect_format_cellranger_filtered_dir(tmp_path):
    """A directory containing filtered_feature_bc_matrix/ should be detected as cellranger."""
    filtered = tmp_path / "filtered_feature_bc_matrix"
    filtered.mkdir()
    (filtered / "matrix.mtx").write_text("fake")
    fmt = detect_format(tmp_path)
    assert fmt == "cellranger"


def test_detect_format_cellranger_mtx_gz(tmp_path):
    """A directory containing matrix.mtx.gz should be detected as cellranger."""
    (tmp_path / "matrix.mtx.gz").write_text("fake gz")
    fmt = detect_format(tmp_path)
    assert fmt == "cellranger"


def test_detect_format_h5ad(tmp_path):
    """An .h5ad file should be detected as h5ad."""
    h5ad_path = tmp_path / "data.h5ad"
    adata = sc.AnnData(X=np.ones((5, 3), dtype=np.float32))
    adata.obs_names = [f"c{i}" for i in range(5)]
    adata.var_names = [f"g{i}" for i in range(3)]
    adata.write_h5ad(str(h5ad_path))

    fmt = detect_format(h5ad_path)
    assert fmt == "h5ad"


def test_detect_format_csv(tmp_path):
    """A .csv file should be detected as csv."""
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        np.ones((5, 3)),
        index=[f"gene{i}" for i in range(5)],
        columns=[f"cell{i}" for i in range(3)],
    )
    df.to_csv(csv_path)

    fmt = detect_format(csv_path)
    assert fmt == "csv"


def test_detect_format_tsv(tmp_path):
    """A .tsv file should be detected as csv."""
    tsv_path = tmp_path / "data.tsv"
    tsv_path.write_text("gene1\tgene2\ncell1\t1.0\t2.0\n")

    fmt = detect_format(tsv_path)
    assert fmt == "tsv"


def test_detect_format_unknown(tmp_path):
    """An unknown file extension should raise InputError."""
    weird_path = tmp_path / "data.xyz"
    weird_path.write_text("not a real file")

    with pytest.raises(InputError):
        detect_format(weird_path)


def test_detect_format_unknown_dir(tmp_path):
    """A directory with no recognizable files should raise InputError."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(InputError):
        detect_format(empty_dir)


def test_validate_input_csv(tmp_path):
    """validate_input should return True for a valid CSV."""
    csv_path = tmp_path / "valid.csv"
    df = pd.DataFrame(
        np.random.default_rng(42).poisson(2, size=(20, 10)),
        index=[f"GENE{i}" for i in range(20)],
        columns=[f"CELL{i:03d}" for i in range(10)],
    )
    df.to_csv(csv_path)

    result = validate_input(csv_path)
    assert result is True


def test_load_data_csv(tmp_path):
    """load_data should return AnnData with correct shape for CSV."""
    csv_path = tmp_path / "matrix.csv"
    n_genes = 30
    n_cells = 15
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        rng.poisson(2, size=(n_genes, n_cells)),
        index=[f"GENE{i}" for i in range(n_genes)],
        columns=[f"CELL{i:03d}" for i in range(n_cells)],
    )
    df.to_csv(csv_path)

    adata = load_data(csv_path)
    # CSV is genes x cells, AnnData should be cells x genes
    assert adata.shape == (n_cells, n_genes)


def test_load_data_h5ad(tmp_path):
    """load_data should return AnnData for h5ad files."""
    h5ad_path = tmp_path / "test.h5ad"
    n_obs = 25
    n_vars = 10
    adata = sc.AnnData(
        X=np.random.default_rng(42).poisson(3, size=(n_obs, n_vars)).astype(np.float32)
    )
    adata.obs_names = [f"bc_{i}" for i in range(n_obs)]
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]
    adata.write_h5ad(str(h5ad_path))

    result = load_data(h5ad_path)
    assert isinstance(result, sc.AnnData)
    assert result.shape == (n_obs, n_vars)


def test_supported_formats_dict():
    """SUPPORTED_FORMATS should contain the expected format keys."""
    assert "cellranger" in SUPPORTED_FORMATS
    assert "h5ad" in SUPPORTED_FORMATS
    assert "csv" in SUPPORTED_FORMATS
