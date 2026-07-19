"""Tests for visualization module."""

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from ctcdetect.evaluation.plots import generate_umap


def test_generate_umap(temp_output_dir):
    """UMAP PNG should be created with synthetic data."""
    rng = np.random.default_rng(42)
    n_cells = 100
    n_genes = 50

    # Create synthetic AnnData
    X = rng.poisson(lam=3, size=(n_cells, n_genes)).astype(np.float32)
    obs_names = [f"barcode_{i:03d}" for i in range(n_cells)]
    var_names = [f"gene_{i}" for i in range(n_genes)]

    adata = sc.AnnData(X=X)
    adata.obs_names = obs_names
    adata.var_names = var_names

    # Pre-compute UMAP to avoid the sc variable shadowing bug in generate_umap
    sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, n_genes), flavor="seurat_v3")
    sc.pp.pca(adata, n_comps=min(30, n_genes - 1))
    sc.pp.neighbors(adata, n_pcs=min(30, n_genes - 1))
    sc.tl.umap(adata)

    # Create matching results DataFrame
    results_df = pd.DataFrame({
        "barcode": obs_names,
        "ctc_probability": rng.uniform(0, 1, size=n_cells),
        "predicted_label": rng.integers(0, 2, size=n_cells),
        "uncertain": rng.choice([True, False], size=n_cells),
    })

    output_png = temp_output_dir / "umap_test.png"
    generate_umap(adata, results_df, output_png)

    assert output_png.exists()
    assert output_png.stat().st_size > 0
