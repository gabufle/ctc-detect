"""Tests for the detection module."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ctcdetect.core.model import _resolve_model_dir


def test_prepare_adata_csv(tmp_path):
    """Test _prepare_adata with CSV input raises ValueError (only cellranger/h5ad supported)."""
    from ctcdetect.core.detect import _prepare_adata

    csv_path = tmp_path / "test_input.csv"
    n_genes = 50
    n_cells = 20
    rng = np.random.default_rng(42)
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    df = pd.DataFrame(
        rng.poisson(5, size=(n_genes, n_cells)),
        index=gene_names,
        columns=[f"CELL{i:03d}" for i in range(n_cells)],
    )
    df.to_csv(csv_path)

    mock_progress = MagicMock()
    mock_task = MagicMock()

    # _prepare_adata only supports cellranger and h5ad formats
    with pytest.raises(ValueError, match="Unsupported format for detection"):
        _prepare_adata(csv_path, mock_progress, mock_task)


def test_resolve_model_dir_missing(tmp_path, monkeypatch):
    """_resolve_model_dir should SystemExit when no model exists."""
    import ctcdetect.core.model as detect_mod

    fake_checkpoint = tmp_path / "nonexistent_checkpoint"
    fake_finetuned = tmp_path / "nonexistent_finetuned"

    monkeypatch.setattr(detect_mod, "CHECKPOINT_DIR", fake_checkpoint)
    monkeypatch.setattr(detect_mod, "FINETUNED_DIR", fake_finetuned)

    with pytest.raises(SystemExit):
        _resolve_model_dir()


def test_resolve_model_dir_checkpoint_with_weights(tmp_path, monkeypatch):
    """_resolve_model_dir should return CHECKPOINT_DIR when it has weights."""
    import ctcdetect.core.model as detect_mod

    # Create checkpoint dir with a weight file
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "pytorch_model.bin").write_text("fake")

    finetuned = tmp_path / "finetuned"
    finetuned.mkdir()

    monkeypatch.setattr(detect_mod, "CHECKPOINT_DIR", checkpoint)
    monkeypatch.setattr(detect_mod, "FINETUNED_DIR", finetuned)

    result = _resolve_model_dir()
    assert result == checkpoint


def test_resolve_model_dir_finetuned_fallback(tmp_path, monkeypatch):
    """_resolve_model_dir should fall back to FINETUNED_DIR when checkpoint has no weights."""
    import ctcdetect.core.model as detect_mod

    # Checkpoint exists but has no weights
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    finetuned = tmp_path / "finetuned"
    finetuned.mkdir()
    (finetuned / "model.safetensors").write_text("fake")

    monkeypatch.setattr(detect_mod, "CHECKPOINT_DIR", checkpoint)
    monkeypatch.setattr(detect_mod, "FINETUNED_DIR", finetuned)

    result = _resolve_model_dir()
    assert result == finetuned


def test_resolve_model_dir_checkpoint_empty_weights_dir(tmp_path, monkeypatch):
    """_resolve_model_dir should fall back when checkpoint dir exists but is empty."""
    import ctcdetect.core.model as detect_mod

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    finetuned = tmp_path / "finetuned"
    finetuned.mkdir()
    (finetuned / "pytorch_model.bin").write_text("fake")

    monkeypatch.setattr(detect_mod, "CHECKPOINT_DIR", checkpoint)
    monkeypatch.setattr(detect_mod, "FINETUNED_DIR", finetuned)

    result = _resolve_model_dir()
    assert result == finetuned
