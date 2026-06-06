"""Tests for configuration module."""

import pytest

from ctcdetect.config import (
    MODEL_REGISTRY,
    VERSION_MAP,
    DEFAULT_MODEL,
    get_version,
    get_model_cache_path,
    get_system_info,
    __version__,
)


def test_version():
    """Package version should be a non-empty string."""
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_model_registry():
    """MODEL_REGISTRY should contain expected model aliases."""
    assert "latest" in MODEL_REGISTRY
    assert "v1.0" in MODEL_REGISTRY
    assert "geneformer-base" in MODEL_REGISTRY


def test_get_version_latest():
    """get_version('latest') should return the latest model repo."""
    repo = get_version("latest")
    assert isinstance(repo, str)
    assert "/" in repo


def test_get_version_v1():
    """get_version('v1.0') should return the v1.0 model repo."""
    repo = get_version("v1.0")
    assert isinstance(repo, str)
    assert "/" in repo


def test_get_version_invalid():
    """get_version with an unknown version should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown model version"):
        get_version("nonexistent_version")


def test_get_model_cache_path():
    """get_model_cache_path should return a Path that exists."""
    path = get_model_cache_path("latest")
    assert path.exists()
    assert path.is_dir()


def test_get_model_cache_path_invalid_version():
    """get_model_cache_path with invalid version should fall back to 'latest'."""
    path = get_model_cache_path("nonexistent")
    assert path.exists()


def test_get_system_info():
    """get_system_info should return a dict with expected keys."""
    info = get_system_info()
    assert isinstance(info, dict)
    assert "version" in info
    assert "python" in info
    assert "platform" in info
    assert "pytorch" in info
    assert "cuda_available" in info
    assert "cached_models" in info
    assert "geneformer_installed" in info
    assert "checkpoint_available" in info


def test_default_model():
    """DEFAULT_MODEL should be a non-empty string."""
    assert isinstance(DEFAULT_MODEL, str)
    assert len(DEFAULT_MODEL) > 0
