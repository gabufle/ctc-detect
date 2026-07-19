"""System information utilities for CTC-Detect CLI."""

import platform
import sys
from importlib.metadata import version as _pkg_version

from ctcdetect.config.paths import (
    CHECKPOINT_DIR,
    GENEFORMER_DIR,
    MODEL_CACHE_DIR,
)
from ctcdetect.config.registry import MODEL_REGISTRY


def get_system_info() -> dict:
    """Return a dict of system information for the ``info`` command."""

    info = {
        "version": "0.1.0",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch": "not installed",
        "cuda_available": False,
        "cuda_version": None,
        "gpu": None,
    }

    try:
        info["version"] = _pkg_version("ctc-detect")
    except Exception:
        pass

    try:
        import torch

        info["pytorch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "unknown"
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    available_models = []
    for alias, _meta in MODEL_REGISTRY.items():
        cache_path = MODEL_CACHE_DIR / alias
        if cache_path.exists() and any(cache_path.iterdir()):
            available_models.append(alias)
    info["cached_models"] = available_models

    info["geneformer_installed"] = GENEFORMER_DIR.exists()
    info["geneformer_path"] = str(GENEFORMER_DIR)

    has_checkpoint = (
        CHECKPOINT_DIR.exists()
        and (
            (CHECKPOINT_DIR / "pytorch_model.bin").exists()
            or (CHECKPOINT_DIR / "model.safetensors").exists()
            or (CHECKPOINT_DIR / "adapter_model.bin").exists()
            or (CHECKPOINT_DIR / "adapter_model.safetensors").exists()
        )
    )
    info["checkpoint_available"] = has_checkpoint
    info["checkpoint_path"] = str(CHECKPOINT_DIR)

    try:
        import shutil

        disk = shutil.disk_usage(str(MODEL_CACHE_DIR))
        info["disk_total_gb"] = round(disk.total / (1024**3), 1)
        info["disk_used_gb"] = round(disk.used / (1024**3), 1)
        info["disk_free_gb"] = round(disk.free / (1024**3), 1)
    except OSError:
        info["disk_total_gb"] = None
        info["disk_used_gb"] = None
        info["disk_free_gb"] = None

    return info
