"""Training package for CTC-Detect.

Provides fine-tuning pipeline for Geneformer with LoRA.
"""

from ctcdetect.training.trainer import (
    create_lora_config,
    load_base_model,
    train_model,
)

__all__ = [
    "create_lora_config",
    "load_base_model",
    "train_model",
]
