"""Plugin/extension interface for CTC-Detect model backends.

Allows swapping Geneformer for other models (scGPT, scFoundation, custom models)
without modifying core pipeline code.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch


class ModelBackend(ABC):
    """Abstract base class for model backends.

    Implement this interface to add a new model backend (e.g., scGPT, scFoundation).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this backend (e.g., 'geneformer', 'scgpt')."""
        pass

    @property
    @abstractmethod
    def supported_tasks(self) -> list[str]:
        """List of supported tasks (e.g., ['classification', 'embedding'])."""
        pass

    @abstractmethod
    def load(self, model_dir: Path, device: torch.device | None = None) -> Any:
        """Load model from directory.

        Args:
            model_dir: Path to model directory.
            device: Device to load model on.

        Returns:
            Loaded model object.
        """
        pass

    @abstractmethod
    def predict(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on tokenized input.

        Args:
            model: Loaded model object.
            input_ids: Tokenized input IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].

        Returns:
            Tuple of (probabilities, predictions).
            - probabilities: Shape [batch_size, n_classes]
            - predictions: Shape [batch_size]
        """
        pass

    @abstractmethod
    def validate(self, model_dir: Path) -> bool:
        """Validate that model_dir contains a valid model for this backend.

        Args:
            model_dir: Path to model directory.

        Returns:
            True if valid, False otherwise.
        """
        pass

    @abstractmethod
    def get_config(self, model_dir: Path) -> dict[str, Any]:
        """Get model configuration.

        Args:
            model_dir: Path to model directory.

        Returns:
            Dict with model configuration (e.g., max_seq_len, num_labels).
        """
        pass


class GeneformerBackend(ModelBackend):
    """Geneformer model backend using PEFT/LoRA adapters."""

    @property
    def name(self) -> str:
        return "geneformer"

    @property
    def supported_tasks(self) -> list[str]:
        return ["classification"]

    def load(self, model_dir: Path, device: torch.device | None = None) -> Any:
        from peft import PeftConfig, PeftModel
        from transformers import AutoConfig, AutoModelForSequenceClassification

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        adapter_config = model_dir / "adapter_config.json"
        if adapter_config.exists():
            # PEFT/LoRA adapter
            peft_config = PeftConfig.from_pretrained(str(model_dir))
            base_model_name = peft_config.base_model_name_or_path

            # Resolve base model path
            if not Path(base_model_name).exists():
                possible_bases = [
                    Path(peft_config.base_model_name_or_path),
                    Path.home() / ".cache" / "huggingface" / "hub" / "models--ctheodoris--Geneformer",
                ]
                for base_path in possible_bases:
                    if base_path.exists():
                        base_model_name = str(base_path)
                        break

            config = AutoConfig.from_pretrained(base_model_name, num_labels=2)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                config=config,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )
            model = PeftModel.from_pretrained(base_model, str(model_dir))
            model = model.merge_and_unload()  # Merge for faster inference
        else:
            # Full model checkpoint
            config = AutoConfig.from_pretrained(str(model_dir), num_labels=2)
            model = AutoModelForSequenceClassification.from_pretrained(
                str(model_dir),
                config=config,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )

        model.to(device)
        model.eval()
        return model

    def predict(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
        return probs, preds

    def validate(self, model_dir: Path) -> bool:
        adapter_config = model_dir / "adapter_config.json"
        if adapter_config.exists():
            import json
            try:
                with open(adapter_config) as f:
                    cfg = json.load(f)
                return str(cfg.get("peft_type", "")).upper() == "LORA"
            except Exception:
                return False
        # Full checkpoint validation
        return (
            (model_dir / "pytorch_model.bin").exists()
            or (model_dir / "model.safetensors").exists()
        )

    def get_config(self, model_dir: Path) -> dict[str, Any]:
        adapter_config = model_dir / "adapter_config.json"
        if adapter_config.exists():
            import json
            with open(adapter_config) as f:
                cfg = json.load(f)
            base_model_name = cfg.get("base_model_name_or_path", "")
            return {
                "max_seq_len": 2048,  # Geneformer V1 default
                "num_labels": 2,
                "is_peft": True,
                "base_model": base_model_name,
            }
        return {
            "max_seq_len": 2048,
            "num_labels": 2,
            "is_peft": False,
        }


# Registry for backends
_backends: dict[str, type[ModelBackend]] = {}


def register_backend(backend_class: type[ModelBackend]) -> None:
    """Register a new model backend.

    Args:
        backend_class: ModelBackend subclass to register.
    """
    instance = backend_class()
    _backends[instance.name] = backend_class


def get_backend(name: str) -> ModelBackend:
    """Get a backend instance by name.

    Args:
        name: Backend name (e.g., 'geneformer').

    Returns:
        Backend instance.

    Raises:
        ValueError: If backend not found.
    """
    if name not in _backends:
        raise ValueError(f"Unknown backend: {name}. Available: {list(_backends.keys())}")
    return _backends[name]()


def list_backends() -> list[str]:
    """List all registered backend names."""
    return list(_backends.keys())


# Register built-in backends
register_backend(GeneformerBackend)


__all__ = [
    "ModelBackend",
    "GeneformerBackend",
    "register_backend",
    "get_backend",
    "list_backends",
]
