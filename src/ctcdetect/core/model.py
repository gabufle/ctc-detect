"""Model loading and validation for CTC-Detect.

Handles both PEFT/LoRA adapters and full model checkpoints.
"""

import json
import sys
from pathlib import Path

import torch
from rich.console import Console

from ctcdetect.config.paths import CHECKPOINT_DIR, FINETUNED_DIR
from ctcdetect.exceptions import InferenceError, ModelError

console = Console()


def _validate_adapter_config(model_dir: Path) -> dict:
    """Validate that ``model_dir`` holds a well-formed PEFT/LoRA adapter config.

    The fine-tuned CTC model is a LoRA adapter, so ``adapter_config.json`` must
    exist and describe a LoRA adapter with a base model. If it is missing,
    unreadable, or not a LoRA adapter, fail loudly rather than silently loading
    the wrong weights (e.g. the base Geneformer, which scores near random).

    Args:
        model_dir: Directory expected to contain a PEFT/LoRA adapter.

    Returns:
        The parsed adapter config as a dict.

    Raises:
        SystemExit: If the adapter config is missing or malformed.
    """
    adapter_config = model_dir / "adapter_config.json"
    if not adapter_config.exists():
        console.print(
            f"[red]Error:[/red] Expected a PEFT/LoRA adapter config at {adapter_config}, "
            "but none was found."
        )
        console.print(
            "The CTC model must be a LoRA adapter (adapter_config.json + adapter weights). "
            "Point at a fine-tuned adapter directory, or retrain."
        )
        raise SystemExit(1)

    try:
        with open(adapter_config) as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        console.print(f"[red]Error:[/red] Could not read adapter config {adapter_config}: {e}")
        raise SystemExit(1)

    peft_type = str(cfg.get("peft_type", "")).upper()
    if peft_type != "LORA":
        console.print(
            f"[red]Error:[/red] {adapter_config} is not a LoRA adapter "
            f"(peft_type={cfg.get('peft_type')!r}, expected 'LORA')."
        )
        raise SystemExit(1)

    if not cfg.get("base_model_name_or_path"):
        console.print(
            f"[red]Error:[/red] {adapter_config} is missing 'base_model_name_or_path'; "
            "cannot locate the base Geneformer model for the adapter."
        )
        raise SystemExit(1)

    return cfg


def _resolve_model_dir() -> Path:
    """Resolve which model directory to use.

    Priority:
    1. results/checkpoints/best_model/ (user-trained checkpoint)
    2. Geneformer/Geneformer-V1-10M/ (pre-trained fallback)

    Returns:
        Path to the model directory.

    Raises:
        SystemExit: If no valid model is found.
    """
    # Check if checkpoint has actual model files
    if CHECKPOINT_DIR.exists():
        has_weights = (
            (CHECKPOINT_DIR / "pytorch_model.bin").exists()
            or (CHECKPOINT_DIR / "model.safetensors").exists()
            or (CHECKPOINT_DIR / "adapter_model.bin").exists()
            or (CHECKPOINT_DIR / "adapter_model.safetensors").exists()
        )
        if has_weights:
            console.print(f"[green]✓[/green] Using fine-tuned checkpoint: {CHECKPOINT_DIR}")
            return CHECKPOINT_DIR
        else:
            console.print(
                f"[yellow]Warning:[/yellow] Checkpoint directory exists but has no model weights: {CHECKPOINT_DIR}"
            )

    # Fallback to pre-trained fine-tuned model
    if FINETUNED_DIR.exists():
        has_weights = (
            (FINETUNED_DIR / "pytorch_model.bin").exists()
            or (FINETUNED_DIR / "model.safetensors").exists()
        )
        if has_weights:
            console.print(
                f"[yellow]Warning:[/yellow] No trained checkpoint found. Using pre-trained model: {FINETUNED_DIR}"
            )
            console.print(
                "To use your own trained model, save it to: results/checkpoints/best_model/"
            )
            return FINETUNED_DIR

    # No model found at all
    console.print("[red]Error:[/red] No model found.")
    console.print("")
    console.print("You need a fine-tuned Geneformer model to run CTC detection.")
    console.print("")
    console.print("Option 1 — Train a model (recommended):")
    console.print("  python -m ctcdetect.training.train  # saves to results/checkpoints/best_model/")
    console.print("")
    console.print("Option 2 — Download the pre-trained CTC model:")
    console.print("  huggingface-cli download ctheodoris/Geneformer-V1-10M \\")
    console.print("    --local-dir Geneformer/Geneformer-V1-10M")
    console.print("")
    console.print("Option 3 — Use the base Geneformer (no fine-tuning):")
    console.print("  huggingface-cli download ctheodoris/Geneformer \\")
    console.print("    --local-dir Geneformer/Geneformer-V1-10M")
    raise SystemExit(1)


def _load_model(model_dir: Path):
    """Load the fine-tuned Geneformer model.

    Handles both PEFT/LoRA adapters and full model checkpoints.
    Returns (model, device).
    """
    from transformers import AutoConfig, AutoModelForSequenceClassification
    from peft import PeftConfig, PeftModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"  Device: {device}")

    # Check if this is a PEFT/LoRA adapter
    adapter_config = model_dir / "adapter_config.json"

    if adapter_config.exists():
        # PEFT adapter: validate config, then load base model + adapter
        _validate_adapter_config(model_dir)
        console.print("  Loading PEFT/LoRA adapter model...")
        peft_config = PeftConfig.from_pretrained(str(model_dir))
        base_model_name = peft_config.base_model_name_or_path

        # If base model name is a local path that doesn't exist, try to find it
        if not Path(base_model_name).exists():
            # Try common locations
            possible_bases = [
                Path(peft_config.base_model_name_or_path),
                FINETUNED_DIR.parent / "Geneformer-V1-10M",
                Path.home() / ".cache" / "huggingface" / "hub" / "models--ctheodoris--Geneformer",
            ]
            for base_path in possible_bases:
                if base_path.exists():
                    base_model_name = str(base_path)
                    break
            else:
                # Use the model_dir itself as base (might be a full model with adapter)
                base_model_name = str(model_dir)
                console.print(f"  [yellow]Warning:[/yellow] Base model not found, using {base_model_name}")

        config = AutoConfig.from_pretrained(base_model_name, num_labels=2)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            config=config,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        model = PeftModel.from_pretrained(base_model, str(model_dir))
        model = model.merge_and_unload()  # Merge for faster inference
    else:
        # Full model checkpoint (not a PEFT adapter)
        console.print("  Loading full model checkpoint...")
        config = AutoConfig.from_pretrained(str(model_dir), num_labels=2)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            config=config,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )

    model.to(device)
    model.eval()

    # Log trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"  Model params: {total_params:,} total, {trainable_params:,} trainable")

    return model, device


def load_model(model_dir: Path | None = None) -> tuple:
    """Public API: Load the CTC detection model.

    Args:
        model_dir: Optional model directory. If None, auto-resolves.

    Returns:
        Tuple of (model, device).
    """
    if model_dir is None:
        model_dir = _resolve_model_dir()
    return _load_model(model_dir)


def check_geneformer_available() -> bool:
    """Check if Geneformer directory exists with required files.

    Returns:
        True if Geneformer is available, False otherwise.
    """
    from ctcdetect.config.paths import GENEFORMER_DIR, TOKEN_DICT, GENE_MEDIAN, GENE_MAPPING

    if not GENEFORMER_DIR.exists():
        return False

    for f in [TOKEN_DICT, GENE_MEDIAN, GENE_MAPPING]:
        if not f.exists():
            return False

    return True


__all__ = [
    "load_model",
    "check_geneformer_available",
    "_resolve_model_dir",
    "_load_model",
    "_validate_adapter_config",
]