"""Training loop and LoRA configuration for CTC-Detect.

Fine-tunes Geneformer with PEFT/LoRA for CTC classification.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset


def create_lora_config(
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[list] = None,
) -> LoraConfig:
    """Create LoRA configuration for Geneformer fine-tuning.

    Args:
        r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: Dropout rate for LoRA layers.
        target_modules: List of module names to apply LoRA to.
                       If None, uses default for Geneformer (query, value).

    Returns:
        LoraConfig object.
    """
    if target_modules is None:
        target_modules = ["query", "value"]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.SEQ_CLS,
        bias="none",
    )


def load_base_model(
    model_path: Path,
    num_labels: int = 2,
    device: Optional[torch.device] = None,
) -> Tuple:
    """Load base Geneformer model for fine-tuning.

    Args:
        model_path: Path to base Geneformer model directory.
        num_labels: Number of output labels (2 for binary CTC classification).
        device: Device to load model on.

    Returns:
        Tuple of (model, tokenizer, config).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(str(model_path), num_labels=num_labels)
    config.problem_type = "single_label_classification"

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        config=config,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )

    model.to(device)
    return model, config


def train_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    base_model_path: Path,
    output_dir: Path,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    logging_steps: int = 50,
    eval_steps: int = 100,
    save_steps: int = 100,
    max_seq_len: int = 2048,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    seed: int = 42,
    fp16: bool = True,
    gradient_accumulation_steps: int = 1,
) -> Path:
    """Train Geneformer with LoRA for CTC detection.

    Args:
        train_dataset: Tokenized training dataset.
        val_dataset: Tokenized validation dataset.
        base_model_path: Path to base Geneformer model.
        output_dir: Directory to save checkpoints and logs.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        warmup_steps: Number of warmup steps.
        weight_decay: Weight decay for AdamW.
        logging_steps: Log every N steps.
        eval_steps: Evaluate every N steps.
        save_steps: Save checkpoint every N steps.
        max_seq_len: Maximum sequence length.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_dropout: LoRA dropout.
        seed: Random seed.
        fp16: Use mixed precision training.
        gradient_accumulation_steps: Gradient accumulation steps.

    Returns:
        Path to the best model checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load base model
    model, config = load_base_model(base_model_path, num_labels=2, device=device)

    # Apply LoRA
    lora_config = create_lora_config(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_auroc",
        greater_is_better=True,
        fp16=fp16 and device.type == "cuda",
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        report_to="none",  # Disable wandb/tensorboard by default
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # Custom metrics function
    def compute_metrics(eval_pred):
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            f1_score,
            accuracy_score,
        )
        import numpy as np

        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        preds = np.argmax(logits, axis=-1)

        auroc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)
        f1 = f1_score(labels, preds, zero_division=0)
        acc = accuracy_score(labels, preds)

        return {
            "auroc": auroc,
            "auprc": auprc,
            "f1": f1,
            "accuracy": acc,
        }

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save best model
    best_model_path = output_dir / "best_model"
    trainer.save_model(str(best_model_path))
    print(f"Best model saved to {best_model_path}")

    return best_model_path


__all__ = [
    "create_lora_config",
    "load_base_model",
    "train_model",
]