"""Core detection logic for CTC-Detect.

This module runs the Geneformer model on preprocessed single-cell
expression data and outputs per-cell CTC probability scores.
"""

import os
import pickle
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID

from ctcdetect.config import (
    GENEFORMER_DIR,
    TOKEN_DICT,
    GENE_MEDIAN,
    GENE_MAPPING,
    CHECKPOINT_DIR,
    FINETUNED_DIR,
)
from ctcdetect.exceptions import (
    ConfigurationError,
    InputError,
    ModelError,
    TokenizationError,
    InferenceError,
    GeneMappingError,
    OutputError,
)
from ctcdetect.preprocess import (
    detect_format,
    load_data,
    run_qc,
    normalize,
    map_genes_to_ensembl,
)

console = Console()


def _check_geneformer():
    """Verify Geneformer directory exists with required files."""
    if not GENEFORMER_DIR.exists():
        console.print(f"[red]Error:[/red] Geneformer directory not found at {GENEFORMER_DIR}")
        console.print("Please clone the Geneformer repository from HuggingFace:")
        console.print("  git clone https://huggingface.co/ctheodoris/Geneformer \\")
        console.print("    Geneformer/")
        console.print("")
        console.print("Your project directory should look like:")
        console.print("  ctc-detect/")
        console.print("  └── Geneformer/")
        console.print("      └── geneformer/")
        console.print("          ├── token_dictionary_gc104M.pkl")
        console.print("          ├── gene_median_dictionary_gc104M.pkl")
        console.print("          ├── ensembl_mapping_dict_gc104M.pkl")
        console.print("          └── ...")
        raise SystemExit(1)

    missing = []
    for f in [TOKEN_DICT, GENE_MEDIAN, GENE_MAPPING]:
        if not f.exists():
            missing.append(f.name)
    if missing:
        console.print(f"[red]Error:[/red] Missing Geneformer files: {', '.join(missing)}")
        console.print(f"Expected at: {GENEFORMER_DIR}")
        raise SystemExit(1)


def _resolve_model_dir() -> Path:
    """Resolve which model directory to use.

    Priority:
    1. results/checkpoints/best_model/ (user-trained checkpoint)
    2. Geneformer/Geneformer-V1-10M/ (pre-trained fallback)
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
    console.print("  python train.py # saves to results/checkpoints/best_model/")
    console.print("")
    console.print("Option 2 — Download the pre-trained CTC model:")
    console.print("  huggingface-cli download ctheodoris/Geneformer-V1-10M \\")
    console.print("    --local-dir Geneformer/Geneformer-V1-10M")
    console.print("")
    console.print("Option 3 — Use the base Geneformer (no fine-tuning):")
    console.print("  huggingface-cli download ctheodoris/Geneformer \\")
    console.print("    --local-dir Geneformer/Geneformer-V1-10M")
    raise SystemExit(1)


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
    import json

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


def _load_model(model_dir: Path):
    """Load the fine-tuned Geneformer model.

    Handles both PEFT/LoRA adapters and full model checkpoints.
    Returns (model, device).
    """
    from transformers import AutoModelForSequenceClassification, AutoConfig
    from peft import PeftModel, PeftConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"  Device: {device}")

    # Check if this is a PEFT/LoRA adapter
    adapter_config = model_dir / "adapter_config.json"

    if adapter_config.exists():
        # PEFT adapter: validate the config, then load base model + adapter
        _validate_adapter_config(model_dir)
        console.print("  Loading PEFT/LoRA adapter model...")
        peft_config = PeftConfig.from_pretrained(str(model_dir))
        base_model_name = peft_config.base_model_name_or_path

        # If base model name is a local path that doesn't exist, use the default
        if not Path(base_model_name).exists():
            # Try to find the base model in common locations
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
                # Use the model_dir itself as base (it might be a full model with adapter)
                base_model_name = str(model_dir)
                console.print(f"  [yellow]Warning:[/yellow] Could not find base model, using {base_model_name}")

        config = AutoConfig.from_pretrained(base_model_name)
        config.num_labels = 2
        config.problem_type = "single_label_classification"

        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, config=config
        )
        model = PeftModel.from_pretrained(base_model, str(model_dir))
        if not isinstance(model, PeftModel):
            console.print(
                "[red]Error:[/red] Loaded model is not a PeftModel despite an adapter "
                "config being present. The adapter may be incompatible with the installed "
                "peft version."
            )
            raise SystemExit(1)
    else:
        # No adapter_config.json: this is a full checkpoint, NOT a LoRA adapter.
        console.print(
            "[yellow]Warning:[/yellow] No adapter_config.json found — loading a full "
            "checkpoint, not a PEFT/LoRA adapter. If you expected the fine-tuned CTC "
            "adapter, this may be the base Geneformer model, which scores near random."
        )
        console.print("  Loading full model checkpoint...")
        config = AutoConfig.from_pretrained(str(model_dir))
        config.num_labels = 2
        config.problem_type = "single_label_classification"
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), config=config
        )

    model.eval()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"  Total params: {total_params:,}")
    console.print(f"  Trainable params: {trainable_params:,}")

    return model, device


def _prepare_adata(input_path: Path, progress: Progress, task: TaskID) -> sc.AnnData:
    """Load and preprocess AnnData from various input formats.

    Steps:
    1. Load data (Cell Ranger MEX or h5ad)
    2. QC filter (config-driven)
    3. Normalize (config-driven)
    4. Map gene symbols to Ensembl IDs
    """
    fmt = detect_format(input_path)

    # Load data
    if fmt == "cellranger":
        if (input_path / "filtered_feature_bc_matrix").exists():
            mtx_dir = input_path / "filtered_feature_bc_matrix"
        else:
            mtx_dir = input_path
        adata = sc.read_10x_mtx(
            str(mtx_dir), var_names="gene_symbols", cache=True
        )
    elif fmt == "h5ad":
        adata = sc.read_h5ad(str(input_path))
    else:
        raise ValueError(f"Unsupported format for detection: {fmt}")

    console.print(f"  Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")
    progress.update(task, advance=0.15)

    # QC filter
    adata = run_qc(adata)
    progress.update(task, advance=0.2)

    # Normalize
    adata = normalize(adata)
    progress.update(task, advance=0.15)

    # Map gene symbols to Ensembl IDs
    with open(GENE_MAPPING, "rb") as f:
        gene_mapping = pickle.load(f)

    adata = map_genes_to_ensembl(adata, gene_mapping)
    progress.update(task, advance=0.15)

    # Add n_counts for tokenizer
    if hasattr(adata.X, "toarray"):
        n_counts = np.array(adata.X.sum(axis=1)).flatten()
    else:
        n_counts = np.array(adata.X.sum(axis=1)).flatten()
    adata.obs["n_counts"] = n_counts

    progress.update(task, advance=0.15)
    return adata


def _tokenize(adata: sc.AnnData, progress: Progress, task: TaskID):
    """Tokenize AnnData using Geneformer's TranscriptomeTokenizer.

    Returns tokenized dataset and the processed AnnData (for UMAP).
    """
    from geneformer import TranscriptomeTokenizer

    # Save AnnData to temp h5ad for tokenizer
    tmp_dir = tempfile.mkdtemp()
    tmp_h5ad = os.path.join(tmp_dir, "input.h5ad")
    adata.write_h5ad(tmp_h5ad)

    try:
        # Initialize tokenizer (V1 model: special_token=False, model_input_size=2048)
        tk = TranscriptomeTokenizer(
            token_dictionary_file=str(TOKEN_DICT),
            gene_median_file=str(GENE_MEDIAN),
            gene_mapping_file=str(GENE_MAPPING),
            nproc=4,
            model_input_size=2048,
            special_token=False,
            collapse_gene_ids=True,
            model_version="V1",
        )

        # Tokenize
        tokenized_cells, cell_metadata, tokenized_counts = tk.tokenize_anndata(
            tmp_h5ad, target_sum=10000, file_format="h5ad"
        )

        if len(tokenized_cells) == 0:
            console.print("[red]Error:[/red] No cells were tokenized. Check your input data.")
            raise SystemExit(1)

        console.print(f"  Tokenized {len(tokenized_cells)} cells")

        # Create HuggingFace dataset
        dataset = tk.create_dataset(
            tokenized_cells, cell_metadata, tokenized_counts,
            use_generator=False, keep_uncropped_input_ids=False
        )

        console.print(f"  Dataset: {len(dataset)} examples, keys: {list(dataset[0].keys())}")
        progress.update(task, advance=1.0)

        return dataset, adata

    finally:
        # Cleanup temp file
        if os.path.exists(tmp_h5ad):
            os.unlink(tmp_h5ad)
        os.rmdir(tmp_dir)


def _run_inference(model, device, dataset, progress: Progress, task: TaskID):
    """Run batched inference on tokenized dataset.

    Returns (barcodes, ctc_probabilities, predicted_labels, uncertain_flags).
    """
    model.eval()

    # Determine max sequence length (capped at 2048)
    max_len = min(max(len(x) for x in dataset["input_ids"]), 2048)
    n = len(dataset)
    batch_size = 32

    console.print(f"  Sequences: {n}, max length: {max_len}, batch size: {batch_size}")

    # Pre-pad all sequences
    padded = torch.zeros((n, max_len), dtype=torch.long)
    attention_mask = torch.zeros((n, max_len), dtype=torch.long)

    for i in range(n):
        ids = dataset["input_ids"][i][:max_len]
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1

    # Run inference in batches
    all_probs = []
    all_preds = []
    n_batches = (n + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n)

            batch_input = padded[start:end].to(device)
            batch_mask = attention_mask[start:end].to(device)

            outputs = model(input_ids=batch_input, attention_mask=batch_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            # CTC probability = probability of class 1
            ctc_probs = probs[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_probs.extend(ctc_probs.tolist())
            all_preds.extend(preds.tolist())

            progress.update(task, advance=(end - start) / n)

    # Uncertainty: max probability < 0.6
    all_max_probs = []
    with torch.no_grad():
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n)
            batch_input = padded[start:end].to(device)
            batch_mask = attention_mask[start:end].to(device)
            outputs = model(input_ids=batch_input, attention_mask=batch_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            max_probs = torch.max(probs, dim=-1)[0].cpu().numpy()
            all_max_probs.extend(max_probs.tolist())

    uncertain = [mp < 0.6 for mp in all_max_probs]

    # Get barcodes from dataset if available, otherwise use indices
    if "barcode" in dataset.features:
        barcodes = dataset["barcode"]
    else:
        barcodes = [f"cell_{i}" for i in range(n)]

    return list(barcodes), all_probs, all_preds, uncertain


def _generate_umap(adata: sc.AnnData, results_df: pd.DataFrame, output_path: Path):
    """Generate four-panel UMAP visualization.

    Panels:
    1. CTC probability
    2. Predicted label
    3. Uncertainty flag
    4. Expression overlay (placeholder)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    console.print("  Computing UMAP on expression data...")

    # Work on a copy for UMAP computation
    adata_umap = adata.copy()

    # Standard scanpy UMAP pipeline
    sc.pp.highly_variable_genes(adata_umap, n_top_genes=2000, flavor="seurat_v3")
    sc.pp.pca(adata_umap)
    sc.pp.neighbors(adata_umap, n_neighbors=15, metric="cosine")
    sc.tl.umap(adata_umap, min_dist=0.5, spread=1.0, random_state=42)

    # Add results to adata
    barcode_to_idx = {b: i for i, b in enumerate(adata.obs_names)}
    matched_barcodes = [b for b in results_df["barcode"] if b in barcode_to_idx]
    matched_idx = [barcode_to_idx[b] for b in matched_barcodes]

    ctc_probs = np.zeros(adata_umap.shape[0])
    predicted = np.zeros(adata_umap.shape[0], dtype=int)
    uncertain = np.zeros(adata_umap.shape[0], dtype=bool)

    for i, idx in enumerate(matched_idx):
        ctc_probs[idx] = results_df.iloc[i]["ctc_probability"]
        predicted[idx] = results_df.iloc[i]["predicted_label"]
        uncertain[idx] = results_df.iloc[i]["uncertain"]

    adata_umap.obs["ctc_probability"] = ctc_probs
    adata_umap.obs["predicted_label"] = predicted.astype(str)
    adata_umap.obs["uncertain"] = uncertain

    # Four-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: CTC probability
    sc.pl.umap(
        adata_umap, color="ctc_probability", cmap="viridis",
        ax=axes[0, 0], show=False, size=20, title="CTC Probability",
    )

    # Panel 2: Predicted label
    sc.pl.umap(
        adata_umap, color="predicted_label",
        palette={"0": "lightblue", "1": "red"},
        ax=axes[0, 1], show=False, size=20, title="Predicted Label",
    )

    # Panel 3: Uncertainty
    sc.pl.umap(
        adata_umap, color="uncertain",
        palette={True: "orange", False: "green"},
        ax=axes[1, 0], show=False, size=20, title="Uncertain (max prob < 0.6)",
    )

    # Panel 4: Expression of key markers (placeholder)
    # Use a common epithelial marker if available
    marker_candidates = ["EPCAM", "KRT19", "KRT8", "KRT18"]
    found_marker = None
    for m in marker_candidates:
        if m in adata_umap.var_names:
            found_marker = m
            break

    if found_marker:
        sc.pl.umap(
            adata_umap, color=found_marker, cmap="plasma",
            ax=axes[1, 1], show=False, size=20,
            title=f"Expression: {found_marker}",
        )
    else:
        axes[1, 1].text(0.5, 0.5, "No common epithelial\nmarker found",
                        ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Expression Overlay (placeholder)")

    plt.tight_layout()
    fig.savefig(output_path / "umap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    console.print(f"  UMAP saved to {output_path / 'umap.png'}")


def _generate_report(
    results_df: pd.DataFrame,
    adata: sc.AnnData,
    output_path: Path,
    threshold: float,
):
    """Generate text and HTML summary reports."""
    n_total = len(results_df)
    n_ctc = int((results_df["ctc_probability"] >= threshold).sum())
    n_non_ctc = n_total - n_ctc
    n_uncertain = int(results_df["uncertain"].sum())

    mean_ctc_prob = results_df["ctc_probability"].mean()
    median_ctc_prob = results_df["ctc_probability"].median()

    # Text report
    with open(output_path / "summary.txt", "w") as f:
        f.write("CTC-Detect Summary Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total cells analyzed: {n_total}\n")
        f.write(f"CTC calls (prob >= {threshold}): {n_ctc} ({n_ctc/n_total*100:.1f}%)\n")
        f.write(f"Non-CTC calls: {n_non_ctc} ({n_non_ctc/n_total*100:.1f}%)\n")
        f.write(f"Uncertain calls (max prob < 0.6): {n_uncertain}\n\n")
        f.write(f"Mean CTC probability: {mean_ctc_prob:.4f}\n")
        f.write(f"Median CTC probability: {median_ctc_prob:.4f}\n")
        f.write(f"Min CTC probability: {results_df['ctc_probability'].min():.4f}\n")
        f.write(f"Max CTC probability: {results_df['ctc_probability'].max():.4f}\n")

    # HTML report
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>CTC-Detect Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        .metric {{ display: inline-block; background: #f8f9fa; border-radius: 8px; padding: 16px; margin: 8px; min-width: 150px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #6c757d; font-size: 0.9em; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background: #f8f9fa; }}
        .ctc {{ color: #dc3545; }}
        .non-ctc {{ color: #28a745; }}
    </style>
</head>
<body>
    <h1>CTC-Detect Report</h1>

    <div class="metric">
        <div class="metric-value">{n_total}</div>
        <div class="metric-label">Total Cells</div>
    </div>
    <div class="metric ctc">
        <div class="metric-value">{n_ctc}</div>
        <div class="metric-label">CTC Calls (>= {threshold})</div>
    </div>
    <div class="metric non-ctc">
        <div class="metric-value">{n_non_ctc}</div>
        <div class="metric-label">Non-CTC Calls</div>
    </div>
    <div class="metric">
        <div class="metric-value">{n_uncertain}</div>
        <div class="metric-label">Uncertain Calls</div>
    </div>

    <h2>Probability Statistics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Mean CTC Probability</td><td>{mean_ctc_prob:.4f}</td></tr>
        <tr><td>Median CTC Probability</td><td>{median_ctc_prob:.4f}</td></tr>
        <tr><td>Min CTC Probability</td><td>{results_df['ctc_probability'].min():.4f}</td></tr>
        <tr><td>Max CTC Probability</td><td>{results_df['ctc_probability'].max():.4f}</td></tr>
    </table>

    <h2>Top CTC Candidates</h2>
    <table>
        <tr><th>Rank</th><th>Barcode</th><th>CTC Probability</th><th>Predicted</th><th>Uncertain</th></tr>
"""

    top_ctc = results_df.nlargest(20, "ctc_probability")
    for rank, (_, row) in enumerate(top_ctc.iterrows(), 1):
        predicted_label = "CTC" if row["predicted_label"] == 1 else "Non-CTC"
        uncertain_str = "Yes" if row["uncertain"] else "No"
        html += f"<tr><td>{rank}</td><td>{row['barcode']}</td><td>{row['ctc_probability']:.4f}</td><td>{predicted_label}</td><td>{uncertain_str}</td></tr>"

    html += """
    </table>
</body>
</html>"""

    with open(output_path / "report.html", "w") as f:
        f.write(html)

    console.print(f"  Text report: {output_path / 'summary.txt'}")
    console.print(f"  HTML report: {output_path / 'report.html'}")


def run_detection(
    input_path: Path,
    output_path: Path,
    cancer_type: Optional[str] = None,
    threshold: float = 0.5,
    skip_umap: bool = False,
):
    """Run CTC detection on a single sample.

    Args:
        input_path: Path to Cell Ranger output directory or .h5ad file.
        output_path: Path to output directory.
        cancer_type: Cancer type for model selection (not yet implemented).
        threshold: Probability threshold for CTC calls.
        skip_umap: Skip UMAP visualization for faster runs.
    """
    print_banner()
    console.print(f"[bold]Input:[/bold]  {input_path}")
    console.print(f"[bold]Output:[/bold] {output_path}")
    console.print()

    # Verify Geneformer
    _check_geneformer()

    # Resolve model
    model_dir = _resolve_model_dir()

    # Load model
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        model, device = _load_model(model_dir)

    # Run pipeline with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        # Prepare data
        task = progress.add_task("Preprocessing data...", total=1.0)
        adata = _prepare_adata(input_path, progress, task)

        # Tokenize
        task = progress.add_task("Tokenizing...", total=1.0)
        dataset, adata_processed = _tokenize(adata, progress, task)

        # Inference
        task = progress.add_task("Running inference...", total=1.0)
        barcodes, probs, preds, uncertain = _run_inference(model, device, dataset, progress, task)

    # Build results DataFrame
    results_df = pd.DataFrame({
        "barcode": barcodes,
        "ctc_probability": probs,
        "predicted_label": preds,
        "uncertain": uncertain,
    })

    # Apply threshold
    results_df["ctc_call"] = (results_df["ctc_probability"] >= threshold).astype(int)

    # Save results
    output_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path / "ctc_probabilities.csv", index=False)
    console.print(f"[green]✓[/green] Results saved to {output_path / 'ctc_probabilities.csv'}")

    # Generate UMAP
    if not skip_umap:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating UMAP...", total=None)
            _generate_umap(adata, results_df, output_path)

    # Generate reports
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating reports...", total=None)
        _generate_report(results_df, adata, output_path, threshold)

    console.print(f"\n[green]✓[/green] Detection complete. Results in {output_path}")


def print_banner():
    """Print the CTC-Detect banner."""
    console.print()
    console.print("[bold cyan]██████╗███████╗██████╗ ██╗   ██╗███████╗██████╗ [/bold cyan]")
    console.print("[bold cyan]██╔════╝██╔════╝██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗[/bold cyan]")
    console.print("[bold cyan]██║     █████╗  ██████╔╝ ╚████╔╝ █████╗  ██████╔╝[/bold cyan]")
    console.print("[bold cyan]██║     ██╔══╝  ██╔══██╗  ╚██╔╝  ██╔══╝  ██╔══██╗[/bold cyan]")
    console.print("[bold cyan]╚██████╗███████╗██║  ██║   █║   ██║   ███████╗██║  ██║[/bold cyan]")
    console.print("[bold cyan] ╚═════╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝[/bold cyan]")
    console.print("       [dim]Geneformer-based CTC Detection[/dim]")
    console.print()


# Entry point for direct script execution
if __name__ == "__main__":
    import typer
    app = typer.Typer()
    app.command()(run_detection)
    app()