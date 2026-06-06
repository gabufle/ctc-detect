"""Core detection logic for CTC-Detect.

This module runs the Geneformer model on preprocessed single-cell
expression data and outputs per-cell CTC probability scores.
"""

import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID

console = Console()

# Geneformer paths
GENEFORMER_DIR = Path(__file__).resolve().parents[2] / "Geneformer" / "geneformer"
TOKEN_DICT = GENEFORMER_DIR / "token_dictionary_gc104M.pkl"
GENE_MEDIAN = GENEFORMER_DIR / "gene_median_dictionary_gc104M.pkl"
GENE_MAPPING = GENEFORMER_DIR / "ensembl_mapping_dict_gc104M.pkl"

# Model paths
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "results" / "checkpoints" / "best_model"
# Fallback: the fine-tuned model from the original training
FINETUNED_DIR = Path(__file__).resolve().parents[2] / "Geneformer" / "Geneformer-V1-10M"


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
    console.print("  python train.py  # saves to results/checkpoints/best_model/")
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
    from transformers import AutoModelForSequenceClassification, AutoConfig
    from peft import PeftModel, PeftConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"  Device: {device}")

    # Check if this is a PEFT/LoRA adapter
    adapter_config = model_dir / "adapter_config.json"

    if adapter_config.exists():
        # PEFT adapter: load base model + adapter
        console.print("  Loading PEFT/LoRA adapter model...")
        peft_config = PeftConfig.from_pretrained(str(model_dir))
        base_model_name = peft_config.base_model_name_or_path

        # If base model name is a local path that doesn't exist, use the default
        if not Path(base_model_name).exists():
            # Try to find the base model in common locations
            possible_bases = [
                Path(peft_config.base_model_name_or_path),
                FINETUNED_DIR.parent / "Geneformer-V1-10M",
                Path.home() / ".cache" / "huggingface" / "hub" / f"models--ctheodoris--Geneformer",
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
    else:
        # Full model checkpoint
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
    2. QC filter
    3. Normalize
    4. Map gene symbols to Ensembl IDs
    """
    from ctcdetect.preprocess import detect_format

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
    progress.update(task, advance=0.2)

    # QC: calculate metrics
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    # QC: filter cells
    n_before = adata.shape[0]
    adata = adata[
        (adata.obs["n_genes_by_counts"] >= 200)
        & (adata.obs["n_genes_by_counts"] <= 6000)
        & (adata.obs["pct_counts_mt"] <= 20),
        :,
    ].copy()
    n_after = adata.shape[0]
    console.print(f"  QC: {n_before} → {n_after} cells ({n_before - n_after} removed)")
    progress.update(task, advance=0.2)

    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    progress.update(task, advance=0.2)

    # Map gene symbols to Ensembl IDs
    with open(GENE_MAPPING, "rb") as f:
        gene_mapping = pickle.load(f)

    ensembl_ids = []
    mapped = 0
    for gene in adata.var_names:
        if gene in gene_mapping:
            ensembl_ids.append(gene_mapping[gene])
            mapped += 1
        else:
            ensembl_ids.append(None)

    adata.var["ensembl_id"] = ensembl_ids
    adata = adata[:, adata.var["ensembl_id"].notna()].copy()
    # Use .values to avoid type issues with scanpy's var_names setter
    new_names = adata.var["ensembl_id"].values.astype(str)
    adata.var_names = pd.Index(new_names)
    # Make var_names unique
    adata.var_names_make_unique()

    console.print(f"  Gene mapping: {mapped}/{len(ensembl_ids)} genes mapped to Ensembl IDs")
    console.print(f"  After mapping: {adata.shape[0]} cells × {adata.shape[1]} genes")

    if adata.shape[1] == 0:
        console.print("[red]Error:[/red] No genes could be mapped to Ensembl IDs.")
        console.print("This usually means your input data uses gene symbols that are not in the")
        console.print("Ensembl mapping dictionary. Ensure your data uses standard HGNC gene symbols")
        console.print("(e.g., TP53, BRCA1, EGFR) rather than numeric IDs or custom names.")
        raise SystemExit(1)

    progress.update(task, advance=0.2)

    # Add n_counts for tokenizer
    if hasattr(adata.X, "toarray"):
        n_counts = np.array(adata.X.sum(axis=1)).flatten()
    else:
        n_counts = np.array(adata.X.sum(axis=1)).flatten()
    adata.obs["n_counts"] = n_counts

    progress.update(task, advance=0.2)
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
    # Recompute from stored values
    uncertain = []
    # We need the raw probabilities to determine uncertainty
    # Re-run with full softmax to get max probs
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
    import seaborn as sns

    console.print("  Computing UMAP on expression data...")

    # Work on a copy for UMAP computation
    adata_umap = adata.copy()

    # Standard scanpy UMAP pipeline
    sc.pp.highly_variable_genes(adata_umap, n_top_genes=2000, flavor="seurat_v3")
    sc.pp.pca(adata_umap, n_comps=30)
    sc.pp.neighbors(adata_umap, n_pcs=30)
    sc.tl.umap(adata_umap)

    # Create barcode -> index mapping
    barcode_to_idx = {bc: i for i, bc in enumerate(adata_umap.obs_names)}

    # Map results to UMAP cells
    umap_probs = np.full(adata_umap.shape[0], np.nan)
    umap_preds = np.full(adata_umap.shape[0], np.nan)
    umap_uncertain = np.full(adata_umap.shape[0], False)

    for _, row in results_df.iterrows():
        bc = row["barcode"]
        if bc in barcode_to_idx:
            idx = barcode_to_idx[bc]
            umap_probs[idx] = row["ctc_probability"]
            umap_preds[idx] = row["predicted_label"]
            umap_uncertain[idx] = row["uncertain"]

    # Create four-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("CTC-Detect Results", fontsize=16, fontweight="bold")

    umap_coords = adata_umap.obsm["X_umap"]

    # Panel 1: CTC Probability
    ax = axes[0, 0]
    valid = ~np.isnan(umap_probs)
    scatter = ax.scatter(
        umap_coords[valid, 0], umap_coords[valid, 1],
        c=umap_probs[valid], cmap="RdYlBu_r", s=3, alpha=0.7,
        vmin=0, vmax=1,
    )
    ax.scatter(
        umap_coords[~valid, 0], umap_coords[~valid, 1],
        c="lightgray", s=1, alpha=0.3,
    )
    plt.colorbar(scatter, ax=ax, label="CTC Probability")
    ax.set_title("Predicted CTC Probability")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Panel 2: Predicted Label
    ax = axes[0, 1]
    valid = ~np.isnan(umap_preds)
    colors = ["#2196F3", "#F44336"]  # blue=non-CTC, red=CTC
    cmap = ListedColormap(colors)
    ax.scatter(
        umap_coords[valid, 0], umap_coords[valid, 1],
        c=umap_preds[valid], cmap=cmap, s=3, alpha=0.7,
        vmin=0, vmax=1,
    )
    ax.scatter(
        umap_coords[~valid, 0], umap_coords[~valid, 1],
        c="lightgray", s=1, alpha=0.3,
    )
    ax.set_title("Predicted Label (0=non-CTC, 1=CTC)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Panel 3: Uncertainty
    ax = axes[1, 0]
    ax.scatter(
        umap_coords[umap_uncertain, 0], umap_coords[umap_uncertain, 1],
        c="#FF9800", s=3, alpha=0.7, label="Uncertain",
    )
    ax.scatter(
        umap_coords[~umap_uncertain, 0], umap_coords[~umap_uncertain, 1],
        c="#9E9E9E", s=1, alpha=0.3, label="Confident",
    )
    ax.set_title("Uncertainty Flag (prob < 0.6)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=5)

    # Panel 4: Score histogram
    ax = axes[1, 1]
    valid_probs = umap_probs[~np.isnan(umap_probs)]
    ax.hist(valid_probs, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(x=0.5, color="red", linestyle="--", label="Threshold (0.5)")
    ax.set_xlabel("CTC Probability")
    ax.set_ylabel("Number of Cells")
    ax.set_title("Score Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"  UMAP saved to {output_path}")


def _generate_summary(results_df: pd.DataFrame, output_path: Path):
    """Generate plain-language clinical summary report."""
    total = len(results_df)
    ctc = (results_df["predicted_label"] == 1).sum()
    non_ctc = (results_df["predicted_label"] == 0).sum()
    uncertain = results_df["uncertain"].sum()
    mean_score = results_df["ctc_probability"].mean()
    median_score = results_df["ctc_probability"].median()

    lines = [
        "=" * 60,
        "          CTC-DETECT SUMMARY REPORT",
        "=" * 60,
        "",
        f"  Total cells analyzed:     {total}",
        f"  CTCs detected:            {ctc} ({ctc/total*100:.1f}%)",
        f"  Non-CTCs:                 {non_ctc} ({non_ctc/total*100:.1f}%)",
        f"  Uncertain predictions:    {uncertain} ({uncertain/total*100:.1f}%)",
        "",
        "  CTC Probability Scores:",
        f"    Mean:   {mean_score:.4f}",
        f"    Median: {median_score:.4f}",
        "",
        "  Interpretation:",
    ]

    if ctc == 0:
        lines.append("    No circulating tumor cells were detected in this sample.")
    elif ctc / total < 0.01:
        lines.append(
            f"    Rare CTCs detected ({ctc} cells, {ctc/total*100:.2f}% of total). "
            "This is consistent with early-stage or low-burden disease."
        )
    elif ctc / total < 0.1:
        lines.append(
            f"    Moderate CTC burden ({ctc} cells, {ctc/total*100:.1f}% of total). "
            "Clinical correlation recommended."
        )
    else:
        lines.append(
            f"    High CTC burden ({ctc} cells, {ctc/total*100:.1f}% of total). "
            "This may indicate significant circulating tumor burden."
        )

    if uncertain / total > 0.3:
        lines.append(
            f"    Note: {uncertain/total*100:.0f}% of cells had uncertain predictions. "
            "Consider additional QC or manual review of borderline cases."
        )

    lines.extend([
        "",
        "=" * 60,
        "  Generated by CTC-Detect — Powered by Geneformer",
        "=" * 60,
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    console.print(f"  Summary saved to {output_path}")


def run_detection(
    input_path: Path,
    output_path: Path,
    cancer_type: Optional[str] = None,
    threshold: float = 0.5,
    skip_umap: bool = False,
) -> None:
    """Run CTC detection on a single sample.

    Takes Cell Ranger output (filtered feature-barcode matrix) or h5ad file
    and produces per-cell CTC probability scores, UMAP visualizations,
    and a clinical summary report.

    Args:
        input_path: Path to Cell Ranger output directory or h5ad file.
        output_path: Path to output directory (created if needed).
        cancer_type: Optional cancer type hint (currently unused).
        threshold: Probability threshold for calling a cell a CTC (default 0.5).
        skip_umap: If True, skip UMAP computation for faster runs.
    """
    _check_geneformer()
    model_dir = _resolve_model_dir()

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:3.0f}%"),
        console=console,
    ) as progress:

        # Step 1: Load and preprocess
        task1 = progress.add_task("[cyan]Loading and preprocessing data...", total=1)
        adata = _prepare_adata(input_path, progress, task1)

        # Step 2: Tokenize
        task2 = progress.add_task("[cyan]Tokenizing with Geneformer...", total=1)
        dataset, adata = _tokenize(adata, progress, task2)

        # Step 3: Load model
        task3 = progress.add_task("[cyan]Loading model...", total=1)
        model, device = _load_model(model_dir)
        progress.update(task3, advance=1)

        # Step 4: Run inference
        task4 = progress.add_task("[cyan]Running inference...", total=1)
        barcodes, probs, preds, uncertain = _run_inference(
            model, device, dataset, progress, task4
        )

        # Step 5: Write results
        task5 = progress.add_task("[cyan]Writing results...", total=1)

        results_df = pd.DataFrame({
            "barcode": barcodes,
            "ctc_probability": probs,
            "predicted_label": preds,
            "uncertain": uncertain,
        })

        # Apply threshold to predicted labels
        results_df["predicted_label"] = (results_df["ctc_probability"] >= threshold).astype(int)

        csv_path = output_path / "ctc_probabilities.csv"
        results_df.to_csv(csv_path, index=False)
        console.print(f"  CSV saved to {csv_path}")

        # Generate UMAP (unless skipped)
        if not skip_umap:
            umap_path = output_path / "umap.png"
            _generate_umap(adata, results_df, umap_path)
        else:
            console.print("  UMAP skipped (--skip-umap)")

        # Generate summary
        summary_path = output_path / "summary.txt"
        _generate_summary(results_df, summary_path)

        progress.update(task5, advance=1)

    console.print(f"\n[green]✓[/green] All results written to {output_path}")
    console.print(f"  - ctc_probabilities.csv  ({len(results_df)} cells)")
    if not skip_umap:
        console.print(f"  - umap.png               (4-panel visualization)")
    console.print(f"  - summary.txt            (clinical summary)")
    console.print(f"  Threshold: {threshold}")
