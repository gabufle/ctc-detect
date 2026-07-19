"""Core detection pipeline for CTC-Detect.

Orchestrates: load data -> preprocess -> tokenize -> inference -> results -> reports.
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
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from ctcdetect.config.paths import (
    CHECKPOINT_DIR,
    FINETUNED_DIR,
    GENE_MAPPING,
    GENEFORMER_DIR,
    TOKEN_DICT,
    GENE_MEDIAN,
)
from ctcdetect.core.model import _load_model, _resolve_model_dir, check_geneformer_available
from ctcdetect.core.preprocess import (
    detect_format,
    run_qc,
    normalize,
    map_genes_to_ensembl,
)
from ctcdetect.evaluation.plots import generate_umap
from ctcdetect.evaluation.reports import generate_report, generate_html_report

console = Console()


def _prepare_adata(input_path: Path, progress: Progress, task) -> sc.AnnData:
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


def _tokenize(adata: sc.AnnData, progress: Progress, task):
    """Tokenize AnnData using Geneformer's TranscriptomeTokenizer.

    Returns tokenized dataset and the processed AnnData (for UMAP).
    """
    from geneformer import TranscriptomeTokenizer

    # Save AnnData to temp h5ad for tokenizer
    tmp_dir = tempfile.mkdtemp()
    tmp_h5ad = os.path.join(tmp_dir, "input.h5ad")
    adata.write_h5ad(tmp_h5ad)

    try:
        # Initialize tokenizer (V1 model)
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


def _run_inference(model, device, dataset, progress: Progress, task):
    """Run batched inference on tokenized dataset.

    Returns (barcodes, ctc_probabilities, predicted_labels, uncertain_flags).
    """
    model.eval()

    # Determine max sequence length
    max_len = min(max(len(x) for x in dataset["input_ids"]), 2048)
    n = len(dataset)
    batch_size = 32

    console.print(f"  Sequences: {n}, max length: {max_len}, batch size: {batch_size}")

    # Pre-pad all sequences
    padded = torch.zeros((n, max_len), dtype=torch.long)
    attention_mask = torch.zeros((n, max_len), dtype=torch.long)

    for i in range(n):
        ids = dataset["input_ids"][i][:max_len]
        padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :len(ids)] = 1

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
    from ctcdetect.utils import print_banner

    print_banner()
    console.print(f"[bold]Input:[/bold]  {input_path}")
    console.print(f"[bold]Output:[/bold] {output_path}")
    console.print()

    # Verify Geneformer
    if not check_geneformer_available():
        console.print(f"[red]Error:[/red] Geneformer directory not found at {GENEFORMER_DIR}")
        console.print("Please clone the Geneformer repository from HuggingFace:")
        console.print("  git clone https://huggingface.co/ctheodoris/Geneformer \\")
        console.print("    Geneformer/")
        raise SystemExit(1)

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
            generate_umap(adata, results_df, output_path)

    # Generate reports
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating reports...", total=None)
        generate_report(results_df, output_path, threshold)
        generate_html_report(results_df, output_path, threshold)

    console.print(f"\n[green]✓[/green] Detection complete. Results in {output_path}")


__all__ = [
    "run_detection",
    "_prepare_adata",
    "_tokenize",
    "_run_inference",
]