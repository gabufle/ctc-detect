"""CTC-Detect CLI — Geneformer-based circulating tumor cell detection.

Detect circulating tumor cells (CTCs) from single-cell RNA-seq data
produced by 10x Genomics Cell Ranger. Outputs per-cell CTC probability
scores, UMAP visualizations, and clinical summary reports.
"""

import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ctcdetect.utils import validate_input_path, validate_output_path, print_banner
from ctcdetect.config import (
    MODEL_REGISTRY,
    DEFAULT_MODEL,
    get_version,
    get_model_cache_path,
    get_system_info,
)

app = typer.Typer(
    name="ctc-detect",
    help=(
        "Detect circulating tumor cells (CTCs) from single-cell RNA-seq data.\n\n"
        "CTC-Detect uses a fine-tuned Geneformer model to score each cell\n"
        "in your Cell Ranger output for likelihood of being a CTC."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Sub-app for model management commands
model_app = typer.Typer(
    help="Manage Geneformer models — download, list, and inspect available models."
)
app.add_typer(model_app, name="model")

console = Console()


# ---------------------------------------------------------------------------
# RUN command
# ---------------------------------------------------------------------------
@app.command()
def run(
    input: str = typer.Option(
        ...,
        "--input", "-i",
        help=(
            "Path to Cell Ranger output directory or .h5ad file.\n"
            "This should be the filtered_feature_bc_matrix folder\n"
            "from your 10x Genomics run."
        ),
        rich_help_panel="Input/Output",
    ),
    output: str = typer.Option(
        ...,
        "--output", "-o",
        help=(
            "Path to output directory.\n"
            "CTC-Detect will write ctc_probabilities.csv, umap.png,\n"
            "and summary.txt to this directory."
        ),
        rich_help_panel="Input/Output",
    ),
    cancer_type: str = typer.Option(
        None,
        "--cancer-type", "-c",
        help=(
            "Cancer type for model selection (e.g. 'breast', 'lung', 'prostate').\n"
            "If not specified, a pan-cancer model will be used."
        ),
        rich_help_panel="Model Options",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold", "-t",
        help=(
            "Probability threshold for calling a cell a CTC.\n"
            "Cells with CTC probability >= this value are called CTCs.\n"
            "Lower values increase sensitivity; higher values increase specificity."
        ),
        rich_help_panel="Model Options",
    ),
    skip_umap: bool = typer.Option(
        False,
        "--skip-umap",
        help="Skip UMAP visualization for faster runs.",
        rich_help_panel="Model Options",
    ),
):
    """Run CTC detection on a single sample.

    Takes your Cell Ranger output and scores each cell for
    circulating tumor cell probability using Geneformer.
    """
    print_banner()
    input_path = validate_input_path(input, "Input path")
    output_path = validate_output_path(output)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Loading model and tokenizing...", total=None)

    from ctcdetect.detect import run_detection
    run_detection(input_path, output_path, cancer_type, threshold=threshold, skip_umap=skip_umap)

    console.print(f"\n[green]✓[/green] Results written to {output_path}")


# ---------------------------------------------------------------------------
# VALIDATE command
# ---------------------------------------------------------------------------
@app.command()
def validate(
    input: str = typer.Option(
        ...,
        "--input", "-i",
        help=(
            "Path to Cell Ranger output directory or .h5ad file.\n"
            "CTC-Detect will check that the data is valid and readable."
        ),
        rich_help_panel="Input/Output",
    ),
):
    """Validate input data without running detection.

    Use this to check that your Cell Ranger output is complete
    and in the expected format before running the full pipeline.
    """
    print_banner()
    input_path = validate_input_path(input, "Input path")

    from ctcdetect.preprocess import validate_input
    validate_input(input_path)

    console.print(f"\n[green]✓[/green] Input data looks good: {input_path}")


# ---------------------------------------------------------------------------
# BATCH command
# ---------------------------------------------------------------------------
@app.command()
def batch(
    input_dir: str = typer.Option(
        ...,
        "--input-dir",
        help=(
            "Directory containing multiple Cell Ranger output folders.\n"
            "Each subfolder should be one sample."
        ),
        rich_help_panel="Input/Output",
    ),
    output_dir: str = typer.Option(
        ...,
        "--output-dir",
        help=(
            "Directory to write results for all samples.\n"
            "One results CSV will be created per sample."
        ),
        rich_help_panel="Input/Output",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold", "-t",
        help="Probability threshold for CTC calls (default 0.5).",
        rich_help_panel="Model Options",
    ),
    skip_umap: bool = typer.Option(
        False,
        "--skip-umap",
        help="Skip UMAP visualization for faster runs.",
        rich_help_panel="Model Options",
    ),
):
    """Run CTC detection on multiple samples at once.

    Point this at a folder containing multiple Cell Ranger outputs
    and CTC-Detect will process each one sequentially.
    """
    print_banner()
    input_path = validate_input_path(input_dir, "Input directory")
    output_path = validate_output_path(output_dir)

    # Discover sample subdirectories
    samples = sorted([
        d for d in input_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not samples:
        console.print(f"[red]Error:[/red] No subdirectories found in {input_path}")
        console.print("Each sample should be in its own subdirectory.")
        raise SystemExit(1)

    console.print(f"Found {len(samples)} samples to process.\n")

    from ctcdetect.detect import run_detection

    success_count = 0
    fail_count = 0

    for i, sample_dir in enumerate(samples, 1):
        sample_name = sample_dir.name
        sample_output = output_path / sample_name
        console.print(f"[bold][{i}/{len(samples)}] Processing: {sample_name}[/bold]")

        try:
            run_detection(
                sample_dir, sample_output,
                threshold=threshold, skip_umap=skip_umap,
            )
            success_count += 1
            console.print(f"[green]✓[/green] {sample_name} complete\n")
        except SystemExit as e:
            fail_count += 1
            console.print(f"[red]✗[/red] {sample_name} failed (exit code {e.code})\n")
        except Exception as e:
            fail_count += 1
            console.print(f"[red]✗[/red] {sample_name} failed: {e}\n")

    # Summary
    console.print("=" * 50)
    console.print(f"Batch complete: {success_count} succeeded, {fail_count} failed")
    if fail_count > 0:
        console.print("[yellow]Check individual sample outputs for error details.[/yellow]")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# INFO command
# ---------------------------------------------------------------------------
@app.command()
def info():
    """Show CTC-Detect version and system information.

    Displays the installed version, available models, and
    system configuration. Useful for troubleshooting.
    """
    print_banner()

    info = get_system_info()

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="bold cyan", min_width=22)
    table.add_column("Value")

    table.add_row("Version", info["version"])
    table.add_row("Python", info["python"])
    table.add_row("Platform", info["platform"])
    table.add_row("PyTorch", info["pytorch"])
    table.add_row("CUDA available", "Yes" if info["cuda_available"] else "No")
    if info["cuda_available"]:
        table.add_row("CUDA version", info["cuda_version"])
        table.add_row("GPU", info["gpu"])
    table.add_row("Geneformer installed", "Yes" if info["geneformer_installed"] else "No")
    if info["geneformer_installed"]:
        table.add_row("Geneformer path", info["geneformer_path"])
    table.add_row("Checkpoint available", "Yes" if info["checkpoint_available"] else "No")
    if info["checkpoint_available"]:
        table.add_row("Checkpoint path", info["checkpoint_path"])

    cached = info.get("cached_models", [])
    table.add_row("Cached models", ", ".join(cached) if cached else "None")

    console.print(table)
    console.print()
    console.print("For documentation, visit: https://github.com/gabufle/ctc-detect")


# ---------------------------------------------------------------------------
# MODEL sub-commands
# ---------------------------------------------------------------------------
@model_app.command("download")
def model_download(
    version: str = typer.Option(
        "latest",
        "--version", "-v",
        help=(
            "Model version to download (e.g. 'v1.0', 'latest').\n"
            "Use 'latest' for the most recent release."
        ),
        rich_help_panel="Model Options",
    ),
):
    """Download a pre-trained Geneformer model.

    Models are cached locally so you only need to download once.
    If you have not downloaded a model yet, CTC-Detect will
    prompt you to run this command.
    """
    print_banner()

    # Resolve version to repo ID
    try:
        repo_id = get_version(version)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    cache_path = get_model_cache_path(version)

    console.print(f"Model:    {version}")
    console.print(f"Repo:     {repo_id}")
    console.print(f"Cache:    {cache_path}")
    console.print()

    # Check if already cached
    if cache_path.exists() and any(cache_path.iterdir()):
        console.print(f"[yellow]Note:[/yellow] Model already exists at {cache_path}")
        overwrite = typer.confirm("Re-download?", default=False)
        if not overwrite:
            console.print("[green]✓[/green] Using cached model.")
            return

    # Download using huggingface_hub
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        console.print("[red]Error:[/red] huggingface_hub is not installed.")
        console.print("Install it with: pip install huggingface_hub")
        raise SystemExit(1)

    # Check available disk space (require at least 2 GB free)
    try:
        import shutil
        disk_usage = shutil.disk_usage(str(cache_path))
        free_gb = disk_usage.free / (1024 ** 3)
        if free_gb < 2.0:
            console.print(
                f"[red]Error:[/red] Insufficient disk space "
                f"({free_gb:.1f} GB free, need at least 2 GB)."
            )
            raise SystemExit(1)
    except OSError:
        pass  # If we can't check, proceed anyway

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(f"Downloading model '{version}'...", total=None)

            snapshot_download(
                repo_id=repo_id,
                local_dir=str(cache_path),
                local_dir_use_symlinks=False,
            )
    except ConnectionError as e:
        console.print(f"\n[red]Error:[/red] Network failure: {e}")
        console.print("Check your network connection and try again.")
        raise SystemExit(1)
    except OSError as e:
        if "No space left on device" in str(e):
            console.print("\n[red]Error:[/red] Disk full during download.")
            console.print("Free up space and try again.")
        else:
            console.print(f"\n[red]Error:[/red] OS error: {e}")
        raise SystemExit(1)
    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg or "unauthorized" in error_msg or "forbidden" in error_msg:
            console.print(f"\n[red]Error:[/red] Access denied for repo '{repo_id}'.")
            console.print("This model may require authentication.")
            console.print("Visit https://huggingface.co/settings/tokens to set up a token.")
        elif "404" in error_msg or "not found" in error_msg:
            console.print(f"\n[red]Error:[/red] Model repo '{repo_id}' not found.")
            console.print("Check the version alias and try again.")
        elif "network" in error_msg or "connection" in error_msg or "timeout" in error_msg:
            console.print(f"\n[red]Error:[/red] Network error: {e}")
            console.print("Check your network connection and try again.")
        else:
            console.print(f"\n[red]Error:[/red] Download failed: {e}")
            console.print("Check your network connection and try again.")
        raise SystemExit(1)

    console.print(f"\n[green]✓[/green] Model downloaded to {cache_path}")


@model_app.command("list")
def model_list():
    """List available models and their download status."""
    print_banner()

    table = Table(title="Available Models")
    table.add_column("Alias", style="bold cyan")
    table.add_column("Description")
    table.add_column("Repo")
    table.add_column("Status")

    for alias, meta in MODEL_REGISTRY.items():
        cache_path = get_model_cache_path(alias)
        if cache_path.exists() and any(cache_path.iterdir()):
            status = "[green]Downloaded[/green]"
        else:
            status = "[dim]Not downloaded[/dim]"

        table.add_row(alias, meta["description"], meta["repo"], status)

    console.print(table)
    console.print()
    console.print("Download a model with: ctc-detect model download")


# ---------------------------------------------------------------------------
# EVALUATE command
# ---------------------------------------------------------------------------
@app.command()
def evaluate(
    predictions: str = typer.Option(
        ...,
        "--predictions", "-p",
        help=(
            "Path to predictions CSV from 'ctc-detect run'.\n"
            "Should contain columns: barcode, ctc_probability, predicted_label."
        ),
        rich_help_panel="Input/Output",
    ),
    ground_truth: str = typer.Option(
        None,
        "--ground-truth", "-g",
        help=(
            "Optional path to ground-truth CSV with 'barcode' and 'is_ctc' columns.\n"
            "If provided, AUROC, AUPRC, and confusion matrix will be computed."
        ),
        rich_help_panel="Input/Output",
    ),
    output: str = typer.Option(
        None,
        "--output", "-o",
        help=(
            "Output directory for evaluation reports.\n"
            "Defaults to the same directory as the predictions file."
        ),
        rich_help_panel="Input/Output",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold", "-t",
        help="Threshold for binary CTC calls (default 0.5).",
        rich_help_panel="Model Options",
    ),
):
    """Evaluate CTC detection results.

    With ground truth: computes AUROC, AUPRC, sensitivity, specificity,
    confusion matrix, and generates ROC/PR curves.

    Without ground truth: shows score distribution stats and CTC call counts.
    """
    print_banner()

    pred_path = validate_input_path(predictions, "Predictions file")

    # Determine output directory
    if output:
        out_path = validate_output_path(output)
    else:
        out_path = pred_path.parent

    out_path.mkdir(parents=True, exist_ok=True)

    # Load predictions
    import pandas as pd
    import numpy as np

    pred_df = pd.read_csv(pred_path)
    required_cols = {"barcode", "ctc_probability", "predicted_label"}
    missing = required_cols - set(pred_df.columns)
    if missing:
        console.print(f"[red]Error:[/red] Predictions CSV missing columns: {missing}")
        raise SystemExit(1)

    console.print(f"Loaded {len(pred_df)} predictions from {pred_path}")

    if ground_truth:
        gt_path = validate_input_path(ground_truth, "Ground truth file")
        gt_df = pd.read_csv(gt_path)

        if "barcode" not in gt_df.columns or "is_ctc" not in gt_df.columns:
            console.print("[red]Error:[/red] Ground truth CSV must have 'barcode' and 'is_ctc' columns.")
            raise SystemExit(1)

        # Merge on barcode
        merged = pred_df.merge(gt_df[["barcode", "is_ctc"]], on="barcode", how="inner")
        console.print(f"Matched {len(merged)} cells with ground truth.")

        if len(merged) == 0:
            console.print("[red]Error:[/red] No barcodes matched between predictions and ground truth.")
            raise SystemExit(1)

        y_true = merged["is_ctc"].values.astype(int)
        y_scores = merged["ctc_probability"].values

        _run_evaluation_with_ground_truth(y_true, y_scores, out_path, threshold)
    else:
        _run_evaluation_without_ground_truth(pred_df, out_path, threshold)


def _run_evaluation_with_ground_truth(y_true, y_scores, out_path, threshold):
    """Compute full metrics and generate evaluation report with ground truth."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        confusion_matrix, precision_recall_curve, roc_curve,
        classification_report,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Compute metrics
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    y_pred = (y_scores >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    # Print summary
    console.print(f"\n[bold]Evaluation Results (threshold={threshold})[/bold]")
    console.print(f"  AUROC:        {auroc:.4f}")
    console.print(f"  AUPRC:        {auprc:.4f}")
    console.print(f"  F1:           {f1:.4f}")
    console.print(f"  Sensitivity:  {sensitivity:.4f}")
    console.print(f"  Specificity:  {specificity:.4f}")
    console.print(f"  PPV:          {ppv:.4f}")
    console.print(f"  NPV:          {npv:.4f}")
    console.print(f"\n  Confusion Matrix (threshold={threshold}):")
    console.print(f"                 Predicted")
    console.print(f"                 non-CTC    CTC")
    console.print(f"  Actual non-CTC  {tn:6d}  {fp:6d}")
    console.print(f"  Actual CTC      {fn:6d}  {tp:6d}")

    # Classification report
    console.print(f"\n{classification_report(y_true, y_pred, target_names=['non-CTC', 'CTC'], zero_division=0)}")

    # Generate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="blue")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    roc_path = out_path / "roc_curve.png"
    fig.savefig(str(roc_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  ROC curve saved to {roc_path}")

    # Generate PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, "r-", linewidth=2, label=f"AUPRC = {auprc:.4f}")
    ax.axhline(y=y_true.mean(), color="k", linestyle="--", alpha=0.5, label=f"Baseline = {y_true.mean():.4f}")
    ax.fill_between(recall, precision, alpha=0.1, color="red")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    pr_path = out_path / "pr_curve.png"
    fig.savefig(str(pr_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  PR curve saved to {pr_path}")

    # Generate confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["non-CTC", "CTC"], yticklabels=["non-CTC", "CTC"],
           xlabel="Predicted label", ylabel="True label",
           title=f"Confusion Matrix (threshold={threshold})")
    thresh_color = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh_color else "black",
                    fontsize=16, fontweight="bold")
    cm_path = out_path / "confusion_matrix.png"
    fig.savefig(str(cm_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  Confusion matrix saved to {cm_path}")

    # Write text report
    report_lines = [
        "=" * 50,
        "CTC-DETECT EVALUATION REPORT",
        "=" * 50,
        "",
        f"Total cells evaluated: {len(y_true)}",
        f"Ground truth CTCs: {y_true.sum()} ({y_true.mean()*100:.1f}%)",
        f"Ground truth non-CTCs: {(1-y_true).sum()} ({(1-y_true).mean()*100:.1f}%)",
        "",
        f"Threshold: {threshold}",
        "",
        "Metrics:",
        f"  AUROC:        {auroc:.4f}",
        f"  AUPRC:        {auprc:.4f}",
        f"  F1:           {f1:.4f}",
        f"  Sensitivity:  {sensitivity:.4f}",
        f"  Specificity:  {specificity:.4f}",
        f"  PPV:          {ppv:.4f}",
        f"  NPV:          {npv:.4f}",
        "",
        f"Confusion Matrix (threshold={threshold}):",
        f"  TP: {tp}  FP: {fp}",
        f"  FN: {fn}  TN: {tn}",
        "",
        "=" * 50,
    ]
    report_path = out_path / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    console.print(f"  Evaluation report saved to {report_path}")

    console.print(f"\n[green]✓[/green] Evaluation complete. Results in {out_path}")


def _run_evaluation_without_ground_truth(pred_df, out_path, threshold):
    """Show score distribution stats without ground truth."""
    import numpy as np

    scores = pred_df["ctc_probability"].values
    n_ctc = (scores >= threshold).sum()
    n_non_ctc = (scores < threshold).sum()

    console.print(f"\n[bold]Score Distribution (no ground truth)[/bold]")
    console.print(f"  Total cells: {len(pred_df)}")
    console.print(f"  CTC calls (prob >= {threshold}): {n_ctc} ({n_ctc/len(pred_df)*100:.1f}%)")
    console.print(f"  Non-CTC calls (prob < {threshold}): {n_non_ctc} ({n_non_ctc/len(pred_df)*100:.1f}%)")
    console.print(f"\n  Score statistics:")
    console.print(f"    Mean:   {scores.mean():.4f}")
    console.print(f"    Median: {np.median(scores):.4f}")
    console.print(f"    Std:    {scores.std():.4f}")
    console.print(f"    Min:    {scores.min():.4f}")
    console.print(f"    Max:    {scores.max():.4f}")

    # Histogram
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(scores, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(x=threshold, color="red", linestyle="--", label=f"Threshold ({threshold})")
    ax.set_xlabel("CTC Probability")
    ax.set_ylabel("Number of Cells")
    ax.set_title("CTC Probability Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    hist_path = out_path / "score_distribution.png"
    fig.savefig(str(hist_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"\n  Score distribution plot saved to {hist_path}")

    console.print(f"\n[green]✓[/green] Evaluation complete. Results in {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()
