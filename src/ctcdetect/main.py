"""CTC-Detect CLI — Geneformer-based circulating tumor cell detection.

Detect circulating tumor cells (CTCs) from single-cell RNA-seq data
produced by 10x Genomics Cell Ranger. Outputs per-cell CTC probability
scores, UMAP visualizations, and clinical summary reports.
"""

import typer
from typing import List, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ctcdetect.utils import validate_input_path, validate_output_path, print_banner
from ctcdetect.config import (
    MODEL_REGISTRY,
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
    files: Optional[List[str]] = typer.Option(
        None,
        "--files", "-f",
        help=(
            "Multiple input files to process together.\n"
            "Can be used instead of --input to process multiple samples."
        ),
        rich_help_panel="Input/Output",
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
    failed_samples: list[str] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing samples...", total=len(samples))

        for sample_dir in samples:
            sample_name = sample_dir.name
            sample_output = output_path / sample_name
            progress.update(task, description=f"[bold]Processing: {sample_name}[/bold]")

            try:
                sample_output.mkdir(parents=True, exist_ok=True)
                run_detection(
                    sample_dir, sample_output,
                    threshold=threshold, skip_umap=skip_umap,
                )
                success_count += 1
            except SystemExit as e:
                fail_count += 1
                failed_samples.append(sample_name)
                console.print(f"[red]✗[/red] {sample_name} failed (exit code {e.code})")
            except Exception as e:
                fail_count += 1
                failed_samples.append(sample_name)
                console.print(f"[red]✗[/red] {sample_name} failed: {e}")

            progress.advance(task)

    # Summary
    console.print()
    console.print("=" * 50)
    summary_table = Table(title="Batch Processing Summary", show_header=False, box=None)
    summary_table.add_column("Key", style="bold cyan", min_width=18)
    summary_table.add_column("Value")
    summary_table.add_row("Total samples", str(len(samples)))
    summary_table.add_row("Successful", f"[green]{success_count}[/green]")
    summary_table.add_row("Failed", f"[red]{fail_count}[/red]" if fail_count > 0 else "0")
    if failed_samples:
        summary_table.add_row("Failed samples", ", ".join(failed_samples))
    console.print(summary_table)

    if fail_count > 0:
        console.print("[yellow]Check individual sample outputs for error details.[/yellow]")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# MULTI command
# ---------------------------------------------------------------------------
@app.command()
def multi(
    files: List[str] = typer.Argument(
        ...,
        help=(
            "List of input files to process.\n"
            "Can be Cell Ranger directories, .h5ad files, CSV/TSV files, or other supported formats."
        ),
    ),
    output: str = typer.Option(
        ...,
        "--output", "-o",
        help=(
            "Path to output directory.\n"
            "Results for all files will be written to this directory."
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
    """Run CTC detection on multiple individual files.

    Process multiple input files in a single command. Each file will be
    processed separately and results will be saved in individual subdirectories.
    Supports all file formats that the 'run' command supports.
    """
    print_banner()
    input_paths = [validate_input_path(f, "Input file") for f in files]
    output_path = validate_output_path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"Processing {len(input_paths)} files...")
    
    from ctcdetect.detect import run_detection
    
    for i, input_path in enumerate(input_paths, 1):
        console.print(f"\n[bold]Processing file {i}/{len(input_paths)}: {input_path.name}[/bold]")
        sample_output = output_path / f"sample_{i}_{input_path.stem}"
        sample_output.mkdir(exist_ok=True)
        run_detection(input_path, sample_output, threshold=threshold, skip_umap=skip_umap)
    
    console.print(f"\n[green]✓[/green] All {len(input_paths)} samples processed. Results in {output_path}")


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

    # Disk space
    disk_free = info.get("disk_free_gb")
    disk_total = info.get("disk_total_gb")
    if disk_free is not None and disk_total is not None:
        table.add_row("Disk space (cache)", f"{disk_free} GB free / {disk_total} GB total")
    else:
        table.add_row("Disk space (cache)", "N/A")

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
            "Should contain columns: barcode, ctc_probability, predicted_label, uncertain."
        ),
        rich_help_panel="Input/Output",
    ),
    ground_truth: str = typer.Option(
        None,
        "--ground-truth", "-g",
        help=(
            "Optional path to ground-truth CSV with 'barcode' and 'true_label' columns.\n"
            "If provided, AUROC, AUPRC, sensitivity, specificity, and confusion\n"
            "matrix will be computed along with ROC/PR curve plots."
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
    import numpy as np
    import pandas as pd

    from ctcdetect.evaluate import (
        compute_metrics,
        generate_eval_report,
        generate_eval_html_report,
        plot_roc_pr,
    )

    print_banner()

    pred_path = validate_input_path(predictions, "Predictions file")

    # Determine output directory
    if output:
        out_path = validate_output_path(output)
    else:
        out_path = pred_path.parent

    out_path.mkdir(parents=True, exist_ok=True)

    # Load predictions
    pred_df = pd.read_csv(pred_path)
    # Also accept CSVs that have at least the core columns
    core_cols = {"barcode", "ctc_probability", "predicted_label"}
    core_missing = core_cols - set(pred_df.columns)
    if core_missing:
        console.print(f"[red]Error:[/red] Predictions CSV missing columns: {core_missing}")
        raise SystemExit(1)

    console.print(f"Loaded {len(pred_df)} predictions from {pred_path}")

    if ground_truth:
        gt_path = validate_input_path(ground_truth, "Ground truth file")
        gt_df = pd.read_csv(gt_path)

        if "barcode" not in gt_df.columns or "true_label" not in gt_df.columns:
            console.print("[red]Error:[/red] Ground truth CSV must have 'barcode' and 'true_label' columns.")
            raise SystemExit(1)

        # Merge on barcode
        merged = pred_df.merge(gt_df[["barcode", "true_label"]], on="barcode", how="inner")
        console.print(f"Matched {len(merged)} cells with ground truth.")

        if len(merged) == 0:
            console.print("[red]Error:[/red] No barcodes matched between predictions and ground truth.")
            raise SystemExit(1)

        y_true = merged["true_label"].values.astype(int)
        y_scores = merged["ctc_probability"].values

        # Compute metrics
        metrics = compute_metrics(y_true, y_scores, threshold)

        # Print summary
        console.print(f"\n[bold]Evaluation Results (threshold={threshold})[/bold]")
        console.print(f"  AUROC:        {metrics['auroc']:.4f}")
        console.print(f"  AUPRC:        {metrics['auprc']:.4f}")
        console.print(f"  F1:           {metrics['f1']:.4f}")
        console.print(f"  Sensitivity:  {metrics['sensitivity']:.4f}")
        console.print(f"  Specificity:  {metrics['specificity']:.4f}")
        console.print(f"  PPV:          {metrics['ppv']:.4f}")
        console.print(f"  NPV:          {metrics['npv']:.4f}")
        console.print(f"\n  Confusion Matrix (threshold={threshold}):")
        console.print("                 Predicted")
        console.print("                 non-CTC    CTC")
        console.print(f"  Actual non-CTC  {metrics['tn']:6d}  {metrics['fp']:6d}")
        console.print(f"  Actual CTC      {metrics['fn']:6d}  {metrics['tp']:6d}")

        # Classification report
        from sklearn.metrics import classification_report
        y_pred = (y_scores >= threshold).astype(int)
        console.print(f"\n{classification_report(y_true, y_pred, target_names=['non-CTC', 'CTC'], zero_division=0)}")

        # Generate reports
        generate_eval_report(metrics, out_path)
        console.print(f"  Text report saved to {out_path / 'eval_report.txt'}")

        generate_eval_html_report(metrics, out_path)
        console.print(f"  HTML report saved to {out_path / 'eval_report.html'}")

        # Generate plots
        plot_roc_pr(metrics, out_path)
        console.print(f"  ROC curve saved to {out_path / 'roc.png'}")
        console.print(f"  PR curve saved to {out_path / 'pr.png'}")

        console.print(f"\n[green]✓[/green] Evaluation complete. Results in {out_path}")

    else:
        # No ground truth: show score distribution stats
        scores = pred_df["ctc_probability"].values
        n_ctc = int((scores >= threshold).sum())
        n_non_ctc = int((scores < threshold).sum())

        console.print("\n[bold]Score Distribution (no ground truth)[/bold]")
        console.print(f"  Total cells: {len(pred_df)}")
        console.print(f"  CTC calls (prob >= {threshold}): {n_ctc} ({n_ctc/len(pred_df)*100:.1f}%)")
        console.print(f"  Non-CTC calls (prob < {threshold}): {n_non_ctc} ({n_non_ctc/len(pred_df)*100:.1f}%)")
        console.print("\n  Score statistics:")
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
