"""Evaluate command for CTC-Detect CLI."""

import typer
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from ctcdetect.cli.utils import validate_input_path, validate_output_path, print_banner, console
from ctcdetect.evaluation.metrics import compute_metrics
from ctcdetect.evaluation.reports import generate_report, generate_html_report
from ctcdetect.evaluation.plots import plot_roc_pr, plot_score_distribution


@typer.Typer(
    help="Evaluate CTC detection results.",
)
def evaluate_app():
    pass


@evaluate_app.command()
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
    ground_truth: Optional[str] = typer.Option(
        None,
        "--ground-truth", "-g",
        help=(
            "Optional path to ground-truth CSV with 'barcode' and 'true_label' columns.\n"
            "If provided, AUROC, AUPRC, sensitivity, specificity, and confusion\n"
            "matrix will be computed along with ROC/PR curve plots."
        ),
        rich_help_panel="Input/Output",
    ),
    output: Optional[str] = typer.Option(
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
    confusion matrix, and generates ROC/PR curve plots.

    Without ground truth: shows score distribution stats and CTC call counts.
    """
    from sklearn.metrics import classification_report

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
    core_cols = {"barcode", "ctc_probability", "predicted_label"}
    core_missing = core_cols - set(pred_df.columns)
    if core_missing:
        console.print(f"[red]Error:[/red] Predictions CSV missing columns: {core_missing}")
        raise typer.Exit(1)

    console.print(f"Loaded {len(pred_df)} predictions from {pred_path}")

    if ground_truth:
        gt_path = validate_input_path(ground_truth, "Ground truth file")
        gt_df = pd.read_csv(gt_path)

        if "barcode" not in gt_df.columns or "true_label" not in gt_df.columns:
            console.print("[red]Error:[/red] Ground truth CSV must have 'barcode' and 'true_label' columns.")
            raise typer.Exit(1)

        # Merge on barcode
        merged = pred_df.merge(gt_df[["barcode", "true_label"]], on="barcode", how="inner")
        console.print(f"Matched {len(merged)} cells with ground truth.")

        if len(merged) == 0:
            console.print("[red]Error:[/red] No barcodes matched between predictions and ground truth.")
            raise typer.Exit(1)

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
        y_pred = (y_scores >= threshold).astype(int)
        console.print(f"\n{classification_report(y_true, y_pred, target_names=['non-CTC', 'CTC'], zero_division=0)}")

        # Generate reports
        generate_report(metrics, out_path)
        console.print(f"  Text report saved to {out_path / 'eval_report.txt'}")

        generate_html_report(metrics, out_path)
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
        plot_score_distribution(scores, out_path / "score_distribution.png", threshold)
        console.print(f"\n  Score distribution plot saved to {out_path / 'score_distribution.png'}")

        console.print(f"\n[green]✓[/green] Evaluation complete. Results in {out_path}")