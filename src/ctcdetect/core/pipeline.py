"""High-level pipeline orchestration for CTC-Detect.

Provides a clean, configurable API for running the full detection pipeline
with customizable components.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import scanpy as sc
from rich.console import Console

from ctcdetect.config import get_config
from ctcdetect.core.detect import detect_format
from ctcdetect.core.model import check_geneformer_available, load_model
from ctcdetect.core.preprocess import (
    load_data,
    map_genes_to_ensembl,
    normalize,
    run_qc,
    validate_input,
)
from ctcdetect.evaluation import (
    generate_html_report,
    generate_report,
    generate_umap,
)

console = Console()


@dataclass
class PipelineConfig:
    """Configuration for the detection pipeline."""

    threshold: float = 0.5
    skip_umap: bool = False
    skip_reports: bool = False
    cancer_type: str | None = None
    # Custom hooks
    on_preprocess: Callable[[sc.AnnData], sc.AnnData] | None = None
    on_tokenize: Callable[[Any], Any] | None = None
    on_inference: Callable[[Any, Any], Any] | None = None
    on_results: Callable[[pd.DataFrame], pd.DataFrame] | None = None


@dataclass
class PipelineResults:
    """Results from pipeline execution."""

    results_df: pd.DataFrame
    adata: sc.AnnData
    output_path: Path
    metrics: dict = field(default_factory=dict)


class CTCDetectionPipeline:
    """High-level orchestration for CTC detection.

    This class provides a clean API for running the full detection pipeline
    with configurable components and hooks for customization.

    Example:
        >>> pipeline = CTCDetectionPipeline(threshold=0.6)
        >>> results = pipeline.run(
        ...     input_path="data/raw/sample",
        ...     output_path="results/sample"
        ... )
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        config_path: Path | None = None,
    ):
        """Initialize the pipeline.

        Args:
            config: PipelineConfig object. If None, uses defaults.
            config_path: Path to preprocessing config YAML.
        """
        self.config = config or PipelineConfig()
        if config_path:
            self.preprocess_config = get_config(config_path)
        else:
            self.preprocess_config = get_config()

        self._model = None
        self._device = None

    def run(
        self,
        input_path: Path,
        output_path: Path,
        cancer_type: str | None = None,
        threshold: float | None = None,
        skip_umap: bool | None = None,
    ) -> PipelineResults:
        """Run the full detection pipeline.

        Args:
            input_path: Path to Cell Ranger output directory or .h5ad file.
            output_path: Path to output directory.
            cancer_type: Cancer type for model selection (not yet implemented).
            threshold: Probability threshold for CTC calls.
            skip_umap: Skip UMAP visualization.

        Returns:
            PipelineResults with results DataFrame and paths.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Override config if provided
        threshold = threshold if threshold is not None else self.config.threshold
        skip_umap = skip_umap if skip_umap is not None else self.config.skip_umap

        console.print(f"[bold]Input:[/bold]  {input_path}")
        console.print(f"[bold]Output:[/bold] {output_path}")
        console.print()

        # Step 1: Validate input
        console.print("[bold]Step 1/6:[/bold] Validating input...")
        validate_input(input_path)
        fmt = detect_format(input_path)
        console.print(f"  Format: {fmt}")

        # Step 2: Load data
        console.print("[bold]Step 2/6:[/bold] Loading data...")
        adata = load_data(input_path)
        console.print(f"  Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")

        # Step 3: Preprocess (QC, normalize, map genes)
        console.print("[bold]Step 3/6:[/bold] Preprocessing...")
        adata = run_qc(adata)
        adata = normalize(adata)

        # Map gene symbols to Ensembl IDs
        import pickle

        from ctcdetect.config.paths import GENE_MAPPING
        with open(GENE_MAPPING, "rb") as f:
            gene_mapping = pickle.load(f)
        adata = map_genes_to_ensembl(adata, gene_mapping)

        # Custom preprocess hook
        if self.config.on_preprocess:
            adata = self.config.on_preprocess(adata)

        # Step 4: Load model
        console.print("[bold]Step 4/6:[/bold] Loading model...")
        if not check_geneformer_available():
            raise RuntimeError("Geneformer not available. Run 'ctc-detect model download' first.")
        self._model, self._device = load_model()

        # Step 5: Tokenize and inference
        console.print("[bold]Step 5/6:[/bold] Tokenizing and running inference...")
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

        from ctcdetect.core.detect import _run_inference, _tokenize

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Tokenizing...", total=1.0)
            dataset, _ = _tokenize(adata, progress, task)

            task = progress.add_task("Running inference...", total=1.0)
            barcodes, probs, preds, uncertain = _run_inference(
                self._model, self._device, dataset, progress, task
            )

        # Build results DataFrame
        results_df = pd.DataFrame({
            "barcode": barcodes,
            "ctc_probability": probs,
            "predicted_label": preds,
            "uncertain": uncertain,
        })
        results_df["ctc_call"] = (results_df["ctc_probability"] >= threshold).astype(int)

        # Custom results hook
        if self.config.on_results:
            results_df = self.config.on_results(results_df)

        # Step 6: Save results and generate outputs
        console.print("[bold]Step 6/6:[/bold] Saving results and generating reports...")
        output_path.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path / "ctc_probabilities.csv", index=False)
        console.print(f"  Results saved to {output_path / 'ctc_probabilities.csv'}")

        # Generate UMAP
        if not skip_umap:
            generate_umap(adata, results_df, output_path / "umap.png")

        # Generate reports
        if not self.config.skip_reports:
            generate_report(results_df, output_path, threshold)
            generate_html_report(results_df, output_path, threshold)

        # Calculate summary metrics
        metrics = {
            "total_cells": len(results_df),
            "ctc_calls": int((results_df["ctc_probability"] >= threshold).sum()),
            "non_ctc_calls": int((results_df["ctc_probability"] < threshold).sum()),
            "uncertain_calls": int(results_df["uncertain"].sum()),
            "mean_probability": float(results_df["ctc_probability"].mean()),
            "median_probability": float(results_df["ctc_probability"].median()),
        }

        console.print(f"\n[green]✓[/green] Detection complete. Results in {output_path}")

        return PipelineResults(
            results_df=results_df,
            adata=adata,
            output_path=output_path,
            metrics=metrics,
        )

    def run_batch(
        self,
        input_dir: Path,
        output_dir: Path,
        **kwargs,
    ) -> list[PipelineResults]:
        """Run detection on multiple samples in a directory.

        Args:
            input_dir: Directory containing sample subdirectories.
            output_dir: Output directory for all results.
            **kwargs: Additional arguments passed to run().

        Returns:
            List of PipelineResults for each sample.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        samples = sorted([d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

        if not samples:
            raise ValueError(f"No sample directories found in {input_dir}")

        console.print(f"Found {len(samples)} samples to process.\n")

        results = []
        for sample_dir in samples:
            sample_name = sample_dir.name
            sample_output = output_dir / sample_name
            console.print(f"\n[bold cyan]Processing: {sample_name}[/bold cyan]")
            try:
                result = self.run(sample_dir, sample_output, **kwargs)
                results.append(result)
            except Exception as e:
                console.print(f"[red]✗[/red] {sample_name} failed: {e}")
                # Create failed result
                results.append(PipelineResults(
                    results_df=pd.DataFrame(),
                    adata=sc.AnnData(),
                    output_path=sample_output,
                    metrics={"error": str(e)},
                ))

        return results


def run_detection(
    input_path: Path,
    output_path: Path,
    cancer_type: str | None = None,
    threshold: float = 0.5,
    skip_umap: bool = False,
) -> PipelineResults:
    """Convenience function for single-sample detection.

    This is a thin wrapper around CTCDetectionPipeline.run() for backward compatibility.

    Args:
        input_path: Path to Cell Ranger output or .h5ad file.
        output_path: Output directory.
        cancer_type: Cancer type (not yet implemented).
        threshold: CTC probability threshold.
        skip_umap: Skip UMAP generation.

    Returns:
        PipelineResults object.
    """
    pipeline = CTCDetectionPipeline()
    return pipeline.run(
        input_path=input_path,
        output_path=output_path,
        cancer_type=cancer_type,
        threshold=threshold,
        skip_umap=skip_umap,
    )


__all__ = [
    "CTCDetectionPipeline",
    "PipelineConfig",
    "PipelineResults",
    "run_detection",
]
