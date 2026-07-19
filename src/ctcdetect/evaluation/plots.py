"""Visualization and plotting utilities for CTC-Detect.

Generates UMAP plots, ROC/PR curves, and score distributions.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from rich.console import Console

console = Console()


def generate_umap(
    adata: sc.AnnData,
    results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate a four-panel UMAP figure showing CTC detection results.

    Panels:
    1. UMAP colored by CTC probability
    2. UMAP colored by predicted label
    3. UMAP colored by uncertainty flag
    4. Score distribution histogram

    Args:
        adata: AnnData object with expression data (will compute UMAP if needed).
        results_df: DataFrame with barcode, ctc_probability, predicted_label, uncertain.
        output_path: Path to write the figure (PNG).
    """
    # Compute UMAP if not already done
    if "X_umap" not in adata.obsm:
        console.print("  Computing UMAP on expression data...")
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
        sc.pp.pca(adata, n_comps=30)
        sc.pp.neighbors(adata, n_pcs=30)
        sc.tl.umap(adata)

    # Map results to adata cells
    barcode_to_idx = {bc: i for i, bc in enumerate(adata.obs_names)}
    n = adata.shape[0]

    umap_probs = np.full(n, np.nan)
    umap_preds = np.full(n, np.nan)
    umap_uncertain = np.zeros(n, dtype=bool)

    for _, row in results_df.iterrows():
        bc = str(row["barcode"])
        if bc in barcode_to_idx:
            idx = barcode_to_idx[bc]
            umap_probs[idx] = row["ctc_probability"]
            umap_preds[idx] = row["predicted_label"]
            umap_uncertain[idx] = row["uncertain"]

    umap_coords = adata.obsm["X_umap"]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("CTC-Detect Results", fontsize=16, fontweight="bold")

    # Panel 1: CTC Probability
    ax = axes[0, 0]
    valid = ~np.isnan(umap_probs)
    if valid.any():
        scatter = ax.scatter(
            umap_coords[valid, 0], umap_coords[valid, 1],
            c=umap_probs[valid], cmap="RdYlBu_r", s=3, alpha=0.7,
            vmin=0, vmax=1,
        )
        plt.colorbar(scatter, ax=ax, label="CTC Probability")
    ax.scatter(
        umap_coords[~valid, 0], umap_coords[~valid, 1],
        c="lightgray", s=1, alpha=0.3,
    )
    ax.set_title("Predicted CTC Probability")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Panel 2: Predicted Label
    ax = axes[0, 1]
    valid = ~np.isnan(umap_preds)
    if valid.any():
        cmap = ListedColormap(["#2196F3", "#F44336"])
        ax.scatter(
            umap_coords[valid, 0], umap_coords[valid, 1],
            c=umap_preds[valid], cmap=cmap, s=3, alpha=0.7,
        )
        # Legend
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
                   markersize=8, label="Non-CTC (0)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#F44336",
                   markersize=8, label="CTC (1)"),
        ]
        ax.legend(handles=legend_elements, loc="best", markerscale=1)
    ax.scatter(
        umap_coords[~valid, 0], umap_coords[~valid, 1],
        c="lightgray", s=1, alpha=0.3,
    )
    ax.set_title("Predicted Label")
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
    ax.set_title("Uncertainty Flag (max prob < 0.6)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=5)

    # Panel 4: Score Distribution
    ax = axes[1, 1]
    valid_probs = umap_probs[~np.isnan(umap_probs)]
    if len(valid_probs) > 0:
        ax.hist(valid_probs, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(x=0.5, color="red", linestyle="--", label="Threshold (0.5)")
        ax.legend()
    ax.set_xlabel("CTC Probability")
    ax.set_ylabel("Number of Cells")
    ax.set_title("Score Distribution")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"  UMAP saved to {output_path}")


def plot_roc_pr(metrics: dict, output_path: Path) -> None:
    """Generate ROC and PR curve PNG plots.

    Args:
        metrics: Output from ``compute_metrics`` (must contain curve data).
        output_path: Directory to write ``roc.png`` and ``pr.png``.
    """
    # ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(metrics["fpr"], metrics["tpr"], "b-", linewidth=2,
            label=f"AUROC = {metrics['auroc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.fill_between(metrics["fpr"], metrics["tpr"], alpha=0.1, color="blue")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    roc_path = output_path / "roc.png"
    fig.savefig(str(roc_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PR curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(metrics["recall_curve"], metrics["precision_curve"],
            "r-", linewidth=2, label=f"AUPRC = {metrics['auprc']:.4f}")
    baseline = metrics["prevalence"]
    ax.axhline(y=baseline, color="k", linestyle="--", alpha=0.5,
               label=f"Baseline = {baseline:.4f}")
    ax.fill_between(metrics["recall_curve"], metrics["precision_curve"],
                    alpha=0.1, color="red")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    pr_path = output_path / "pr.png"
    fig.savefig(str(pr_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    console.print(f"  ROC curve saved to {roc_path}")
    console.print(f"  PR curve saved to {pr_path}")


def plot_score_distribution(
    scores: np.ndarray,
    output_path: Path,
    threshold: float = 0.5,
    title: str = "CTC Probability Score Distribution",
) -> None:
    """Generate a histogram of CTC probability scores.

    Args:
        scores: Array of CTC probability scores.
        output_path: Path to write the figure.
        threshold: Classification threshold to mark on plot.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(scores, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(x=threshold, color="red", linestyle="--", label=f"Threshold ({threshold})")
    ax.set_xlabel("CTC Probability")
    ax.set_ylabel("Number of Cells")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    console.print(f"  Score distribution saved to {output_path}")


__all__ = [
    "generate_umap",
    "plot_roc_pr",
    "plot_score_distribution",
]
