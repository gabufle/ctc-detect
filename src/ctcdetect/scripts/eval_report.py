#!/usr/bin/env python
"""Generate evaluation artifacts from CTC model predictions and embeddings.

This script is intentionally standalone so evaluation figures and numeric
summaries can be regenerated without running model inference.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_LABELS = ["PBMC", "CTC"]
EMBEDDING_COLUMN_PREFIXES = ("embedding_", "emb_", "cls_", "pooled_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create UMAP, confusion-matrix, AUROC bootstrap CI, and permutation "
            "test reports from CTC predictions, labels, and Geneformer embeddings."
        )
    )
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help=(
            "CSV with at least barcode, ctc_probability, and optionally "
            "predicted_label columns."
        ),
    )
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="CSV with barcode and true_label columns. May also include cancer_type.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        help=(
            "Geneformer CLS/pooled embeddings as .csv, .npy, or .npz. CSV files "
            "should include barcode plus embedding columns. .npy/.npz files must "
            "align with --embedding-barcodes or, if omitted, prediction row order. "
            "If omitted, embeddings are read from prediction columns named "
            "embedding_*, emb_*, cls_*, or pooled_*."
        ),
    )
    parser.add_argument(
        "--embedding-barcodes",
        type=Path,
        help=(
            "Optional barcode list for .npy/.npz embeddings. Accepts a one-column "
            "CSV/TSV/TXT or a CSV with a barcode column."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for PNG outputs and summary.json (default: results/).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used if predictions CSV lacks predicted_label (default: 0.5).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for AUROC CI (default: 1000).",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=1000,
        help="Number of label permutations for null AUROC test (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap/permutation/UMAP (default: 42).",
    )
    parser.add_argument(
        "--group-col",
        default="cancer_type",
        help=(
            "Optional column for per-group confusion matrices when present "
            "(default: cancer_type)."
        ),
    )
    parser.add_argument(
        "--embedding-key",
        default="embeddings",
        help="Array key to read from .npz embeddings (default: embeddings).",
    )
    parser.add_argument(
        "--umap-max-points",
        type=int,
        default=20000,
        help=(
            "Maximum points used for UMAP plotting. Larger datasets are sampled "
            "by true/predicted label strata so misclassification regions remain "
            "visible. Use 0 to plot all rows (default: 20000)."
        ),
    )
    parser.add_argument(
        "--max-group-plots",
        type=int,
        default=25,
        help=(
            "Maximum per-group confusion matrix PNGs to render. Numeric group "
            "metrics are still written for all groups. Use 0 to disable the cap "
            "(default: 25)."
        ),
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=2,
        help=(
            "Minimum rows required before rendering a per-group confusion PNG. "
            "Numeric group metrics are still written for all groups (default: 2)."
        ),
    )
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def _coerce_binary_labels(values: pd.Series, column: str) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(values):
        labels = values.astype(int).to_numpy()
    else:
        normalized = values.astype(str).str.strip().str.lower()
        mapping = {
            "ctc": 1,
            "tumor": 1,
            "cancer": 1,
            "positive": 1,
            "pos": 1,
            "1": 1,
            "true": 1,
            "pbmc": 0,
            "non_ctc": 0,
            "non-ctc": 0,
            "normal": 0,
            "negative": 0,
            "neg": 0,
            "0": 0,
            "false": 0,
        }
        if not normalized.isin(mapping).all():
            bad = sorted(normalized[~normalized.isin(mapping)].unique())
            raise ValueError(f"Column {column!r} contains non-binary labels: {bad}")
        labels = normalized.map(mapping).astype(int).to_numpy()

    unique = set(np.unique(labels).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"Column {column!r} must contain binary labels 0/1.")
    return labels


def load_predictions_and_labels(
    predictions_path: Path,
    labels_path: Path,
    threshold: float,
) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_path)
    labels = pd.read_csv(labels_path)

    _require_columns(predictions, {"barcode", "ctc_probability"}, predictions_path)
    _require_columns(labels, {"barcode", "true_label"}, labels_path)

    if "predicted_label" not in predictions.columns:
        predictions["predicted_label"] = (
            predictions["ctc_probability"].astype(float) >= threshold
        ).astype(int)

    merged = predictions.merge(labels, on="barcode", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No overlapping barcodes between predictions and labels.")

    missing_predictions = len(predictions) - len(merged)
    missing_labels = len(labels) - len(merged)
    if missing_predictions or missing_labels:
        print(
            "Warning: matched "
            f"{len(merged)} rows; dropped {missing_predictions} prediction rows and "
            f"{missing_labels} label rows without a barcode match."
        )

    merged["true_label"] = _coerce_binary_labels(merged["true_label"], "true_label")
    merged["predicted_label"] = _coerce_binary_labels(
        merged["predicted_label"], "predicted_label"
    )
    merged["ctc_probability"] = merged["ctc_probability"].astype(float)
    return merged


def _read_barcode_file(path: Path) -> list[str]:
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        if "barcode" in df.columns:
            return df["barcode"].astype(str).tolist()
        return df.iloc[:, 0].astype(str).tolist()
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _load_npz_array(path: Path, key: str) -> np.ndarray:
    data = np.load(path)
    if key in data:
        return np.asarray(data[key])
    keys = list(data.keys())
    if len(keys) == 1:
        return np.asarray(data[keys[0]])
    raise ValueError(
        f"{path} contains keys {keys}; pass --embedding-key to choose one."
    )


def load_embeddings(
    embeddings_path: Path | None,
    merged: pd.DataFrame,
    embedding_barcodes_path: Path | None,
    embedding_key: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    if embeddings_path is None:
        embedding_cols = [
            c
            for c in merged.columns
            if c.startswith(EMBEDDING_COLUMN_PREFIXES)
        ]
        if not embedding_cols:
            raise ValueError(
                "No --embeddings file was provided and no embedding columns were "
                "found in predictions. Expected columns named embedding_*, emb_*, "
                "cls_*, or pooled_*."
            )
        return merged.copy(), merged[embedding_cols].to_numpy(dtype=float)

    suffix = embeddings_path.suffix.lower()
    if suffix == ".csv":
        emb_df = pd.read_csv(embeddings_path)
        _require_columns(emb_df, {"barcode"}, embeddings_path)
        feature_cols = [c for c in emb_df.columns if c != "barcode"]
        if not feature_cols:
            raise ValueError(f"{embeddings_path} has no embedding feature columns.")
        emb_df["barcode"] = emb_df["barcode"].astype(str)
        aligned = merged.merge(emb_df, on="barcode", how="inner", validate="one_to_one")
        if aligned.empty:
            raise ValueError("No overlapping barcodes between labels and embeddings.")
        embeddings = aligned[feature_cols].to_numpy(dtype=float)
        return aligned, embeddings

    if suffix == ".npy":
        embeddings = np.load(embeddings_path, mmap_mode="r")
    elif suffix == ".npz":
        embeddings = _load_npz_array(embeddings_path, embedding_key)
    else:
        raise ValueError("Embeddings must be .csv, .npy, or .npz.")

    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be a 2D array, got shape {embeddings.shape}."
        )

    if embedding_barcodes_path is not None:
        barcodes = _read_barcode_file(embedding_barcodes_path)
        if len(barcodes) != len(embeddings):
            raise ValueError(
                f"{embedding_barcodes_path} has {len(barcodes)} barcodes but "
                f"embeddings have {len(embeddings)} rows."
            )
        emb_df = pd.DataFrame({"barcode": barcodes})
        emb_df["__embedding_row"] = np.arange(len(embeddings))
        aligned = merged.merge(emb_df, on="barcode", how="inner", validate="one_to_one")
        if aligned.empty:
            raise ValueError("No overlapping barcodes between labels and embeddings.")
        return aligned, embeddings[aligned["__embedding_row"].to_numpy()]

    if len(embeddings) != len(merged):
        raise ValueError(
            "For .npy/.npz embeddings without --embedding-barcodes, row count must "
            f"match matched predictions/labels ({len(merged)}), got {len(embeddings)}."
        )
    return merged.copy(), embeddings


def _safe_normalize_confusion_matrix(cm: np.ndarray) -> list[list[float]]:
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )
    return normalized.astype(float).tolist()


def compute_basic_metrics(
    y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray
) -> dict:
    from sklearn.metrics import confusion_matrix, roc_auc_score

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
    }


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    from sklearn.metrics import roc_auc_score

    bootstrapped_scores: list[float] = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample_true = y_true[idx]
        if len(np.unique(sample_true)) < 2:
            continue
        bootstrapped_scores.append(float(roc_auc_score(sample_true, y_score[idx])))

    if not bootstrapped_scores:
        raise ValueError("All bootstrap resamples contained a single class.")

    ci_low, ci_high = np.percentile(bootstrapped_scores, [2.5, 97.5])
    return float(ci_low), float(ci_high), len(bootstrapped_scores)


def permutation_auroc_pvalue(
    y_true: np.ndarray,
    y_score: np.ndarray,
    observed_auroc: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, list[float]]:
    from sklearn.metrics import roc_auc_score

    null_scores: list[float] = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(y_true)
        null_scores.append(float(roc_auc_score(shuffled, y_score)))

    # One-sided test: how often does the shuffled-label null match or exceed
    # the observed AUROC? Add-one smoothing keeps p nonzero for finite samples.
    p_value = (sum(score >= observed_auroc for score in null_scores) + 1) / (
        n_permutations + 1
    )
    return float(p_value), null_scores


def choose_umap_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose a stratified UMAP subset while preserving error strata."""
    n = len(y_true)
    if max_points <= 0 or n <= max_points:
        return np.arange(n)

    selected: list[np.ndarray] = []
    strata = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for true_label, pred_label in strata:
        idx = np.flatnonzero((y_true == true_label) & (y_pred == pred_label))
        if len(idx) == 0:
            continue

        quota = int(np.ceil(max_points * len(idx) / n))
        # Preserve small error regions whenever possible so the plot can reveal
        # whether false positives/negatives cluster in embedding space.
        if true_label != pred_label:
            quota = max(quota, min(len(idx), max(50, max_points // 20)))
        quota = min(quota, len(idx))
        selected.append(rng.choice(idx, size=quota, replace=False))

    merged = np.unique(np.concatenate(selected))
    if len(merged) > max_points:
        merged = rng.choice(merged, size=max_points, replace=False)
    return np.sort(merged)


def plot_embedding_umap(
    embeddings: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    seed: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from umap import UMAP

    rng = np.random.default_rng(seed)
    plot_idx = choose_umap_indices(y_true, y_pred, max_points, rng)
    plot_embeddings = np.asarray(embeddings[plot_idx], dtype=float)
    plot_true = y_true[plot_idx]
    plot_pred = y_pred[plot_idx]

    n_neighbors = min(15, max(2, len(plot_embeddings) - 1))
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.3,
        metric="cosine",
        random_state=seed,
    )
    coords = reducer.fit_transform(plot_embeddings)

    cmap = ListedColormap(["#4C78A8", "#F58518"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    panels = [
        (axes[0], plot_true, "True label"),
        (axes[1], plot_pred, "Predicted label"),
    ]
    for ax, labels, title in panels:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=labels,
            cmap=cmap,
            vmin=0,
            vmax=1,
            s=18,
            alpha=0.85,
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(True, alpha=0.2)

    handles, _ = scatter.legend_elements(num=[0, 1])
    fig.legend(
        handles,
        DEFAULT_LABELS,
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    suffix = "" if len(plot_idx) == len(y_true) else f" ({len(plot_idx):,}/{len(y_true):,} sampled)"
    fig.suptitle(f"Geneformer embedding UMAP{suffix}")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return coords, plot_idx


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title_suffix: str = "",
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    labels = [0, 1]
    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ConfusionMatrixDisplay(cm_raw, display_labels=DEFAULT_LABELS).plot(
        ax=axes[0], cmap="Blues", colorbar=False, values_format="d"
    )
    axes[0].set_title(f"Counts{title_suffix}")

    ConfusionMatrixDisplay(cm_norm, display_labels=DEFAULT_LABELS).plot(
        ax=axes[1], cmap="Blues", colorbar=False, values_format=".2f"
    )
    axes[1].set_title(f"Normalized by true label{title_suffix}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "raw": cm_raw.astype(int).tolist(),
        "normalized_true": _safe_normalize_confusion_matrix(cm_raw),
    }


def collect_group_confusion(
    merged: pd.DataFrame,
    group_col: str,
    output_dir: Path,
    max_group_plots: int,
    min_group_size: int,
) -> dict[str, Any]:
    if group_col not in merged.columns:
        return {}

    group_dir = output_dir / "confusion_by_group"
    groups = list(merged.groupby(group_col, dropna=False))
    groups.sort(key=lambda item: len(item[1]), reverse=True)

    rendered = 0
    group_confusion: dict[str, Any] = {}
    for group, group_df in groups:
        group_true = group_df["true_label"].to_numpy(dtype=int)
        group_pred = group_df["predicted_label"].to_numpy(dtype=int)
        cm = _confusion_matrix_array(group_true, group_pred)
        group_name = str(group)
        group_confusion[group_name] = {
            "n": int(len(group_df)),
            "n_positive": int(group_true.sum()),
            "n_negative": int(len(group_true) - group_true.sum()),
            "raw": cm.astype(int).tolist(),
            "normalized_true": _safe_normalize_confusion_matrix(cm),
            "plot": None,
        }

        plot_cap_reached = max_group_plots > 0 and rendered >= max_group_plots
        if len(group_df) < min_group_size or plot_cap_reached:
            continue

        group_dir.mkdir(exist_ok=True)
        safe_group = (
            group_name.replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(":", "_")
        )
        group_path = group_dir / f"confusion_matrix_{safe_group}.png"
        plot_confusion_matrix(
            group_true,
            group_pred,
            group_path,
            title_suffix=f" ({group_name})",
        )
        group_confusion[group_name]["plot"] = str(group_path)
        rendered += 1

    return group_confusion


def _confusion_matrix_array(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def build_summary(
    merged: pd.DataFrame,
    metrics: dict,
    ci_low: float,
    ci_high: float,
    n_bootstrap: int,
    bootstrap_valid: int,
    p_value: float,
    n_permutations: int,
    group_col: str,
    group_confusion: dict[str, Any],
    umap_indices: np.ndarray,
) -> dict[str, Any]:
    return {
        "n": metrics["n"],
        "n_positive": metrics["n_positive"],
        "n_negative": metrics["n_negative"],
        "positive_label": "CTC",
        "negative_label": "PBMC",
        "auroc": metrics["auroc"],
        "auroc_ci_95": {
            "low": ci_low,
            "high": ci_high,
            "method": "bootstrap",
            "requested_resamples": int(n_bootstrap),
            "valid_resamples": int(bootstrap_valid),
        },
        "permutation_test": {
            "p_value": p_value,
            "requested_permutations": int(n_permutations),
            "alternative": "model AUROC greater than label-shuffled null",
        },
        "confusion_matrix": {
            "labels": DEFAULT_LABELS,
            "raw": metrics["confusion_matrix"].astype(int).tolist(),
            "normalized_true": _safe_normalize_confusion_matrix(
                metrics["confusion_matrix"]
            ),
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
        },
        "per_group_confusion_matrices": {
            "group_column": group_col if group_col in merged.columns else None,
            "groups": group_confusion,
        },
        "umap": {
            "n_plotted": int(len(umap_indices)),
            "n_total": int(len(merged)),
            "sampled": bool(len(umap_indices) < len(merged)),
            "sample_strategy": "true/predicted-label stratified",
        },
    }


def configure_runtime_cache() -> None:
    """Point matplotlib/numba caches at a writable temp directory."""
    cache_dir = Path(tempfile.gettempdir()) / "ctc-detect-eval-cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir / "numba"))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime_cache()

    merged = load_predictions_and_labels(args.predictions, args.labels, args.threshold)
    merged, embeddings = load_embeddings(
        args.embeddings,
        merged,
        args.embedding_barcodes,
        args.embedding_key,
    )

    y_true = merged["true_label"].to_numpy(dtype=int)
    y_pred = merged["predicted_label"].to_numpy(dtype=int)
    y_score = merged["ctc_probability"].to_numpy(dtype=float)
    if len(np.unique(y_true)) < 2:
        raise ValueError("AUROC requires both positive and negative true labels.")

    rng = np.random.default_rng(args.seed)
    metrics = compute_basic_metrics(y_true, y_score, y_pred)
    ci_low, ci_high, bootstrap_valid = bootstrap_auroc_ci(
        y_true, y_score, args.bootstrap, rng
    )
    p_value, _ = permutation_auroc_pvalue(
        y_true, y_score, metrics["auroc"], args.permutations, rng
    )

    _, umap_indices = plot_embedding_umap(
        embeddings,
        y_true,
        y_pred,
        args.output_dir / "embedding_umap.png",
        args.seed,
        args.umap_max_points,
    )
    plot_confusion_matrix(
        y_true,
        y_pred,
        args.output_dir / "confusion_matrix.png",
    )

    group_confusion = collect_group_confusion(
        merged,
        args.group_col,
        args.output_dir,
        args.max_group_plots,
        args.min_group_size,
    )

    summary = build_summary(
        merged,
        metrics,
        ci_low,
        ci_high,
        args.bootstrap,
        bootstrap_valid,
        p_value,
        args.permutations,
        args.group_col,
        group_confusion,
        umap_indices,
    )

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"Wrote {args.output_dir / 'embedding_umap.png'}")
    print(f"Wrote {args.output_dir / 'confusion_matrix.png'}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
