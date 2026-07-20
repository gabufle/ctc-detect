"""Run detection + evaluation + stats directly, no CLI build needed.

Calls the same functions the `ctc-detect` CLI calls under the hood, so you
don't need to `pip install -e .` or have the CLI entry point set up —
just run this from the repo root (or with `src/` on PYTHONPATH) with the
same deps the repo already needs (transformers, peft, scanpy, etc.).

Usage:
  python scripts/run_and_eval.py \\
      --input data/external/ge_et_al/data.h5ad \\
      --ground-truth data/external/ge_et_al/ground_truth.csv \\
      --output results/ge_et_al
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure src/ is importable even without pip install -e .
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def bootstrap_auroc_ci(y_true, y_scores, n_bootstrap=1000, ci=0.95, seed=42):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_scores[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(roc_auc_score(yt, ys))
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    lower, upper = np.quantile(scores, [alpha, 1 - alpha])
    return lower, upper, scores


def permutation_test(y_true, y_scores, n_permutations=1000, seed=42):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    observed = roc_auc_score(y_true, y_scores)
    null_scores = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(y_true)
        null_scores[i] = roc_auc_score(shuffled, y_scores)
    p_value = (np.sum(null_scores >= observed) + 1) / (n_permutations + 1)
    return observed, p_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="h5ad or Cell Ranger dir")
    parser.add_argument("--ground-truth", required=True, help="CSV with barcode,true_label")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skip-umap", action="store_true")
    args = parser.parse_args()

    from ctcdetect.detect import run_detection
    from ctcdetect.evaluate import compute_metrics, generate_eval_report, plot_roc_pr

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Step 1: run detection directly (same function the CLI calls) ---
    print(f"Running detection on {input_path} ...")
    run_detection(
        input_path=input_path,
        output_path=output_path,
        threshold=args.threshold,
        skip_umap=args.skip_umap,
    )

    # --- Step 2: load predictions + ground truth, compute metrics ---
    pred_df = pd.read_csv(output_path / "ctc_probabilities.csv")
    gt_df = pd.read_csv(args.ground_truth)
    merged = pred_df.merge(gt_df[["barcode", "true_label"]], on="barcode", how="inner")

    if len(merged) == 0:
        raise SystemExit("No barcodes matched between predictions and ground truth.")

    y_true = merged["true_label"].values.astype(int)
    y_scores = merged["ctc_probability"].values
    n_total, n_positive = len(y_true), int(y_true.sum())
    print(f"\nMatched {n_total} cells. n_positive={n_positive} n_negative={n_total - n_positive}")

    n_negative = n_total - n_positive
    if n_positive == 0 or n_negative == 0:
        fpr = float((y_scores >= args.threshold).mean())
        print(f"Single-class dataset — AUROC undefined. False positive rate: {fpr:.4f}")
        summary = {"n_total": n_total, "single_class_label": int(y_true[0]),
                   "false_positive_rate": fpr, "mean_score": float(y_scores.mean())}
    else:
        metrics = compute_metrics(y_true, y_scores, args.threshold)
        generate_eval_report(metrics, output_path)
        plot_roc_pr(metrics, output_path)

        auroc, p_value = permutation_test(y_true, y_scores)
        lower, upper, _ = bootstrap_auroc_ci(y_true, y_scores)

        print(f"AUROC: {auroc:.4f}  95% CI: [{lower:.4f}, {upper:.4f}]")
        print(f"Permutation p-value: {p_value:.4g}")

        summary = {
            "n_total": n_total, "n_positive": n_positive, "n_negative": n_total - n_positive,
            "auroc": float(auroc), "ci_lower": float(lower), "ci_upper": float(upper),
            "p_value": float(p_value),
        }

    with open(output_path / "stats_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll results in {output_path}/ (ctc_probabilities.csv, eval_report.txt, roc.png, pr.png, umap.png, stats_summary.json)")


if __name__ == "__main__":
    main()
