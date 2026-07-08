"""Bootstrap AUROC confidence interval + permutation test p-value.

Runs on top of the same predictions + ground-truth files you'd pass to
`ctc-detect evaluate`. Prints n, AUROC, 95% CI, and permutation p-value,
and writes summary.json to the output dir.

Usage:
  python scripts/stats_test.py \
      --predictions results/ge_et_al/ctc_probabilities.csv \
      --ground-truth data/external/ge_et_al/ground_truth.csv \
      --output results/ge_et_al \
      --n-bootstrap 1000 --n-permutations 1000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def bootstrap_auroc_ci(y_true, y_scores, n_bootstrap=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_scores[idx]
        if len(np.unique(yt)) < 2:
            continue  # skip resamples with only one class present
        scores.append(roc_auc_score(yt, ys))
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    lower, upper = np.quantile(scores, [alpha, 1 - alpha])
    return lower, upper, scores


def permutation_test(y_true, y_scores, n_permutations=1000, seed=42):
    rng = np.random.default_rng(seed)
    observed = roc_auc_score(y_true, y_scores)
    null_scores = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(y_true)
        null_scores[i] = roc_auc_score(shuffled, y_scores)
    p_value = (np.sum(null_scores >= observed) + 1) / (n_permutations + 1)
    return observed, p_value, null_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-permutations", type=int, default=1000)
    args = parser.parse_args()

    pred_df = pd.read_csv(args.predictions)
    gt_df = pd.read_csv(args.ground_truth)
    merged = pred_df.merge(gt_df[["barcode", "true_label"]], on="barcode", how="inner")

    if len(merged) == 0:
        raise SystemExit("No barcodes matched between predictions and ground truth.")

    y_true = merged["true_label"].values.astype(int)
    y_scores = merged["ctc_probability"].values
    n_total = len(y_true)
    n_positive = int(y_true.sum())
    n_negative = n_total - n_positive

    print(f"n_total={n_total}  n_positive={n_positive}  n_negative={n_negative}")

    if n_positive == 0 or n_negative == 0:
        # Pure negative-control dataset (e.g. healthy PBMCs) — AUROC undefined.
        # Report false-positive rate instead, which is what actually matters here.
        fpr_at_threshold = float((y_scores >= 0.5).mean())
        print(f"Single-class dataset (all label={y_true[0]}) — AUROC undefined.")
        print(f"False positive rate at threshold=0.5: {fpr_at_threshold:.4f} ({int((y_scores >= 0.5).sum())}/{n_total})")
        summary = {
            "n_total": n_total,
            "single_class_label": int(y_true[0]),
            "false_positive_rate_at_0.5": fpr_at_threshold,
            "mean_score": float(y_scores.mean()),
            "median_score": float(np.median(y_scores)),
        }
    else:
        auroc, p_value, null_scores = permutation_test(y_true, y_scores, args.n_permutations)
        lower, upper, boot_scores = bootstrap_auroc_ci(y_true, y_scores, args.n_bootstrap)

        print(f"AUROC: {auroc:.4f}  95% CI: [{lower:.4f}, {upper:.4f}]  (n={n_total} boot={len(boot_scores)})")
        print(f"Permutation test p-value: {p_value:.4g}  (n_permutations={args.n_permutations})")

        summary = {
            "n_total": n_total,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "auroc": float(auroc),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "n_bootstrap_valid": int(len(boot_scores)),
            "p_value": float(p_value),
            "n_permutations": args.n_permutations,
        }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stats_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_dir / 'stats_summary.json'}")


if __name__ == "__main__":
    main()
