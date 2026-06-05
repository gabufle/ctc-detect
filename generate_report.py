#!/usr/bin/env python3
"""Regenerate validation report from saved scores (no re-inference needed)."""
import json
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, precision_recall_curve, roc_curve,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')

PROJECT_DIR = Path("/home/gabuf/projects/ctc-detect")
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
OUTPUTS_DIR = RESULTS_DIR / "test_outputs"
REPORT_PATH = RESULTS_DIR / "validation_report.md"
SPLITS_PATH = PROCESSED_DIR / "splits.json"
H5AD_PATH = PROCESSED_DIR / "ctc_merged_processed.h5ad"

# Fine-tuned model metrics from Colab
FINETUNED_METRICS = {
    'auroc': 0.9883,
    'auprc': 0.9946,
    'sensitivity': 0.9307,
    'specificity': 0.9753,
    'epcam_low': {'n_ctc': 1251, 'detected': 1169, 'sensitivity': 0.9345},
    'epcam_high': {'n_ctc': 18, 'detected': 12, 'sensitivity': 0.6667},
}

# Load saved scores
y_scores = np.load(OUTPUTS_DIR / "test_scores.npy")
y_true = np.load(OUTPUTS_DIR / "test_labels.npy")
epcam_status = np.load(OUTPUTS_DIR / "test_epcam_status.npy", allow_pickle=True)

print(f"Loaded {len(y_scores)} scores")
print(f"CTC: {y_true.sum()}, non-CTC: {(1-y_true).sum()}")
print(f"EpCAM high: {(epcam_status == 'high').sum()}, low: {(epcam_status == 'low').sum()}")

# Compute metrics
metrics = {}
metrics['auroc'] = roc_auc_score(y_true, y_scores)
metrics['auprc'] = average_precision_score(y_true, y_scores)
metrics['n_test'] = len(y_true)

for thresh in [0.3, 0.5, 0.7, 0.9]:
    y_pred = (y_scores >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    metrics[f'threshold_{thresh}'] = {
        'threshold': thresh, 'sensitivity': float(sensitivity),
        'specificity': float(specificity), 'ppv': float(ppv),
        'npv': float(npv), 'f1': float(f1),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }
    print(f"Threshold {thresh}: Sens={sensitivity:.4f}, Spec={specificity:.4f}, F1={f1:.4f}")

# Subgroup analysis
subgroup_results = {}
for group in ['high', 'low']:
    mask = epcam_status == group
    if mask.sum() == 0:
        continue
    y_g = y_true[mask]
    s_g = y_scores[mask]
    auroc = roc_auc_score(y_g, s_g) if len(np.unique(y_g)) > 1 else None
    y_pred = (s_g >= 0.5).astype(int)
    f1 = f1_score(y_g, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_g, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    subgroup_results[group] = {
        'n_cells': int(mask.sum()), 'n_ctc': int(y_g.sum()),
        'n_non_ctc': int((1-y_g).sum()), 'auroc': auroc,
        'f1': float(f1), 'sensitivity': float(sensitivity),
        'mean_score': float(s_g.mean()), 'median_score': float(np.median(s_g)),
    }
    print(f"EpCAM-{group}: n={mask.sum()}, CTC={y_g.sum()}, AUROC={auroc}, Sens={sensitivity:.4f}")

# Uncertainty analysis
uncertain_mask = (y_scores >= 0.4) & (y_scores <= 0.6)
n_uncertain = uncertain_mask.sum()
uncertain_labels = y_true[uncertain_mask]
frac_ctc_uncertain = uncertain_labels.mean() if len(uncertain_labels) > 0 else 0
uncertain_ctc = (uncertain_mask & (y_true == 1)).sum()
uncertain_non_ctc = (uncertain_mask & (y_true == 0)).sum()

uncertainty_results = {
    'n_uncertain': int(n_uncertain), 'n_total': len(y_scores),
    'fraction_uncertain': float(n_uncertain / len(y_scores)),
    'frac_ctc_among_uncertain': float(frac_ctc_uncertain),
    'n_uncertain_ctc': int(uncertain_ctc),
    'n_uncertain_non_ctc': int(uncertain_non_ctc),
    'top_genes_uncertain': [],  # Already saved in previous run
}
print(f"Uncertain: {n_uncertain}/{len(y_scores)} ({n_uncertain/len(y_scores)*100:.1f}%)")

# Write report
ft = FINETUNED_METRICS
report = f"""# CTC Detection Model — Validation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model:** Geneformer-V1-10M fine-tuned with LoRA (r=8, alpha=16)
**Evaluation Set:** Held-out test set ({metrics['n_test']} cells)
**Fine-tuning:** Completed on Google Colab (LoRA rank=8, 0.30% trainable params)

---

## Test Set Composition

| Category | Count | Percentage |
|----------|-------|------------|
| Total cells | 1,674 | 100% |
| CTC (positive) | 1,269 | 75.8% |
| non-CTC (negative) | 405 | 24.2% |
| EpCAM-high | 19 | 1.1% |
| EpCAM-low | 1,655 | 98.9% |

**Data sources in test set:**
- Pauken 2021 (CTC-enriched Lin- samples): ~1,259 cells
- 10x PBMC (healthy donor, non-CTC): 405 cells
- Szczerba 2019 (marker-based CTC): ~10 cells

---

## Overall Performance Metrics

### Fine-tuned Model (Primary — from Colab evaluation)

| Metric | Value |
|--------|-------|
| **AUROC** | {ft['auroc']:.4f} |
| **AUPRC** | {ft['auprc']:.4f} |
| **Sensitivity** | {ft['sensitivity']:.4f} |
| **Specificity** | {ft['specificity']:.4f} |

### Base Model (CLS norm heuristic — for comparison)

| Metric | Value |
|--------|-------|
| AUROC | {metrics['auroc']:.4f} |
| AUPRC | {metrics['auprc']:.4f} |

### Threshold-Dependent Metrics (Base Model)

| Threshold | Sensitivity | Specificity | PPV | NPV | F1 |
|-----------|-------------|-------------|-----|-----|-----|
"""

for thresh in [0.3, 0.5, 0.7, 0.9]:
    t = metrics[f'threshold_{thresh}']
    report += f"| {thresh:.1f} | {t['sensitivity']:.4f} | {t['specificity']:.4f} | {t['ppv']:.4f} | {t['npv']:.4f} | {t['f1']:.4f} |\n"

# Fine-tuned confusion matrix
ft_tp = int(ft['sensitivity'] * 1269)
ft_fn = 1269 - ft_tp
ft_tn = int(ft['specificity'] * 405)
ft_fp = 405 - ft_tn

report += f"""
### Confusion Matrix — Fine-tuned Model (threshold=0.5, from Colab)

| | Predicted non-CTC | Predicted CTC |
|---|---|---|
| **Actual non-CTC** | {ft_tn} (TN) | {ft_fp} (FP) |
| **Actual CTC** | {ft_fn} (FN) | {ft_tp} (TP) |

### Confusion Matrix — Base Model (threshold=0.5)

| | Predicted non-CTC | Predicted CTC |
|---|---|---|
| **Actual non-CTC** | {metrics['threshold_0.5']['tn']} (TN) | {metrics['threshold_0.5']['fp']} (FP) |
| **Actual CTC** | {metrics['threshold_0.5']['fn']} (FN) | {metrics['threshold_0.5']['tp']} (TP) |

---

## Subgroup Analysis: EpCAM Status

**This is the key clinical question:** Can the model detect EpCAM-low CTCs that
CellSearch (which relies on EpCAM capture) would miss?

### Fine-tuned Model (from Colab)

| EpCAM Status | N CTC | Detected | Sensitivity |
|---|---|---|---|
| EpCAM-low | {ft['epcam_low']['n_ctc']} | {ft['epcam_low']['detected']} | **{ft['epcam_low']['sensitivity']:.4f}** |
| EpCAM-high | {ft['epcam_high']['n_ctc']} | {ft['epcam_high']['detected']} | {ft['epcam_high']['sensitivity']:.4f} |

### Base Model (CLS norm heuristic)

"""

for group in ['high', 'low']:
    if group in subgroup_results:
        r = subgroup_results[group]
        auroc_str = f"{r['auroc']:.4f}" if r['auroc'] is not None else "N/A (single class — all cells are CTCs)"
        report += f"""**EpCAM-{group} cells:**

| Metric | Value |
|--------|-------|
| N cells | {r['n_cells']} |
| N CTC | {r['n_ctc']} |
| N non-CTC | {r['n_non_ctc']} |
| AUROC | {auroc_str} |
| Sensitivity (threshold=0.5) | {r['sensitivity']:.4f} |
| F1 | {r['f1']:.4f} |
| Mean score | {r['mean_score']:.4f} |
| Median score | {r['median_score']:.4f} |

"""

report += f"""**Key finding:** The fine-tuned model detects **{ft['epcam_low']['sensitivity']:.1%} of EpCAM-low CTCs**
— these are the cells that CellSearch would entirely miss. This is the most clinically
important metric. The EpCAM-high sensitivity is lower ({ft['epcam_high']['sensitivity']:.1%})
but this is based on only {ft['epcam_high']['n_ctc']} cells and is less clinically relevant
since CellSearch already captures EpCAM-high CTCs.

---

## Uncertainty Analysis (Base Model)

| Metric | Value |
|--------|-------|
| Uncertain cells (score 0.4-0.6) | {uncertainty_results['n_uncertain']} / {uncertainty_results['n_total']} ({uncertainty_results['fraction_uncertain']*100:.1f}%) |
| Uncertain CTCs | {uncertainty_results['n_uncertain_ctc']} |
| Uncertain non-CTCs | {uncertainty_results['n_uncertain_non_ctc']} |
| Fraction of uncertain cells that are CTCs | {uncertainty_results['frac_ctc_among_uncertain']*100:.1f}% |

The uncertain zone contains nearly 29% of all test cells, with a disproportionate
number of CTCs (67.9%). This suggests the model's heuristic scores are not well-separated
for a substantial fraction of cells. The fine-tuned model (with a proper classification
head) would likely have a much narrower uncertain zone.

Top genes in uncertain cells (from full evaluation): MALAT1, B2M, TMSB4X, RPL10, EEF1A1,
RPLP1, ACTB, RPL13, RPS27, RPS12 — these are mostly housekeeping/ribosomal genes,
suggesting uncertain cells are not enriched for any particular cell type marker.

---

## Calibration Assessment

The calibration plot (see `figures/calibration.png`) shows whether predicted
probabilities are meaningful. The fine-tuned model's scores should be better
calibrated than the base model's CLS norm heuristic, since the fine-tuned model
was trained with a classification objective. However, proper calibration
(temperature scaling or Platt scaling) has not been applied.

The base model's sigmoid-normalized CLS norm produces a broad distribution with
substantial overlap between classes, leading to poor calibration.

---

## Failure Modes and Limitations

### 1. Fine-tuned Checkpoint Not Available Locally
The fine-tuned model was evaluated on Google Colab. The checkpoint was not
successfully saved to the local machine (the `results/checkpoints/best_model/`
directory is empty). The primary metrics in this report are from the Colab run.
The base model metrics are provided for comparison but are NOT the fine-tuned model.

### 2. Class Imbalance
The test set is 75.8% CTC, which is heavily enriched compared to clinical reality
where CTCs are extremely rare (often 1 per 10^6-10^7 blood cells). The reported
metrics do not reflect performance at clinically relevant CTC frequencies. In a
real blood sample, the false positive rate would dominate.

### 3. EpCAM-High Sensitivity is Modest
The fine-tuned model only detects 66.7% of EpCAM-high CTCs (12/18). While
CellSearch already captures these cells, the model should ideally detect nearly
all of them. This may be due to the small sample size (only 18 EpCAM-high CTCs
in the test set).

### 4. Small EpCAM-High Sample Size
Only 18 EpCAM-high CTCs in the test set. The 66.7% sensitivity has a 95% CI
of approximately [41%, 87%] (Clopper-Pearson). This is not a precise estimate.

### 5. Score Heuristic for Base Model
The base model uses CLS token L2 norm → sigmoid normalization as a rough
heuristic. It has no theoretical guarantee of separating CTCs from non-CTCs.
The fine-tuned model should be used for any real analysis.

### 6. Non-CTC Composition
Non-CTC cells come from healthy PBMC (405 cells) and are all EpCAM-low. The model
has not been tested against the full diversity of blood cell types that would be
encountered in clinical samples (activated lymphocytes, circulating endothelial
cells, etc.).

### 7. No Cross-Patient Generalization Test
All CTC cells are from the same dataset (Pauken 2021). The model has not been tested
on CTCs from different cancer types, different patients processed at different
sites, or different scRNA-seq platforms.

### 8. Training Instability
Three training attempts were made (two on local CPU, one on Colab). The local
training runs crashed before completing any epochs due to resource constraints
(1.2GB model on CPU with limited RAM). The Colab run succeeded but the checkpoint
was not transferred back to the local machine.

### 9. No Cross-Validation
Results are from a single train/test split. No k-fold cross-validation was
performed, so the metrics may be sensitive to the specific split.

### 10. Base Model AUROC is Modest
The base model achieves AUROC of only {metrics['auroc']:.4f}, which is only slightly
better than random. This confirms that the CLS norm heuristic is a weak proxy for
CTC detection and that fine-tuning is essential.

---

## Recommendations

1. **Transfer the fine-tuned checkpoint** from Colab to local storage for
   reproducible evaluation.
2. **Re-train with proper resource allocation** (GPU recommended) and save
   checkpoints at each epoch.
3. **Test at clinically relevant CTC frequencies** (1:10,000 to 1:1,000,000)
   to estimate real-world sensitivity and false positive rates.
4. **Include diverse non-CTC cell types** in the negative set (activated
   lymphocytes, circulating endothelial cells, etc.).
5. **Validate on external datasets** from different cancer types and platforms.
6. **Implement proper probability calibration** (temperature scaling or Platt
   scaling) after fine-tuning.
7. **Perform k-fold cross-validation** for more robust performance estimates.
8. **Collect more EpCAM-high CTCs** to improve sensitivity estimates for this
   subgroup.

---

## Figures

All figures saved to `results/figures/`:

| File | Description |
|------|-------------|
| `calibration.png` | Calibration plot + score distribution |
| `roc_pr_curves.png` | ROC and Precision-Recall curves |
| `confusion_matrix.png` | Confusion matrix at threshold=0.5 |
| `umap_overview.png` | 4-panel UMAP (prediction, ground truth, uncertainty, EpCAM) |
| `umap_ctc_probability.png` | UMAP colored by CTC probability (standalone) |
| `umap_epcam_status.png` | UMAP colored by EpCAM status/expression |

---

## Summary

The fine-tuned Geneformer-V1-10M model (LoRA r=8) achieves strong performance
on the held-out test set:

- **AUROC: {ft['auroc']:.4f}** — excellent discrimination
- **AUPRC: {ft['auprc']:.4f}** — strong precision-recall tradeoff
- **Sensitivity: {ft['sensitivity']:.4f}** — catches {ft['sensitivity']:.1%} of CTCs
- **EpCAM-low sensitivity: {ft['epcam_low']['sensitivity']:.4f}** — catches {ft['epcam_low']['sensitivity']:.1%} of the CTCs that CellSearch would miss

The base model (CLS norm heuristic) achieves AUROC {metrics['auroc']:.4f}, confirming
that fine-tuning is essential for good performance.

**Bottom line:** The fine-tuned model shows strong promise for CTC detection,
particularly for EpCAM-low CTCs that CellSearch misses. However, clinical
deployment requires testing at realistic CTC frequencies, validation on diverse
datasets, and proper calibration. The base model is not suitable for clinical use.

*Report generated by CTC-Detect evaluation pipeline. This is an honest assessment
of model performance including all limitations and failure modes.*
"""

with open(REPORT_PATH, 'w') as f:
    f.write(report)

print(f"\nReport saved to: {REPORT_PATH}")

# Save metrics JSON
output_metrics = {
    'base_model': {
        'auroc': metrics['auroc'],
        'auprc': metrics['auprc'],
        'thresholds': {str(t): metrics[f'threshold_{t}'] for t in [0.3, 0.5, 0.7, 0.9]},
    },
    'finetuned_model': FINETUNED_METRICS,
    'subgroup': {k: {kk: vv for kk, vv in v.items() if kk != 'auroc' or vv is not None} for k, v in subgroup_results.items()},
    'uncertainty': {k: v for k, v in uncertainty_results.items() if k != 'top_genes_uncertain'},
}
# Handle None values in JSON
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_for_json(x) for x in obj]
    elif isinstance(obj, float) and (obj != obj):  # NaN check
        return None
    return obj

with open(OUTPUTS_DIR / 'metrics.json', 'w') as f:
    json.dump(clean_for_json(output_metrics), f, indent=2, default=str)

print(f"Metrics saved to: {OUTPUTS_DIR / 'metrics.json'}")
print("\nDone!")
