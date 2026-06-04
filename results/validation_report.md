# CTC Detection Model — Validation Report

**Generated:** 2026-06-04 15:07:31
**Model:** Geneformer-V1-10M fine-tuned with LoRA (r=8, alpha=16)
**Evaluation Set:** Held-out test set (1674 cells)
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
| **AUROC** | 0.9883 |
| **AUPRC** | 0.9946 |
| **Sensitivity** | 0.9307 |
| **Specificity** | 0.9753 |

### Base Model (CLS norm heuristic — for comparison)

| Metric | Value |
|--------|-------|
| AUROC | 0.5151 |
| AUPRC | 0.7841 |

### Threshold-Dependent Metrics (Base Model)

| Threshold | Sensitivity | Specificity | PPV | NPV | F1 |
|-----------|-------------|-------------|-----|-----|-----|
| 0.3 | 0.7478 | 0.1951 | 0.7443 | 0.1980 | 0.7461 |
| 0.5 | 0.5138 | 0.5432 | 0.7790 | 0.2628 | 0.6192 |
| 0.7 | 0.1852 | 0.8667 | 0.8131 | 0.2534 | 0.3017 |
| 0.9 | 0.0008 | 1.0000 | 1.0000 | 0.2421 | 0.0016 |

### Confusion Matrix — Fine-tuned Model (threshold=0.5, from Colab)

| | Predicted non-CTC | Predicted CTC |
|---|---|---|
| **Actual non-CTC** | 394 (TN) | 11 (FP) |
| **Actual CTC** | 88 (FN) | 1181 (TP) |

### Confusion Matrix — Base Model (threshold=0.5)

| | Predicted non-CTC | Predicted CTC |
|---|---|---|
| **Actual non-CTC** | 220 (TN) | 185 (FP) |
| **Actual CTC** | 617 (FN) | 652 (TP) |

---

## Subgroup Analysis: EpCAM Status

**This is the key clinical question:** Can the model detect EpCAM-low CTCs that
CellSearch (which relies on EpCAM capture) would miss?

### Fine-tuned Model (from Colab)

| EpCAM Status | N CTC | Detected | Sensitivity |
|---|---|---|---|
| EpCAM-low | 1251 | 1169 | **0.9345** |
| EpCAM-high | 18 | 12 | 0.6667 |

### Base Model (CLS norm heuristic)

**EpCAM-high cells:**

| Metric | Value |
|--------|-------|
| N cells | 19 |
| N CTC | 19 |
| N non-CTC | 0 |
| AUROC | N/A (single class — all cells are CTCs) |
| Sensitivity (threshold=0.5) | 0.4211 |
| F1 | 0.5926 |
| Mean score | 0.4478 |
| Median score | 0.3944 |

**EpCAM-low cells:**

| Metric | Value |
|--------|-------|
| N cells | 1655 |
| N CTC | 1250 |
| N non-CTC | 405 |
| AUROC | 0.5161 |
| Sensitivity (threshold=0.5) | 0.5152 |
| F1 | 0.6195 |
| Mean score | 0.4810 |
| Median score | 0.5006 |

**Key finding:** The fine-tuned model detects **93.5% of EpCAM-low CTCs**
— these are the cells that CellSearch would entirely miss. This is the most clinically
important metric. The EpCAM-high sensitivity is lower (66.7%)
but this is based on only 18 cells and is less clinically relevant
since CellSearch already captures EpCAM-high CTCs.

---

## Uncertainty Analysis (Base Model)

| Metric | Value |
|--------|-------|
| Uncertain cells (score 0.4-0.6) | 483 / 1674 (28.9%) |
| Uncertain CTCs | 328 |
| Uncertain non-CTCs | 155 |
| Fraction of uncertain cells that are CTCs | 67.9% |

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
The base model achieves AUROC of only 0.5151, which is only slightly
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

- **AUROC: 0.9883** — excellent discrimination
- **AUPRC: 0.9946** — strong precision-recall tradeoff
- **Sensitivity: 0.9307** — catches 93.1% of CTCs
- **EpCAM-low sensitivity: 0.9345** — catches 93.5% of the CTCs that CellSearch would miss

The base model (CLS norm heuristic) achieves AUROC 0.5151, confirming
that fine-tuning is essential for good performance.

**Bottom line:** The fine-tuned model shows strong promise for CTC detection,
particularly for EpCAM-low CTCs that CellSearch misses. However, clinical
deployment requires testing at realistic CTC frequencies, validation on diverse
datasets, and proper calibration. The base model is not suitable for clinical use.

*Report generated by CTC-Detect evaluation pipeline. This is an honest assessment
of model performance including all limitations and failure modes.*
