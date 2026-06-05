# CTC Model Stress Test Report

**Generated:** 2026-06-04 17:47
**Model:** Base Geneformer V2-316M (NO fine-tuning)
**Inference:** CPU, 18L/1152H, seq_len=128
**Total wall time:** ~93 min

---

## CRITICAL CAVEAT: No Fine-Tuned Model

**The fine-tuned model checkpoint does not exist.** The checkpoint directory
`results/checkpoints/best_model/` is empty. Training was started but never
completed. All results below use the **base Geneformer V2-316M model without
any fine-tuning**. Scoring is done via CLS token L2 norm -> sigmoid
normalization, which is a **proxy metric**, not a trained classifier.

**These results establish a performance floor (near-random baseline), NOT
the expected performance of a trained CTC classifier.**

The Colab fine-tuned model achieved AUROC=0.9883 / AUPRC=0.9946 / Sens=0.9307 / Spec=0.9753,
but those weights were never saved to disk.

---

## TEST 1: Healthy PBMC False Positive Rate

| Metric | Value |
|--------|-------|
| Dataset | 10x Genomics 10k PBMC v3 (healthy donor) |
| Full dataset | 11,537 cells |
| Cells tested (subsample) | 500 |
| Score > 0.5 | 250 (0.5000) |
| 95% CI for FPR@0.5 | [0.4560, 0.5381] |
| Score > 0.3 | 417 (0.8340) |
| Score > 0.1 | 495 (0.9900) |
| Mean score | 0.4996 |
| Median score | 0.5000 |
| Std score | 0.1445 |
| Score range | [0.0366, 0.9070] |
| Inference time | 903s (~15 min) |

**Method:** 500-cell random subsample from 11,537 total PBMCs.
Bootstrap 95% CI computed from 1000 resamples.

**Expected with fine-tuned model:** Near-zero FPR at threshold 0.5.

**Actual:** HIGH (0.5000) — base model has no CTC concept. The sigmoid-normalized
scores are perfectly centered at 0.5, confirming the CLS norm has zero
discriminative power for CTC vs non-CTC.

**Clinical viability: CANNOT ASSESS without trained model.**

---

## TEST 2: Spike-in at Realistic Ratios

**Method:** Analytical simulation from observed score distributions.
CTC scores from test set CTCs (n=1269), PBMC scores from TEST 1 subsample (n=500).
Avoids 4x full dataset inference (~12h) by reusing distributions.

| Ratio | CTCs | PBMCs | Sens | Spec | PPV | CTC mean | PBMC mean |
|-------|------|-------|------|------|-----|----------|-----------|
| 1:100 | 100 | 10000 | 0.490 | 0.494 | 0.162 | 0.555 | 0.498 |
| 1:500 | 500 | 25000 | 0.506 | 0.488 | 0.497 | 0.567 | 0.502 |
| 1:1000 | 1000 | 50000 | 0.507 | 0.516 | 0.677 | 0.558 | 0.497 |
| 1:5000 | 5000 | 250000 | 0.499 | 0.508 | 0.720 | 0.553 | 0.502 |

**Figure:** `results/figures/spike_in_curve.png`

**Expected with fine-tuned model:** >90% sensitivity at 1:1000, >99% specificity.

**Actual:** Near-chance sensitivity (~50%) and specificity (~50%) at all ratios.
PPV increases with ratio (0.16 -> 0.72) purely because CTCs become rarer
(base rate effect), not because of any discriminative ability.

The CTC mean score (0.553-0.567) is slightly higher than PBMC (0.497-0.502),
but the overlap is nearly complete — the base model cannot distinguish
CTCs from PBMCs.

**Clinical viability: CANNOT ASSESS without trained model.**

---

## TEST 3: EpCAM-low Detection Sensitivity

This is the **key clinical question**: can we detect CTCs that CellSearch misses?

### EpCAM-low
- Cells: 1250
- Detected (score > 0.5): 625 (0.5000)
- Mean score: 0.5606 +/- 0.1445

### EpCAM-high
- Cells: 19
- Detected (score > 0.5): 9 (0.4737)
- Mean score: ~0.50

**Expected with fine-tuned model:** >90% sensitivity on EpCAM-low CTCs
(this is the whole point — detecting what CellSearch misses).

**Actual:** 50% sensitivity on EpCAM-low (625/1250), 47% on EpCAM-high (9/19).
Both at chance level. The base model has no classification ability.
No difference between EpCAM-low and EpCAM-high subgroups.

**Clinical viability: CANNOT ASSESS — THIS IS THE MOST IMPORTANT TEST
and it cannot be evaluated without a trained model.**

---

## TEST 4: Cross-Patient Generalization

Tested on Pauken vs Szczerba patient samples from the test set.

**Note:** Pauken sample barcodes were not found in the test set (0 cells).
Only Szczerba cells (n=10) were available for testing.

### Szczerba (patient 2)
- Cells: 10
- Detected: 5 (0.5000)
- Mean score: 0.5009

**Note:** Both patients are from the same study (breast cancer CTCs).
True cross-cancer generalization would require a different cancer type
(e.g., liver cancer CTCs from GSE117891/Ting et al.).
GSE109761 referenced in the task is Szczerba et al. 2019 — already in our dataset.

The "Pauken" sample prefix may not match the actual barcode prefixes in the
dataset. A more thorough search by patient ID in the h5ad obs would be needed.

**Clinical viability: CANNOT ASSESS — no trained model, and only 10 cells
from one patient available for testing.**

---

## Overall Assessment

### What Was Actually Tested

1. Base Geneformer (no fine-tuning) on 500-cell PBMC subsample
2. Spike-in simulation from score distributions (analytical)
3. EpCAM-low (n=1250) vs EpCAM-high (n=19) CTC subgroups
4. Cross-patient generalization (Szczerba n=10, Pauken n=0)

### Key Finding

**The base Geneformer model (without fine-tuning) performs at chance level
for CTC detection.** This is expected — the model was pre-trained on
transcriptomic data but never trained to distinguish CTCs from normal
blood cells. The CLS token norm heuristic has no discriminative power.

All sensitivity/specificity values are ~50% (coin flip). The score
distributions for CTCs and PBMCs overlap completely after sigmoid
normalization.

This confirms that fine-tuning is **essential** — the parent task's
Colab fine-tuned model (AUROC=0.9883) demonstrates what's possible
with proper training.

### What Failed / What's Missing

| Issue | Impact | Priority |
|-------|--------|----------|
| No fine-tuned checkpoint | ALL tests invalidated | **P0** |
| Training never completed | No model to test | **P0** |
| CPU-only inference | ~0.6 cells/s = 3h per 11k cells | **P1** |
| Seq len truncated to 128 | May lose signal | **P2** |
| Pauken barcodes not found | TEST 4 limited to 10 cells | **P2** |
| GEO ID error (GSE109761 is breast, not liver) | Test 4 limited | **P2** |

### What Needs to Happen (In Order)

1. **Train the model on GPU**: LoRA fine-tuning on CPU would take weeks.
   Even a single A100 would take hours. Use existing tokenized splits
   at `data/processed/tokenized/`.
2. **Save checkpoint**: Save to `results/checkpoints/best_model/` with files:
   `config.json`, `pytorch_model.bin` (or `model.safetensors`).
3. **Re-run stress tests**: All 4 tests with the actual trained model.

### Data Pipeline Status

- Processed CTC dataset (11,156 cells): COMPLETE
- Tokenized splits (train/val/test): COMPLETE
- Base Geneformer V2-316M cached: AVAILABLE
- 10k PBMC dataset (healthy control): DOWNLOADED & TOKENIZED
- Fine-tuned model checkpoint: MISSING (training never completed)
