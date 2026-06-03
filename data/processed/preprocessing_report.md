# CTC scRNA-seq Preprocessing Report
**Generated:** 2026-06-02T16:59:06.958225
**Pipeline:** scanpy 1.11.5

---

## Datasets Processed

### Dataset 1: Pauken et al. 2021 (Breast Cancer CTCs)
- **GEO:** GSE174463 / GSE174461
- **Format:** 10x Genomics Chromium MEX (Cell Ranger 3.1.0)
- **Samples:** 6 (3 patients x Lin-/Lin+)
- **Cells (raw):** 9,044
- **Genes (raw):** 33,538
- **Genome:** GRCh38

### Dataset 2: Szczerba et al. 2019 (Breast Cancer CTCs)
- **GEO:** GSE109761
- **Format:** Processed normalized matrix (genes x cells)
- **Samples:** Human cells from breast cancer patients
- **Cells (raw):** Variable (processed matrix)
- **Genes (raw):** 33,214

### Dataset 3: 10x Genomics PBMC 3k (Healthy Donor)
- **Source:** 10x Genomics (Cell Ranger 1.1.0)
- **Format:** 10x MEX
- **Cells (raw):** 2,700
- **Genes (raw):** 32,738
- **Genome:** hg19/GRCh37

---

## Quality Control

### Thresholds Applied
| Parameter | Value |
|-----------|-------|
| Minimum genes per cell | 200 |
| Maximum genes per cell | 6000 |
| Maximum mitochondrial % | 20% |
| Minimum cells per gene | 3 |

### Cell Counts After QC
- **pauken_2021:** 8385 cells (8385 CTC, 0 normal)
- **szczerba_2019:** 73 cells (73 CTC, 0 normal)
- **pbmc_10x:** 2698 cells (0 CTC, 2698 normal)

---

## Normalization
- Method: `scanpy.pp.normalize_total(target_sum=10000)`
- Transform: `log1p` (natural log)
- HVG selection: NOT performed (Geneformer uses its own gene ranking)

---

## CTC Identification

### Pauken 2021
- CTCs identified by marker-based enrichment: Lin- (CD45-depleted) fraction is enriched for CTCs.
- The original publication identified cluster 10 (integrated UMAP, ~5801 high-quality cells) as CTC-candidate based on epithelial/mammary markers.
- All cells in Lin- samples are marked is_ctc=True (CTC-enriched). Lin+ cells in this dataset are marked as is_ctc=True as they originate from the CTC study but contain mostly WBCs.

> **NOTE:** In the Pauken paper, not all Lin- cells are CTCs. The paper reports ~201 cells in the CTC-candidate cluster. However, for training purposes, we keep the full Lin- fraction as CTC-enriched and let the model learn distinguishing features.

### Szczerba 2019
- All cells are from CTC/CTC-WBC cluster samples from breast cancer patients.
- CTC identity is marker-based (experimental isolation of CTC clusters).

### PBMC 3k
- No CTCs expected (healthy donor PBMCs).
- All cells marked is_ctc=False.

---

## EpCAM Status
- **pauken_2021:** 107 EpCAM-high, 8278 EpCAM-low, 0 unknown
- **szczerba_2019:** 4 EpCAM-high, 69 EpCAM-low, 0 unknown
- **pbmc_10x:** 0 EpCAM-high, 2698 EpCAM-low, 0 unknown

---

## Merged Dataset
| Metric | Value |
|--------|-------|
| Total cells | 11156 |
| Total genes | 7637 |
| CTC cells | 8458 (75.8%) |
| Normal cells | 2698 (24.2%) |

---

## Train/Val/Test Splits
- Split ratio: 70 / 15 / 15
- Stratified by is_ctc label
- Random state: 42

---

## Output Files
- `data/processed/ctc_merged_processed.h5ad` — merged, processed AnnData
- `data/processed/splits.json` — train/val/test indices
- `results/figures/initial_umap.png` — UMAP visualization
- This report: `data/processed/preprocessing_report.md`