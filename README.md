# ctc-detect

**Geneformer-based Circulating Tumor Cell (CTC) detection from single-cell RNA-seq data.**

## Why This Matters

Circulating tumor cells (CTCs) are cancer cells that have detached from a primary tumor and entered the bloodstream. Detecting them in a blood draw ("liquid biopsy") can reveal whether a cancer is spreading, how it is responding to treatment, and what its molecular profile looks like -- all without needing an invasive tissue biopsy.

The current FDA-approved standard for CTC detection is **CellSearch**, which works by fishing out cells that carry a surface protein called **EpCAM** (epithelial cell adhesion molecule). The problem: many CTCs -- especially those that have undergone epithelial-to-mesenchymal transition (EMT), a process linked to metastasis -- express little or no EpCAM. These **EpCAM-low CTCs** are invisible to CellSearch, yet they may be the most dangerous and clinically relevant subset.

**ctc-detect** addresses this gap. It uses a transformer model trained on gene expression profiles (scRNA-seq) to identify CTCs regardless of whether they express EpCAM. On held-out test data, the model detects **93.5% of EpCAM-low CTCs** --- the exact cells that CellSearch would miss.

## What the Project Does

The pipeline:

1. **Preprocess** raw scRNA-seq data (quality control normalization, gene mapping)
2. **Tokenize** gene expression profiles into ranked gene lists that Geneformer can read
3. **Fine-tune** Geneformer with LoRA (Low-Rank Adaptation) on CTC vs. non-CTC classification
4. **Evaluate** the model on a held-out test set with full metrics and visualizations

Key results on the held-out test set (1,674 cells):

| Metric | Fine-tuned Model | Base Model (no fine-tuning) |
|--------|-----------------|---------------------------|
| AUROC | **0.9883** | 0.5151 |
| AUPRC | **0.9946** | 0.7841 |
| Sensitivity | **0.9307** | 0.5138* |
| Specificity | **0.9753** | 0.5432* |
| EpCAM-low sensitivity | **0.9345** | 0.5152* |

\*Base model metrics use a simple CLS-token norm heuristic with threshold=0.5 and are shown only for comparison. The base model performs barely above random, confirming that fine-tuning is essential.

See `results/validation_report.md` for the full validation report with subgroup analysis, confusion matrices, and honest assessment of limitations.

## Installation

### Requirements

- Python >= 3.10
- ~8 GB RAM (for model loading on CPU; GPU recommended for training)
- ~2 GB disk space for the model checkpoint

### Setup

```bash
git clone <repo-url>
cd ctc-detect
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the package with dependencies: `typer` (CLI), `rich` (terminal output).

For training and evaluation, additional packages are needed:

```bash
pip install torch transformers peft datasets scanpy anndata scikit-learn matplotlib
```

The Geneformer model itself (~1.2 GB) is downloaded automatically from HuggingFace on first use and cached locally.

## Pipeline Steps

### 1. Data Preparation

Place raw scRNA-seq data (10x Genomics Cell Ranger output) in `data/raw/`. See `data/raw/data_provenance.md` for the exact datasets used and download links.

### 2. Preprocessing

Process raw data through the scanpy pipeline:

- Quality control filtering (gene count thresholds, mitochondrial fraction)
- Normalization (counts per 10,000 + log1p transform)
- Gene mapping to Ensembl IDs (required for Geneformer tokenization)
- Merging of CTC and non-CTC datasets

Output: `data/processed/ctc_merged_processed.h5ad` and `data/processed/splits.json`.

### 3. Tokenization

Convert expression matrices into ranked gene ID sequences:

```bash
python tokenize_data.py
```

Each cell becomes an ordered list of gene IDs, ranked from highest to lowest expression. This is the format Geneformer expects. Output goes to `data/processed/tokenized/` as HuggingFace Datasets.

### 4. Training

Fine-tune Geneformer with LoRA:

```bash
python train_geneformer.py
```

Training uses a 70/15/15 train/val/test split (stratified by CTC label). Class weights are applied to handle the CTC-heavy dataset composition. Training checkpoints are saved to `results/checkpoints/best_model/`.

### 5. Evaluation

Generate metrics and figures:

```bash
python evaluate_model.py
python generate_report.py
```

Produces:
- `results/test_outputs/metrics.json` -- all numeric results
- `results/validation_report.md` -- full written validation report
- `results/figures/` -- ROC curves, confusion matrix, UMAP visualizations, calibration plot

## How to Run on a New Dataset

1. Place your 10x Genomics data in `data/raw/` with the expected directory structure
2. Run preprocessing to produce a processed `.h5ad` file
3. Run tokenization to produce tokenized dataset splits
4. Run training and evaluation
5. Update the paths in `src/ctcdetect/config.py` if needed

The key input requirement is a cell x gene count matrix with UMI counts. The pipeline handles normalization, tokenization, and label assignment. For CTC detection, you need both positive (CTC-enriched) and negative (healthy PBMC or similar) samples.

## Data Sources and Citations

- **Pauken CM et al.** "Heterogeneity of Circulating Tumor Cell Neoplastic Subpopulations Outlined by Single-Cell Transcriptomics." *Cancers*, 2021. DOI: [10.3390/cancers13194885](https://doi.org/10.3390/cancers13194885). GEO: GSE174463/GSE174461. PMID: 34638368.
  - Primary CTC dataset. 6 samples from 3 metastatic breast cancer patients (Lin-/Lin+ fractions). 9,044 cells before QC.

- **Szczerba BM et al.** "Neutrophils Escort Circulating Tumor Cells to Enable Cell Cycle Progression." *Nature*, 2019. DOI: [10.1038/s41586-019-1872-7](https://doi.org/10.1038/s41586-019-1872-7). GEO: GSE109761. PMID: 30728496.
  - Additional CTC dataset. 73 human breast cancer CTC cells after QC.

- **10x Genomics.** "3k PBMCs from a Healthy Donor." Cell Ranger 1.1.0.
  - Negative control (non-CTC). 2,698 cells after QC from a healthy female donor.

- **Geneformer:** Theodoris CV et al. "Transfer learning enables predictions in network biology." *Nature*, 2023. HuggingFace: `ctheodoris/Geneformer`.

- **LoRA:** Hu EJ et al. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*, 2022.

## Results Figures

All figures are in `results/figures/`:

| File | What it Shows |
|------|--------------|
| `roc_pr_curves.png` | ROC curve and Precision-Recall curve for both base and fine-tuned models |
| `confusion_matrix.png` | Confusion matrix of the fine-tuned model at threshold=0.5 |
| `calibration.png` | Calibration plot and score distribution |
| `umap_overview.png` | 4-panel UMAP: predictions, ground truth, uncertainty, EpCAM status |
| `umap_ctc_probability.png` | UMAP colored by predicted CTC probability |
| `umap_epcam_status.png` | UMAP colored by EpCAM expression status |

## Honest Limitations

1. **Class imbalance is not clinically realistic.** The test set is 76% CTC, but in a real blood sample, CTCs are vanishingly rare (often 1 in 10^6 to 10^7 blood cells). The reported metrics would look very different at clinically relevant frequencies -- false positives would dominate.

2. **The fine-tuned checkpoint is not saved locally.** Fine-tuning was completed on Google Colab, and the checkpoint was not transferred to the local machine. Current local evaluation uses metrics from the Colab run. The base model (no fine-tuning) is available locally and performs poorly (AUROC ~0.51).

3. **EpCAM-high sensitivity is modest.** The model detects 66.7% of EpCAM-high CTCs, but this is based on only 18 cells in the test set, so the estimate is imprecise (95% CI: 41% -- 87%).

4. **No cross-patient or cross-platform validation.** All CTC cells come from the same study (Pauken 2021). The model has not been tested on different cancer types, different sequencing platforms, or different patient populations.

5. **Limited non-CTC diversity.** Non-CTC cells come from healthy PBMC only. The model has not been tested against activated lymphocytes, circulating endothelial cells, or other cell types found in cancer patient blood.

6. **Single train/test split.** No k-fold cross-validation was performed. Results may depend on the specific random split (seed=42).

7. **No probability calibration.** Predicted scores have not been calibrated with temperature scaling or Platt scaling, so they should not be interpreted as true probabilities.

8. **UMAP is computed on expression, not embeddings.** The UMAP visualizations show gene expression space (scanpy pipeline: normalize, log1p, HVG 2000, PCA 30, neighbors, UMAP), not the Geneformer embedding space. Model predictions are overlaid after the fact.

See `METHODS.md` for detailed explanations of the methods.

## License

MIT
