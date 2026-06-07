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

## Installation

### Requirements

- Python >= 3.10
- ~8 GB RAM (for model loading on CPU; GPU recommended for training)
- ~2 GB disk space for the model checkpoint

### Setup

```bash
git clone https://github.com/gabufle/ctc-detect.git
cd ctc-detect
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the CLI tool with dependencies: `typer` (CLI), `rich` (terminal output).

For training and evaluation, additional packages are needed:

```bash
pip install torch transformers peft datasets scanpy anndata scikit-learn matplotlib
```

The Geneformer model itself (~1.2 GB) is downloaded automatically from HuggingFace on first use and cached locally.

## CLI Usage

### Download the model

```bash
ctc-detect model download
```

### Validate input data

```bash
ctc-detect validate --input data/raw/sample.h5ad
```

### Run detection on a single sample

```bash
ctc-detect run --input data/raw/sample.h5ad --output results/sample_output/
```

Options:
- `--threshold 0.5` — classification threshold (default: 0.5)
- `--skip-umap` — skip UMAP visualization (faster)

### Batch process multiple samples

```bash
ctc-detect batch --input-dir data/raw/ --output-dir results/batch_output/
```

### Evaluate predictions against ground truth

```bash
ctc-detect evaluate --predictions results/predictions.csv --ground-truth data/labels.csv --output results/eval/
```

### Model info

```bash
ctc-detect model info
ctc-detect model list
```

## Training

See `notebooks/ctc_detect_colab.ipynb` for the full training workflow (Google Colab, GPU required). The notebook covers:

1. Data loading and inspection
2. Gene symbol to Ensembl ID conversion
3. Tokenization for Geneformer
4. Train/val/test split
5. LoRA fine-tuning
6. Evaluation and visualization

To train locally:

```bash
python src/ctcdetect/train_geneformer.py
```

## Project Structure

```
ctc-detect/
├── src/ctcdetect/          # Source code
│   ├── main.py             # CLI entry point (Typer)
│   ├── detect.py           # Inference logic
│   ├── preprocess.py       # Input format handling
│   ├── evaluate.py         # Metrics computation
│   ├── report.py           # Report generation
│   ├── visualize.py        # UMAP plots
│   ├── config.py           # Model registry and config
│   ├── utils.py            # Shared utilities
│   └── train_geneformer.py # Training script
├── tests/                  # Test suite (98 tests, 85% coverage)
│   ├── test_main.py
│   ├── test_detect.py
│   ├── test_preprocess.py
│   ├── test_evaluate.py
│   ├── test_report.py
│   ├── test_visualize.py
│   ├── test_extra.py
│   ├── test_main_mocked.py
│   └── conftest.py
├── notebooks/               # Jupyter notebooks
│   └── ctc_detect_colab.ipynb
├── .github/workflows/      # CI/CD (lint, test, typecheck)
├── pyproject.toml          # Package config and dependencies
├── Makefile                # Test runner targets
└── README.md
```

## Data Sources and Citations

- **Pauken CM et al.** "Heterogeneity of Circulating Tumor Cell Neoplastic Subpopulations Outlined by Single-Cell Transcriptomics." *Cancers*, 2021. DOI: [10.3390/cancers13194885](https://doi.org/10.3390/cancers13194885). GEO: GSE174463/GSE174461. PMID: 34638368.
  - Primary CTC dataset. 6 samples from 3 metastatic breast cancer patients (Lin-/Lin+ fractions). 9,044 cells before QC.

- **Szczerba BM et al.** "Neutrophils Escort Circulating Tumor Cells to Enable Cell Cycle Progression." *Nature*, 2019. DOI: [10.1038/s41586-019-1872-7](https://doi.org/10.1038/s41586-019-1872-7). GEO: GSE109761. PMID: 30728496.
  - Additional CTC dataset. 73 human breast cancer CTC cells after QC.

- **10x Genomics.** "3k PBMCs from a Healthy Donor." Cell Ranger 1.1.0.
  - Negative control (non-CTC). 2,698 cells after QC from a healthy female donor.

- **Geneformer:** Theodoris CV et al. "Transfer learning enables predictions in network biology." *Nature*, 2023. HuggingFace: `ctheodoris/Geneformer`.

- **LoRA:** Hu EJ et al. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*, 2022.

## Honest Limitations

1. **Class imbalance is not clinically realistic.** The test set is 76% CTC, but in a real blood sample, CTCs are vanishingly rare (often 1 in 10^6 to 10^7 blood cells). The reported metrics would look very different at clinically relevant frequencies -- false positives would dominate.

2. **The fine-tuned checkpoint is not saved locally.** Fine-tuning was completed on Google Colab, and the checkpoint was not transferred to the local machine. Current local evaluation uses metrics from the Colab run. The base model (no fine-tuning) is available locally and performs poorly (AUROC ~0.51).

3. **EpCAM-high sensitivity is modest.** The model detects 66.7% of EpCAM-high CTCs, but this is based on only 18 cells in the test set, so the estimate is imprecise (95% CI: 41% -- 87%).

4. **No cross-patient or cross-platform validation.** All CTC cells come from the same study (Pauken 2021). The model has not been tested on different cancer types, different sequencing platforms, or different patient populations.

5. **Limited non-CTC diversity.** Non-CTC cells come from healthy PBMC only. The model has not been tested against activated lymphocytes, circulating endothelial cells, or other cell types found in cancer patient blood.

6. **Single train/test split.** No k-fold cross-validation was performed. Results may depend on the specific random split (seed=42).

7. **No probability calibration.** Predicted scores have not been calibrated with temperature scaling or Platt scaling, so they should not be interpreted as true probabilities.

8. **UMAP is computed on expression, not embeddings.** The UMAP visualizations show gene expression space (scanpy pipeline: normalize, log1p, HVG 2000, PCA 30, neighbors, UMAP), not the Geneformer embedding space. Model predictions are overlaid after the fact.

## License

MIT
