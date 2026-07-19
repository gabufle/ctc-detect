# ctc-detect

**Geneformer-based Circulating Tumor Cell (CTC) detection from single-cell RNA-seq data.**

> ### ⚠️ Project Status — Proof of Concept (weights not yet published)
>
> **The fine-tuned LoRA adapter weights are not yet published, and the headline results below have not been independently reproduced.** The reported AUROC (~0.99) comes from a one-off Google Colab training run whose checkpoint was never exported. We are deliberately holding off on publishing weights until the model has been **retrained and validated on a patient-level split** — and, ideally, on an external dataset — specifically to rule out confirmation bias from the current single, non-grouped train/test split.
>
> As a direct consequence, **the CLI cannot currently perform real CTC inference.** It runs on the base Geneformer model only (which scores near random), and the detection path is gated behind a `PeftModel` adapter check until a validated adapter is published. See **Intended Use / Out-of-Scope Use**, **Current Limitations**, and **Roadmap / Next Steps** below.

## Why This Matters

Circulating tumor cells (CTCs) are cancer cells that have detached from a primary tumor and entered the bloodstream. Detecting them in a blood draw ("liquid biopsy") can reveal whether a cancer is spreading, how it is responding to treatment, and what its molecular profile looks like -- all without needing an invasive tissue biopsy.

The current FDA-approved standard for CTC detection is **CellSearch**, which works by fishing out cells that carry a surface protein called **EpCAM** (epithelial cell adhesion molecule). The problem: many CTCs -- especially those that have undergone epithelial-to-mesenchymal transition (EMT), a process linked to metastasis -- express little or no EpCAM. These **EpCAM-low CTCs** are invisible to CellSearch, yet they may be the most dangerous and clinically relevant subset.

That background motivates the question this project explores — but **ctc-detect is not an attempt to compete with or replace CellSearch or any clinical-grade detection assay.** CellSearch performs physical *enrichment/capture* of cells from blood; that is not what this project does. Instead, ctc-detect is a **proof-of-concept for post-enrichment classification**: given single cells that have already been isolated and sequenced (scRNA-seq), can a transformer model (Geneformer) use learned gene-expression embeddings to distinguish CTCs — including EpCAM-low CTCs — from normal blood cells? The contribution is a demonstration of transformer-based classification *downstream* of enrichment, not a replacement for the enrichment step itself.

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

> **These fine-tuned numbers are preliminary and unverified.** They come from the single, unreproduced Colab run described in **Project Status** above — a single train/test split with no patient-level grouping and no external validation. Treat them as a proof-of-concept signal, not a validated result.

## Intended Use / Out-of-Scope Use

**Intended use.** This project is a **research and portfolio demonstration** of applying a fine-tuned transformer (Geneformer + LoRA) to single-cell RNA-seq classification. It is intended for methodological exploration, learning, and discussion of transformer-based post-enrichment CTC classification.

**Out-of-scope use.** This is **not** a clinical or diagnostic tool. It must **not** be used to inform patient care, diagnosis, prognosis, or treatment decisions. The model has **not** been validated across patients or cohorts, has not been calibrated, and has not undergone any regulatory review. It is not a substitute for CellSearch or any validated CTC assay.

## Quick Start

```bash
# 1. Install (Python 3.10–3.12 recommended; see note below)
git clone https://github.com/gabufle/ctc-detect.git
cd ctc-detect
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Verify your input data works
ctc-detect validate --input data/raw/sample.h5ad

# 3. Check system info
ctc-detect info

# 4. Download base Geneformer model (optional, auto-downloads on first run)
ctc-detect model download
```

> **Python version note:** This project uses `transformers==4.41.0` and `peft==0.11.0` which were validated with Python 3.10–3.11 and PyTorch 2.3.x. Python 3.12+ will pull newer PyTorch (≥2.6) which is API-compatible for inference but not tested for training. For exact training parity, use Python 3.10–3.11 (CI tests on 3.10/3.11/3.12).

## CLI Usage — What Works Today

| Command | Status | Notes |
|---------|--------|-------|
| `validate` | ✅ Working | Format/QC checks only, no model needed |
| `evaluate` | ✅ Working | Metrics from existing predictions CSV |
| `info` | ✅ Working | System info, cached models, disk space |
| `model list` | ✅ Working | Shows available model versions |
| `model download` | ✅ Working | Downloads base Geneformer from HF |
| `onboard` | ✅ Working | Interactive dataset onboarding (NEW) |
| `run` | ❌ Gated | Needs published LoRA adapter |
| `batch` | ❌ Gated | Same adapter gating as `run` |
| `multi` | ❌ Gated | Same adapter gating as `run` |

> **⚠️ Real CTC inference is not functional yet.** The `run`, `batch`, and `multi` commands are gated behind a `PeftModel` adapter check and will refuse to run without a proper LoRA adapter. The base Geneformer model scores near random (AUROC ~0.51). See **Project Status** and **Roadmap**.

## Dataset Onboarding (Prepare External Data)

Before you can evaluate or train on external datasets, they must be converted to the standardized format (`data.h5ad` + `ground_truth.csv`). The `onboard` command wraps the preparation scripts with **interactive confirmation at every judgment call**:

```bash
ctc-detect onboard --input-path <file_or_dir> --output-dir data/external/<name>
```

**What it does (7 interactive steps):**

| Step | Action |
|------|--------|
| 1. Input shape | Detects single file vs directory of per-cell GSM files vs `.tar.gz` archive |
| 2. Compression + delimiter | Peeks first 5 lines, counts tabs vs commas, shows preview |
| 3. Orientation + metadata cols | Heuristics for genes×cells vs cells×genes; finds metadata columns (Entrez, uniGene, symbol, name…) before sample data; asks for gene ID column + sample start column |
| 4. Normalization state | Peeks numeric values; classifies as raw_counts / log_cpm / tpm_fpkm / unknown; **critical warning** about silent failures |
| 5. Label source | CSV file (barcode→label) **or** colname-regex JSON config (creates template if missing) |
| 6. Run `prepare_external_dataset.py` | Builds exact CLI args; shows full command; asks to execute |
| 7. Patient ID pattern | For `combine_training_datasets.py`; saves to `patient_id_pattern.json` |

**Examples:**

```bash
# Single GSE67980-style file with metadata columns
ctc-detect onboard -i data/raw/GSE67980_processed.txt -o data/external/gse67980

# Directory of per-cell GSM files (runs merge_per_cell_files.py interactively)
ctc-detect onboard -i data/raw/gse109761_raw/ -o data/external/gse109761

# .tar.gz archive (lists contents, offers to extract)
ctc-detect onboard -i data/raw/GSE123456_raw.tar.gz -o data/external/gse123456

# Skip merge step for non-per-cell directories
ctc-detect onboard -i data/raw/some_dir/ -o data/external/some_dir --skip-merge
```

**Output** in `data/external/<name>/`:
- `data.h5ad` — AnnData with `obs['is_ctc']` and `obs['epcam_status']`
- `ground_truth.csv` — `barcode,true_label`
- `patient_id_pattern.json` — regex for `combine_training_datasets.py`

---

### Process multiple individual files (`multi`)

> **New!** Process multiple files of different formats in one command.

```bash
ctc-detect multi sample1.h5ad sample2.csv sample3.tsv --output results/multi_output/
```

**Supported file formats:**
- `.h5ad` — AnnData HDF5 files
- `.csv` — CSV matrices (genes × cells)
- `.tsv` — Tab-separated matrices (genes × cells)
- `.txt` — Text matrices (genes × cells)
- `.mtx` — Matrix Market Exchange format
- `.loom` — Loom files
- Cell Ranger directories (filtered_feature_bc_matrix/)

Options:
- `--threshold 0.5` — classification threshold (default: 0.5)
- `--skip-umap` — skip UMAP visualization (faster)
- `--output -o` — output directory (required)

## Training

See `notebooks/ctc_detect_colab.ipynb` for the full training workflow (Google Colab, GPU required). The notebook covers:

1. Data loading and inspection
2. Gene symbol to Ensembl ID conversion
3. Tokenization for Geneformer
4. Train/val/test split
5. LoRA fine-tuning
6. Evaluation and visualization

To train locally (programmatic API):

```python
from ctcdetect.training import train_model
from ctcdetect.data import prepare_external_dataset

# Prepare data first, then train
train_model(
    train_dataset="data/tokenized/train_dataset",
    val_dataset="data/tokenized/val_dataset",
    base_model_path="Geneformer/Geneformer-V1-10M",
    output_dir="results/checkpoints/best_model",
)
```

## Project Structure

```
ctc-detect/
├── src/ctcdetect/           # Source code (package)
│   ├── __init__.py          # Public API
│   ├── cli/                 # CLI commands (Typer)
│   │   ├── app.py           # App factory
│   │   ├── utils.py         # CLI utilities
│   │   └── commands/        # Individual commands
│   │       ├── run.py       # run, batch, multi
│   │       ├── validate.py
│   │       ├── evaluate.py
│   │       ├── info.py
│   │       ├── model.py
│   │       └── onboard.py
│   ├── core/                # Detection pipeline
│   │   ├── detect.py        # run_detection, tokenization, inference
│   │   ├── model.py         # Model loading, adapter validation
│   │   ├── preprocess.py    # Format detection, QC, normalization
│   │   └── pipeline.py      # CTCDetectionPipeline orchestration
│   ├── evaluation/          # Metrics, reports, plots
│   │   ├── metrics.py       # compute_metrics
│   │   ├── reports.py       # generate_report, generate_html_report
│   │   └── plots.py         # generate_umap, plot_roc_pr
│   ├── config/              # Configuration system
│   │   ├── loader.py        # load_config, get_config
│   │   ├── paths.py         # Path constants
│   │   ├── registry.py      # Model registry
│   │   ├── system.py        # get_system_info
│   │   └── schemas.py       # Pydantic validation schemas
│   ├── training/            # Fine-tuning pipeline
│   │   ├── trainer.py       # train_model, create_lora_config
│   │   └── data.py          # Training data preparation
│   ├── data/                # Data loading & onboarding
│   │   └── onboarding.py    # Onboarding utilities
│   ├── extensions/          # Model backend plugin interface
│   │   └── geneformer.py    # GeneformerBackend
│   └── exceptions.py        # Custom exception hierarchy
├── tests/                   # Test suite (98 tests, ~85% coverage)
│   ├── unit/                # Unit tests mirroring src structure
│   ├── integration/         # Integration tests
│   └── fixtures/            # Test fixtures
├── notebooks/               # Jupyter notebooks
│   └── ctc_detect_colab.ipynb
├── configs/                 # Configuration files (YAML)
│   ├── preprocess.yaml      # Main config (includes all sub-configs)
│   ├── qc.yaml
│   ├── normalize.yaml
│   ├── gene_mapping.yaml
│   ├── tokenize.yaml
│   ├── umap.yaml
│   ├── inference.yaml
│   └── output.yaml
├── data/
│   ├── raw/                 # Raw input data
│   ├── external/            # Onboarded datasets
│   └── processed/           # Processed training data
├── scripts/                 # Legacy scripts (kept for compatibility)
│   ├── onboard_new_dataset.py
│   ├── prepare_external_dataset.py
│   ├── combine_training_datasets.py
│   └── merge_per_cell_files.py
├── .github/workflows/      # CI/CD (lint, test, typecheck)
├── pyproject.toml          # Package config and dependencies
├── Makefile                # Test runner targets
├── Dockerfile              # Multi-stage Docker build
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

## Current Limitations

1. **The fine-tuned checkpoint has not been re-exported or published.** Fine-tuning was completed in a one-off Google Colab run, and the resulting LoRA adapter was never exported. The headline metrics come from that run and **have not been independently reproduced**. Only the base Geneformer model is available locally, and it performs near random (AUROC ~0.51).

2. **Single train/test split with no patient-level grouping.** A single random split (seed=42) was used, with no patient-ID-aware grouping and no k-fold cross-validation. Because cells from the same patient can appear in both train and test, the reported metrics may be optimistically biased.

3. **No external or held-out validation dataset.** All evaluation uses an internal split of the same source studies (CTC cells all from Pauken 2021). The model has not been tested on any external/independent CTC dataset, nor on different cancer types, sequencing platforms, or patient populations.

4. **Class imbalance is not clinically realistic.** The test set is 76% CTC, but in a real blood sample, CTCs are vanishingly rare (often 1 in 10^6 to 10^7 blood cells). The reported metrics would look very different at clinically relevant frequencies — false positives would dominate.

5. **EpCAM-high sensitivity is modest.** The model detects 66.7% of EpCAM-high CTCs, but this is based on only 18 cells in the test set, so the estimate is imprecise (95% CI: 41% — 87%).

6. **Limited non-CTC diversity.** Non-CTC cells come from healthy PBMC only. The model has not been tested against activated lymphocytes, circulating endothelial cells, or other cell types found in cancer patient blood.

7. **No probability calibration.** Predicted scores have not been calibrated with temperature scaling or Platt scaling, so they should not be interpreted as true probabilities.

8. **UMAP is computed on expression, not embeddings.** The UMAP visualizations show gene expression space (scanpy pipeline: normalize, log1p, HVG 2000, PCA 30, neighbors, UMAP), not the Geneformer embedding space. Model predictions are overlaid after the fact.

## Roadmap / Next Steps

1. **Re-run training with a patient-ID-aware split** so no patient appears in both train and test, removing the main suspected source of leakage in the current metrics.
2. **Evaluate on an external CTC dataset** (an independent study/cohort) if a suitable one is available.
3. **Publish the LoRA adapter weights** — the adapter only, not a full checkpoint — once external validation looks reasonable.
4. **Re-enable the CLI for real inference** by shipping the published adapter, which satisfies the `PeftModel` adapter check the detection path is gated behind.

## License

MIT