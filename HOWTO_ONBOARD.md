# Quick Start: `onboard_new_dataset.py`

Interactive orchestrator that turns **one raw dataset** into standardized `data.h5ad` + `ground_truth.csv` by chaining existing prep scripts — **pausing for human confirmation at every judgment call**.

---

## Basic Usage

```bash
cd /path/to/ctc-detect
python scripts/onboard_new_dataset.py \
    --input-path <file_or_directory> \
    --output-dir data/external/<dataset_name>
```

---

## What Happens (7 Steps)

| Step | What it detects | You confirm |
|------|-----------------|-------------|
| **1. Input shape** | Single file vs directory of per-cell files vs `.tar.gz` | "Treat as per-cell files needing merge?" |
| **2. Compression + delimiter** | Peeks 5 lines via `gzip.open`/`open`; counts tabs vs commas | Shows first 5 lines + detected delimiter |
| **3. Orientation + metadata cols** | Heuristics: genes×cells vs cells×genes; finds metadata cols (Entrez, uniGene, symbol, name…) | Prints ALL columns in groups of 10; asks for gene ID column index + sample start column index |
| **4. Normalization state** | Peeks numeric values → raw_counts / log_cpm / tpm_fpkm / unknown | **Critical**: shows min/max/decimals/negatives; warns about silent failures |
| **5. Label source** | File-based CSV **or** colname-regex JSON config (creates template if missing) | Prompts for label CSV path/cols/positive values **OR** label config JSON |
| **6. Run `prepare_external_dataset.py`** | Builds exact CLI args matching that script's signature | Shows full command; "Execute?" |
| **7. Patient ID pattern** | For `combine_training_datasets.py` | Dataset name + regex (capture group 1 = patient ID); saves to `patient_id_pattern.json` |

---

## Example: GSE67980-style file (metadata columns before samples)

```bash
python scripts/onboard_new_dataset.py \
    --input-path data/raw/GSE67980_processed.txt \
    --output-dir data/external/gse67980
```

**What you'll see at step 3:**
```
ALL column names:
  [  0] Entrez GeneID
  [  1] uniGene
  [  2] symbol
  [  3] name
  [  4] GSM1234567_sample1
  [  5] GSM1234568_sample2
  ...

Potential metadata columns detected:
  [0] Entrez GeneID
  [1] uniGene
  [2] symbol <- suggested gene ID
  [3] name

Which column index should be used as the GENE IDENTIFIER? [2]:
Which column index does the FIRST SAMPLE COLUMN start at? [4]:
```

---

## Example: Per-cell directory (GEO GSM* files)

```bash
python scripts/onboard_new_dataset.py \
    --input-path data/raw/gse109761_raw/ \
    --output-dir data/external/gse109761
```

- Runs `merge_per_cell_files.py --inspect-only` first
- Prompts for custom `--file-glob`, `--filename-pattern`, `--sep`, `--has-header`
- Continues with the merged matrix

---

## Example: .tar.gz archive

```bash
python scripts/onboard_new_dataset.py \
    --input-path data/raw/GSE123456_raw.tar.gz \
    --output-dir data/external/gse123456
```

- Lists archive contents
- Offers to extract to temp dir and re-run

---

## Label Configuration

### Option 1: Separate CSV file
```
Choose [1/2]: 1
Path to labels CSV: /path/to/labels.csv
Barcode column name [barcode]: barcode
Label column name [label]: cell_type
Comma-separated positive label values: tumor,CTC
```

### Option 2: Regex from column names (no separate file)
```
Choose [1/2]: 2
Path to label config JSON: configs/labels/gse109761.json
```

**Config template created if missing:**
```json
{
  "positive_patterns": [],
  "negative_patterns": ["(Bcells?|Tcells?|NK|Mono|Gra|plts?)$"],
  "unmatched": "exclude"
}
```

---

## Output

```
data/external/<dataset_name>/
├── data.h5ad              # AnnData with adata.obs['is_ctc'], adata.obs['epcam_status']
├── ground_truth.csv       # barcode, true_label
└── patient_id_pattern.json # {dataset_name, patient_id_regex, output_h5ad}
```

---

## Next Steps After Onboarding

```bash
# 1. Spot-check
python -c "import scanpy as sc; a=sc.read('data/external/gse67980/data.h5ad'); print(a.obs[['is_ctc','epcam_status']].value_counts())"

# 2. Evaluate
python scripts/run_and_eval.py --input data/external/gse67980/data.h5ad --ground-truth data/external/gse67980/ground_truth.csv --output results/gse67980

# 3. Combine for training
python scripts/combine_training_datasets.py \
    --datasets gse67980=data/external/gse67980/data.h5ad gse109761=data/external/gse109761/data.h5ad \
    --output data/combined_training_set.h5ad
```

---

## Tips

- **Answer `y` (default)** to accept detected values
- **Press Enter** to accept bracketed defaults
- **Type `e`** at any `[y/n/e]` prompt to edit/override
- **Ctrl+C** to abort at any step
- Use `--skip-merge` if input is a directory but NOT per-cell files