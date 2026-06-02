# Data Provenance — CTC Detection Training Datasets

**Download date:** 2026-06-02  
**Downloaded by:** OWL (data-hunter) for the ctc-detection-project  
**Storage:** `~/projects/ctc-detect/data/raw/`

---

## IMPORTANT NOTE — GSE145926 is NOT the target dataset

The task description referenced GSE145926 as a "Zhang et al. 2021 breast cancer CTC" dataset. **This accession is incorrect.** GSE145926 is:

> "Single-cell landscape of bronchoalveolar immune cells in COVID-19 patients"  
> Liao et al., 2020 (medRxiv / Nat Med) — BALF from COVID-19 patients, **not breast cancer**

The correct primary dataset for breast cancer CTC scRNA-seq from the cited paper (Pauken et al. / Marchetti lab, 2021, *Cancers* 13:4885, doi:10.3390/cancers13194885) is:

- **GSE174463** (SuperSeries) containing GSE174461 (scRNA-seq SubSeries) — the Pauken/Marchetti CTC dataset
- Secondary CTC dataset: **GSE109761** — Szczerba et al., 2019 (*Nature* 566:553)

---

## Dataset 1 — GSE174463 / GSE174461 (PRIMARY)

**Paper:** Pauken CM, Kenney SR, Brayer KJ, Guo Y, Brown-Glaberman UA, Marchetti D. "Heterogeneity of Circulating Tumor Cell Neoplastic Subpopulations Outlined by Single-Cell Transcriptomics." *Cancers* 2021, 13(19), 4885. doi:10.3390/cancers13194885. PMID: 34638368.

**Source:** NCBI GEO SuperSeries GSE174463 (SubSeries GSE174461 for scRNA-seq)  
**Direct download URL:** https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174463  
**BioProject:** PRJNA730015  
**Data type:** 10x Genomics Chromium scRNA-seq (3' gene expression) + bulk RNA-seq  
**Organism:** Homo sapiens

### GSE174461 — Single-cell RNA-seq (Lin−/Lin+ sorted populations)

**Format:** MEX (Matrix Market Exchange) — `barcodes.tsv.gz`, `features.tsv.gz`, `matrix.mtx.gz` per sample  
**Genome:** GRCh38 (Ensembl gene IDs, e.g. ENSG00000243485)

| Sample ID | Description | Cells | Genes | Non-zero entries |
|-----------|-------------|-------|-------|------------------|
| GSM5311807 | Pt1681 Lin− | 841 | 33,538 | 1,088,444 |
| GSM5311808 | Pt1681 Lin+ | 1,968 | 33,538 | 3,675,413 |
| GSM5311809 | Pt2087 Lin− | 1,028 | 33,538 | 547,682 |
| GSM5311810 | Pt2087 Lin+ | 1,033 | 33,538 | 1,542,321 |
| GSM5311811 | Pt2148 Lin− | 874 | 33,538 | 413,909 |
| GSM5311812 | Pt2148 Lin+ | 3,300 | 33,538 | 5,508,913 |
| **Total** | | **9,044** | | |

**Cell sorting:** Lin− (lineage-negative, enriched for CTCs) and Lin+ (CD45+, immune cells) sorted by flow cytometry from metastatic breast cancer patients.

**Metadata / annotations in features.tsv:**
- Column 1: Ensembl gene ID (e.g. ENSG00000243485)
- Column 2: Gene symbol (e.g. MIR1302-2HG)
- Column 3: Feature type ("Gene Expression")

**CTC marker genes to look for in the data:** EPCAM, TACSTD2 (TROP2), KRT7/8/18/19, CD45 (PTPRC), CEACAM6, CLDN4, ERBB2.

**Paper identifies CTC-candidate cluster (cluster 10):** Cells co-expressing epithelial markers (EPCAM, KRTs) found in BOTH Lin− and Lin+, suggesting CTCs exist in both sorted populations. The Lin− fraction is enriched but not pure.

### GSE174463 (additional bulk RNA-seq samples)

37 bulk RNA-seq count files from additional patients (both Lin−/Lin+ sorted populations and PBMC controls). Format: gzipped txt (exon/feature counts). Useful for validation but not single-cell resolution.

**File:** `GSE174461_RAW.tar` (180 MB) — extracted to `GSE174461/`  
**File:** `GSE174463_RAW.tar` (217 MB) — extracted to `GSE174463_pauken_2021_ctc/`

### Integrity verification
- All `.gz` files pass `gzip -t` integrity check
- Features.tsv has 33,538 genes (GRCh38 Ensembl annotation)
- Barcode counts match MTX matrix dimensions
- Files sizes are reasonable (no zero-byte files)

---

## Dataset 2 — GSE109761 (SECONDARY CTC DATASET)

**Paper:** Szczerba BM, Castro-Giner F, Vetter M, et al. "Neutrophils Escort Circulating Tumor Cells to Enable Cell Cycle Progression." *Nature* 2019, 566:553–557. PMID: 30728496.

**Source:** NCBI GEO GSE109761  
**Direct download URLs:**
- RAW: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109761
- Processed matrix (human): https://ftp.ncbi.nlm.nih.gov/geo/series/GSE109nnn/GSE109761/processed/GSE109761_processed_normalized_matrix_hs.txt.gz
- SCE object (human): https://ftp.ncbi.nlm.nih.gov/geo/series/GSE109nnn/GSE109761/processed/GSE109761_sce_hs.rds.gz
**BioProject:** PRJNA431985  
**Data type:** Single-cell RNA-Seq (Smart-seq2 and 10x) — CTCs and CTC-WBC clusters from breast cancer patients + mouse models  
**Organism:** Homo sapiens + Mus musculus (human data separated)

**Processed normalized matrix:** 19,955 genes × 357 cells (human CTCs)  
**Format:** Tab-separated gzipped text. Columns are cell barcodes (GSM IDs), rows are genes (Geneid). Values are normalized expression.

**Sample naming convention:** `{Model}_{Replicate}.hs` (human) or `.mm` (mouse). LM2 = lung metastasis model; PDX = patient-derived xenograft.

**Note:** The RDS files (`sce_hs.rds.gz`, `sce_mm.rds.gz`) are R/Bioconductor SingleCellExperiment objects and are not directly loadable in Python without reticulate. The processed TSV matrix is the most practical Python-friendly format.

**Files:**
- `GSE109761_RAW.tar` (63 MB in archive, extracts to 89 MB) — raw per-cell count files
- `GSE109761_processed_normalized_matrix_hs.txt.gz` (26 MB) — processed, normalized, human-only matrix

---

## Dataset 3 — 10x Genomics PBMC 3k Reference (NEGATIVE CONTROL)

**Descriptor:** PBMC from a Healthy Donor — No Cell Sorting (3k)  
**Source URL:** https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0  
**Direct download URL:** https://cf.10xgenomics.com/samples/cell-exp/2.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz  
**10x Genomics dataset ID:** pbmc3k  
**Version:** Cell Ranger 2.1.0  
**Organism:** Homo sapiens (healthy female donor, age 25)  
**Source:** AllCells (cryopreserved PBMCs)  
**Estimated cells:** 2,700  
**Sequencing:** Illumina NextSeq 500, ~69,000 reads/cell  
**Chemistry:** 10x Chromium 3' v2

**Format:** MEX (Matrix Market) — `barcodes.tsv`, `genes.tsv`, `matrix.mtx`  
**Genome:** hg19 (GRCh37)  
**Genes:** 32,738 (Ensembl gene IDs)  
**Barcodes (cells):** 2,700

**Location in raw directory:**
- `pbmc3k/filtered_gene_bc_matrices/hg19/` — extracted MEX files
- `pbmc_10x_reference/filtered_gene_bc_matrices/hg19/` — duplicate extract from prior download

**Cell types expected (standard PBMC composition):** T cells (CD4+ and CD8+), B cells, NK cells, monocytes (classical and non-classical), dendritic cells, megakaryocytes. **No CTCs expected.** This serves as the negative control / normal blood baseline.

**License:** 10x Genomics data is freely available for academic use; see https://www.10xgenomics.com/datasets

---

## Additional Notes

### What was NOT downloaded
- GSE144494 (Ebright et al. CTC dataset) — only provides read count XLS files (1.8 MB), very processed, low information content for training. Available if needed.
- GSE145926 — confirmed COVID-19 dataset, wrong accession for this task.

### Matched PBMC from the same Pauken paper
The GSE174463 SuperSeries GSE174431 ("Bulk RNAseq Analysis of PBMCs from metastatic breast cancer patients") contains PBMC data from the same study. These are bulk (not single-cell) RNA-seq. The Lin+ sorted fractions from GSE174461 also serve as matched "normal blood" immune cell references, being CD45+ leukocyte-enriched populations from the same patients.

For a true healthy PBMC scRNA-seq negative control, use the 10x PBMC 3k dataset above.

### .gitignore status
The project `.gitignore` in `~/projects/ctc-detect/` should exclude:
```
data/raw/*.h5
data/raw/*.h5ad
data/raw/*.tar
data/raw/*.tar.gz
data/raw/*.mtx
data/raw/*.tsv.gz
data/raw/*.csv.gz
data/raw/*.rds.gz
data/raw/*/filtered_*/
data/raw/GSE*/
```

The provenance file `data/raw/data_provenance.md` is committed to git; raw data files are NOT.
