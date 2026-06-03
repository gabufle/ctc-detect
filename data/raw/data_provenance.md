# Data Provenance

**Project:** CTC Detection Model Training
**Generated:** 2026-06-02

---

## Important Note on GEO Accession Discrepancy

The task description referenced "Zhang et al. 2021 breast cancer CTC scRNA-seq dataset" with GEO accession **GSE145926**. However, GSE145926 corresponds to:

> Liao M, Liu Y, Yuan J, et al. "Single-cell landscape of bronchoalveolar immune cells in COVID-19 patients." *Nature Medicine*, 2020.
> GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE145926

This is **not** a breast cancer CTC dataset. The actual matching dataset is:

> Pauken CM, Kenney SR, Brayer KJ, Guo Y, Brown-Glaberman UA, Marchetti D. "Heterogeneity of Circulating Tumor Cell Neoplastic Subpopulations Outlined by Single-Cell Transcriptomics." *Cancers*, 2021. DOI: [10.3390/cancers13194885](https://doi.org/10.3390/cancers13194885)
> GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174463 (SuperSeries) / GSE174461 (SubSeries with scRNA-seq data)

This is the dataset we downloaded. It contains scRNA-seq data from Lin-/Lin+ cell populations isolated from metastatic breast cancer patients, with CTC identification via epithelial markers (EpCAM, cytokeratins).

---

## Dataset 1: Pauken et al. 2021 — Breast Cancer CTCs (PRIMARY TARGET)

- **Paper:** Pauken CM et al., *Cancers* 2021. DOI: 10.3390/cancers13194885
- **GEO Series:** GSE174463 (SuperSeries) / GSE174461 (scRNA-seq SubSeries)
- **BioProject:** PRJNA728700
- **Organism:** Homo sapiens
- **Platform:** Illumina NovaSeq 6000 (GPL24676) — 10x Genomics Chromium
- **Citation PMID:** 34638368

### Download Details
| Item | Value |
|------|-------|
| Source URL (SuperSeries) | https://ftp.ncbi.nlm.nih.gov/geo/series/GSE174nnn/GSE174463/suppl/GSE174463_RAW.tar |
| Source URL (SubSeries) | https://ftp.ncbi.nlm.nih.gov/geo/series/GSE174nnn/GSE174461/suppl/GSE174461_RAW.tar |
| Download date | 2026-06-02 |
| GSE174463 file size | 216.1 MB |
| GSE174461 file size | 179.1 MB |

### Sample Composition (scRNA-seq data in GSE174461)

The scRNA-seq data consists of 10x Genomics Chromium Cell Ranger 3.1.0 output in MEX (Matrix Market) format with gzipped files. Each sample has three files: `barcodes.tsv.gz`, `features.tsv.gz`, `matrix.mtx.gz`.

| Sample ID | Description | Cells | Genes | Non-zero entries |
|-----------|-------------|-------|-------|------------------|
| GSM5311807 | Pt1681 Lin- | 841 | 33,538 | 1,088,444 |
| GSM5311808 | Pt1681 Lin+ | 1,968 | 33,538 | 3,675,413 |
| GSM5311809 | Pt2087 Lin- | 1,028 | 33,538 | 547,682 |
| GSM5311810 | Pt2087 Lin+ | 1,033 | 33,538 | 1,542,321 |
| GSM5311811 | Pt2148 Lin- | 874 | 33,538 | 413,909 |
| GSM5311812 | Pt2148 Lin+ | 3,300 | 33,538 | 5,508,913 |
| **Total** | | **9,044** | **33,538** | **12,776,882** |

- Lin- = Lineage negative (CD45-, enriched for "classic" CTCs)
- Lin+ = Lineage positive (CD45+, contains WBCs and some CTCs)

### Format
- **Format:** MEX (Matrix Market Exchange format), gzipped
- **Structure:** barcodes.tsv.gz (cell barcodes), features.tsv.gz (gene annotations: ENSEMBL ID, gene symbol, feature type), matrix.mtx.gz (sparse UMI count matrix)
- **Software:** Cell Ranger 3.1.0

### Key Metadata / Marker Genes
- **CTC identity markers:** EPCAM (ENSG00000119888) — present in features; 67/841 EPCAM+ cells in Pt1681 Lin- alone
- **Epithelial markers available:** EPCAM, KRT7/8/9/18/19 (cytokeratins), CLDN4, CLDN7, TACSTD2, CEACAM6
- **Immune markers:** CD45 (PTPRC), CD3D, CD14, CD19, etc. (in Lin+ samples)
- Lin- samples are CD45-depleted (enriched for non-immune cells including CTCs)
- Lin+ samples contain WBC populations plus any CTCs that co-express CD45

### Paper-reported CTC identification
Per the paper, cluster 10 was identified as CTCs based on epithelial/mammary tissue gene expression (within the integrated UMAP clustering of all 6 samples, total ~5801 high-quality cells). The paper notes that the "CTC-candidate" cluster contains 201 cells from both Lin- and Lin+ populations that are enriched for EPCAM, KRT7/9/18/19, TACSTD2, CLDN4, CLDN7, CEACAM6.

### Verification
- Files pass scanpy `read_10x_mtx()` loading test: Pt1681 Lin- loaded successfully (841 cells x 33,538 genes)
- EPCAM expression confirmed: 67/841 cells express EPCAM in Pt1681 Lin-
- File sizes are consistent with GEO records

---

## Dataset 2: Szczerba et al. 2019 — Breast Cancer CTCs (NEGATIVE CONTROL / ADDITIONAL)

- **Paper:** Szczerba BM, Castro-Giner F, Vetter M, et al. "Neutrophils Escort Circulating Tumor Cells to Enable Cell Cycle Progression." *Nature*, 2019. DOI: [10.1038/s41586-019-1872-7](https://doi.org/10.1038/s41586-019-1872-7)
- **GEO Series:** GSE109761
- **BioProject:** PRJNA431985
- **Organisms:** Homo sapiens (primary), Mus musculus (mouse model)
- **Platform:** Illumina NextSeq 500 (GPL18573)
- **Citation PMID:** 30728496

### Download Details
| Item | Value |
|------|-------|
| Source URL | https://ftp.ncbi.nlm.nih.gov/geo/series/GSE109nnn/GSE109761/suppl/GSE109761_RAW.tar |
| Processed matrix URL | https://ftp.ncbi.nlm.nih.gov/geo/series/GSE109nnn/GSE109761/suppl/GSE109761_processed_normalized_matrix_hs.txt.gz |
| Download date | 2026-06-02 |
| RAW tar size | 62.9 MB |
| Processed matrix size | 25.8 MB |

### Sample Composition
- **357 human single-cell samples** (breast cancer patient CTCs and CTC-WBC clusters)
- **114 mouse samples** (4T1 and PyMT mouse models)
- Plus 1 processed normalized expression matrix (human only)

### Format
- **Raw format:** Individual per-cell text files (`{sample_id}.hs.raw_counts.txt.gz`), single-column UMI counts per gene
- **Gene annotation:** gene symbols as row identifiers (33,214 genes)
- **Processed format:** Tab-delimited normalized matrix (genes x 358 cells), gzipped

### Verification
- Processed matrix loads with pandas: 358 columns (cells), 33,214+ genes
- Sample file structure confirmed

---

## Dataset 3: 10x Genomics PBMC 3k — Healthy Donor PBMC Reference (NEGATIVE CONTROL)

- **Source:** 10x Genomics official dataset
- **Dataset:** "3k PBMCs from a Healthy Donor" (Cell Ranger 1.1.0, hg19)
- **Organism:** Homo sapiens
- **Donor:** Healthy female, age 25 (AllCells)
- **Protocol:** 10x Genomics Chromium Single Cell 3' v2
- **Sequencing:** Illumina NextSeq 500, ~69,000 reads/cell
- **Website:** https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0
- **Alternative source (HDF5):** https://figshare.com/articles/dataset/3k_PBMCs_from_a_healthy_donor/28414916 (Cell Ranger 8.0.1, GRCh38)

### Download Details
| Item | Value |
|------|-------|
| Source URL | https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz |
| Download date | 2026-06-02 |
| File size | 7.3 MB |

### Sample Composition
| Metric | Value |
|--------|-------|
|Cells | 2,700 |
| Genes | 32,738 |
| Reference genome | GRCh37/hg19 |
| Chemistry version | 10x Chromium v2 |

### Format
- **Format:** MEX (Matrix Market), uncompressed
- **Files:** barcodes.tsv (cell barcodes), genes.tsv (gene symbols), matrix.mtx (sparse UMI count matrix)
- **Directory:** `filtered_gene_bc_matrices/hg19/`

### Expected Cell Type Markers (verified via scanpy)
| Marker | Cell Type | Positive Cells |
|--------|-----------|---------------|
| CD3D | T cells | 1,405 / 2,700 |
| CD14 | Monocytes | 388 / 2,700 |
| CD19 | B cells | 97 / 2,700 |
| NKG7 | NK cells | 816 / 2,700 |

### Verification
- Loads successfully with scanpy `read_10x_mtx()`: 2,700 cells x 32,738 genes
- Known PBMC marker genes present and expressed in expected proportions
- EPCAM is in gene list but very low expression expected (no epithelial cells)

---

## Data Processing Notes

1. **No format conversion has been performed.** All files are in their original downloaded format.
2. **No quality filtering has been applied.** Cell counts are as reported by the original data providers.
3. **The GSE174463 SuperSeries also contains bulk RNA-seq data** (GSE174431) and additional samples not listed here — these are in the RAW tar but not the scRNA-seq subseries.
4. **GSE109761 mouse data** (114 samples) is included in the download but not the processed matrix.
5. **No matched PBMC data from the Pauken et al. paper** was found as a separate download — the paper's PBMC data is part of the GSE174463 SuperSeries (bulk RNA-seq, GSE174431).

---

## Directory Structure

```
~/projects/ctc-detect/data/raw/
├── GSE174463_pauken_2021_ctc/     # Pauken 2021 CTC dataset (GSE174463 + GSE174461)
│   ├── GSE174463_RAW.tar          # Original archive (216 MB)
│   ├── GSE174461_RAW.tar          # SubSeries archive (179 MB)
│   ├── GSM5311807_Pt1681LinMinus_*.tsv.gz/.mtx.gz  # 6 scRNA-seq samples
│   ├── GSM5311808_Pt1681LinPlus_*.tsv.gz/.mtx.gz
│   ├── GSM5311809_Pt2087LinMinus_*.tsv.gz/.mtx.gz
│   ├── GSM5311810_Pt2087LinPlus_*.tsv.gz/.mtx.gz
│   ├── GSM5311811_Pt2148LinMinus_*.tsv.gz/.mtx.gz
│   ├── GSM5311812_Pt2148LinPlus_*.tsv.gz/.mtx.gz
│   └── [bulk RNA-seq files from GSE174463]
├── GSE109761_szczerba_2019_ctc/   # Szczerba 2019 CTC dataset
│   ├── GSE109761_RAW.tar          # Original archive (63 MB)
│   ├── GSE109761_processed_normalized_matrix_hs.txt.gz  # 26 MB
│   └── [357 human + 114 mouse single-cell files]
└── pbmc_10x_reference/            # 10x Genomics PBMC 3k reference
    ├── pbmc3k_filtered_gene_bc_matrices.tar.gz  # Original archive (7.3 MB)
    └── filtered_gene_bc_matrices/hg19/
        ├── barcodes.tsv           # 2,700 cell barcodes
        ├── genes.tsv              # 32,738 gene annotations
        └── matrix.mtx            # Sparse UMI count matrix (27 MB)
```
