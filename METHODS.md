# Methods

**ctc-detect: Geneformer-based Circulating Tumor Cell Detection from Single-Cell RNA-seq**

*Written for a wet bench biologist audience. All technical concepts are explained in plain language before being used.*

---

## Background and Biological Motivation

### What are CTCs?

When a solid tumor grows, some cells can break away and enter the bloodstream. These **circulating tumor cells (CTCs)** are the seeds of metastasis -- they can travel to distant organs and grow into new tumors. Counting CTCs in a blood sample is an approved prognostic test: more CTCs generally means worse outcomes.

### Why do we need a new detection method?

The current clinical standard, **CellSearch** (FDA-approved), catches CTCs by using antibodies that stick to a surface protein called **EpCAM** (epithelial cell adhesion molecule). Think of it like a magnet that pulls out cells wearing a specific name tag.

The problem is that many CTCs -- especially the ones that are most aggressive and most likely to metastasize -- take off that name tag. Through a process called **epithelial-to-mesenchymal transition (EMT)**, these cells downregulate epithelial proteins like EpCAM and become more mobile. CellSearch simply cannot see them.

Single-cell RNA sequencing (scRNA-seq) measures what genes are active in individual cells, regardless of what proteins are on their surface. This gives us a way to identify CTCs based on what they *are* rather than what they *look like* to an antibody.

### The core idea

If we can teach a computational model to recognize the gene expression "fingerprint" of a CTC from scRNA-seq data, we can detect CTCs that CellSearch misses. That is what this project does.

---

## Data

### CTC-positive samples

**Pauken et al. 2021** (GEO: GSE174463/GSE174461) provided the primary CTC dataset. This study performed scRNA-seq on blood samples from 3 metastatic breast cancer patients. Each patient's blood was split into two fractions:

- **Lin-** (lineage-negative): white blood cells were depleted, enriching for non-immune cells including CTCs
- **Lin+** (lineage-positive): contains mostly white blood cells

After quality control filtering, this contributed **8,458 cells** labeled as CTC-enriched.

**Szczerba et al. 2019** (GEO: GSE109761) provided an additional **73 CTC cells** from breast cancer patients, identified by experimental isolation of CTC clusters.

### Negative control (non-CTC) samples

**10x Genomics PBMC 3k**: Peripheral blood mononuclear cells from a healthy donor (2,698 cells after QC). These represent the "normal" cells you find in blood -- no CTCs expected.

### Composition summary

| Source | Cells (post-QC) | CTC | non-CTC |
|--------|----------------|-----|---------|
| Pauken 2021 | 8,458 | 8,458 | 0 |
| Szczerba 2019 | 73 | 73 | 0 |
| 10x PBMC 3k | 2,698 | 0 | 2,698 |
| **Total** | **11,156** | **8,531** | **2,698** |

The data was split 70/15/15 into train/validation/test sets, stratified by CTC label (meaning each split has the same ~76% / 24% CTC/non-CTC ratio).

### Quality control

Cells were filtered using standard scRNA-seq criteria:
- Minimum 200 genes detected per cell (removes empty droplets)
- Maximum 6,000 genes per cell (removes doublets)
- Maximum 20% mitochondrial reads (removes dying cells)
- Genes must be detected in at least 3 cells

---

## What is Geneformer?

**Geneformer** is a transformer model pretrained on ~30 million human single-cell RNA-seq profiles. To understand what it does, let's break down the key ideas.

### Genes ranked by expression

Traditional RNA-seq analysis treats a cell's gene expression as a vector: a long list of numbers, one per gene. Geneformer takes a different approach. It converts each cell into a **ranked list of genes**, ordered from most expressed to least expressed.

For example, if a cell expresses *MALAT1* most highly, then *B2M*, then *EEF1A1*, and so on, Geneformer sees the sequence:

```
[MALAT1, B2M, EEF1A1, RPL10, ...]
```

Each gene is represented by its Ensembl ID (a stable identifier like ENSG00000251562 for MALAT1). Genes with zero expression are not included. The result is a variable-length "sentence" where the "words" are genes, and the "grammar" is the expression ranking.

### The transformer reads the ranking

A **transformer** is a type of neural network originally developed for natural language processing. It reads a sequence of tokens (in this case, gene IDs) and learns relationships between them. Just as a language model learns that "king" and "queen" are related, Geneformer learns that certain genes tend to appear near each other in expression rankings for specific cell types.

The specific Geneformer checkpoint used here (**Geneformer-V1-10M**) was pretrained on a large collection of human scRNA-seq data using a "masked language modeling" objective: given a partial gene ranking, predict the masked-out genes. This forces the model to learn the statistical structure of human gene expression.

### The CLS token and cell embeddings

Like many transformer models, Geneformer adds a special token at the beginning of every sequence called **[CLS]** (short for "classification"). After the model processes the gene ranking, the CLS token's hidden state serves as a **summary vector** for the entire cell -- a single 1,152-dimensional number, called the **cell embedding**, that captures the essential expression features of that cell.

Before fine-tuning, these embeddings are general -- they capture cell type identity but have no concept of "CTC" vs. "non-CTC." Fine-tuning teaches the model to push CTC embeddings in a different direction from non-CTC embeddings.

---

## What is LoRA?

The base Geneformer model has **317 million parameters** (individual numbers that define its behavior). Fine-tuning all of them would require enormous computational resources -- far more than a typical lab computer has.

**LoRA** (Low-Rank Adaptation) is a technique that dramatically reduces what needs to be trained. Instead of modifying all 317 million parameters, LoRA adds a small "adapter" layer to specific parts of the model. Think of it like adding a tuning knob to an existing machine rather than rebuilding the whole thing.

Concretely, LoRA was applied to the **query** and **value** projection layers of the transformer's attention mechanism -- these are the parts of the model that decide which genes in the ranking to pay attention to. The adapter uses a rank of **8** and an alpha scaling factor of **16**.

The result: only **959,234 parameters** (0.30% of the total) are trainable. The other 99.7% remain frozen at their pretrained values. This makes it feasible to fine-tune on a standard GPU (the training for this project was completed on Google ColRA with a free T4 GPU).

---

## The Classification Head

On top of the frozen (LoRA-augmented) Geneformer, we attach a small **classification head** -- a simple neural network that takes the 1,152-dimensional CLS embedding as input and outputs a single number between 0 and 1, interpreted as the probability that the cell is a CTC.

The classification head is trained with **binary cross-entropy loss**, the standard objective for binary classification. Since the training data has more CTCs than non-CTCs (76% / 24%), **class weights** are applied so the model does not simply learn to predict "CTC" for everything. Specifically, the loss for each non-CTC cell is weighted 1.52x more heavily, and each CTC cell 0.48x, balancing the classes.

---

## Handling Class Imbalance

Class imbalance is a central challenge in this project. There are two separate imbalance issues:

### 1. Dataset-level imbalance

The training data has ~76% CTC cells because it combines CTC-enriched samples with a smaller healthy PBMC reference. The model could achieve 76% accuracy by guessing "CTC" for every cell. We address this with class-weighted loss (described above) and by evaluating with metrics that are insensitive to class proportions (AUROC, AUPRC).

### 2. Clinical-level imbalance (a much bigger problem that we do NOT address)

In a real blood sample from a cancer patient, CTCs are extraordinarily rare -- often 1 in a million or ten million blood cells. Our model was *not* tested at these frequencies. The reported performance (93% sensitivity, 97.5% specificity) would change dramatically at clinically realistic ratios, likely with many more false positives. See the limitations section in the README.

---

## Evaluation Metrics Explained

### AUROC (Area Under the Receiver Operating Characteristics Curve)

The **ROC curve** plots the true positive rate (sensitivity) against the false positive rate (1 - specificity) at every possible classification threshold. A threshold is the cutoff: if the model says "this cell has a 70% chance of being a CTC" and your threshold is 50%, you call it a CTC.

**AUROC** summarizes this curve as a single number between 0 and 1. A value of 0.5 means the model is no better than random guessing. A value of 1.0 means perfect separation: there exists a threshold where every CTC is correctly identified with zero false positives.

**What it tells you:** How well the model *ranks* cells by their likelihood of being a CTC, regardless of what threshold you pick. AUROC = 0.9883 means that if you randomly pick one CTC and one non-CTC, the model will give the CTC a higher score 98.83% of the time.

### AUPRC (Area Under the Precision-Recall Curve)

The **precision-recall curve** plots precision (PPV: of the cells called CTCs, how many actually are?) against recall (sensitivity: of the actual CTCs, how many did we catch?) at every threshold.

**AUPRC** is especially informative when classes are imbalanced. While AUROC can look optimistic when one class dominates, AUPRC focuses on how well the model finds the minority class. In our case, since CTCs are the majority class in the test set, AUPRC is less sensitive to the imbalance issue but still provides a useful complement to AUROC.

**What it tells you:** The model's AUPRC of 0.9946 means that across all thresholds, it maintains very high precision and recall simultaneously.

### Sensitivity and Specificity

- **Sensitivity** (true positive rate): Of all actual CTCs, what fraction did the model correctly identify? Value: 0.9307 (93.1%).
- **Specificity** (true negative rate): Of all actual non-CTCs, what fraction did the model correctly reject? Value: 0.9753 (97.5%).

These are reported at a threshold of 0.5 (cells scoring above 0.5 are called CTCs).

---

## UMAP Visualization Explained

**UMAP** (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique. It takes high-dimensional data (here, each cell has expression measurements for ~7,500 genes) and squishes it down to 2 dimensions that you can plot on a page. Cells with similar expression profiles end up near each other; dissimilar cells end up far apart.

### How the UMAP was computed

This is computed using the standard scanpy single-cell analysis pipeline:
1. Normalize counts to 10,000 per cell, then log-transform
2. Select the 2,000 most variable genes (genes whose expression varies most across cells -- these carry the most information about cell identity)
3. Reduce to 30 principal components (PCA) to remove noise
4. Compute a neighborhood graph (which cells are close to which in the 30-dimensional PCA space)
5. Run UMAP on the neighborhood graph to produce the 2D layout

### How to interpret the probability coloring

In the UMAP plots (`results/figures/umap_ctc_probability.png`), each cell is a dot positioned by its expression profile. The dot is **colored** by the model's predicted CTC probability: one color for low probability (likely non-CTC) and another for high probability (likely CTC).

If the model works well, you'll see clear spatial separation: cells in one region of the UMAP tend to be colored as CTCs, and cells in another region tend to be colored as non-CTCs. The transition zone -- where colors blend -- represents cells the model finds ambiguous.

### What "EpCAM status" means in the UMAP

Cells are colored as EpCAM-high or EpCAM-low based on their EPCAM gene expression level. This lets you visually check whether the model is finding EpCAM-low CTCs (the clinically important ones) or just detecting EpCAM expression. The model does the former: most EpCAM-low cells in the CTC-enriched region score high for CTC probability.

---

## Training Details

- **Model:** Geneformer-V1-10M (pretrained, with LoRA adapters r=8, alpha=16)
- **Trainable parameters:** 959,234 (0.30% of 317,291,522 total)
- **Optimizer:** AdamW with learning rate 1e-4, linear warmup over 976 steps
- **Batch size:** 16 (effective, using gradient accumulation)
- **Epochs:** 20 maximum (with early stopping on validation loss)
- **Platform:** Google Colab (free tier, T4 GPU)
- **Loss function:** Weighted binary cross-entropy (class weights: non-CTC=1.516, CTC=0.484)

---

## Software Versions

- Python >= 3.10
- PyTorch (training framework)
- HuggingFace Transformers + PEFT (model loading and LoRA)
- scanpy 1.11.5 (single-cell analysis)
- scikit-learn (metrics)
- matplotlib (figures)
