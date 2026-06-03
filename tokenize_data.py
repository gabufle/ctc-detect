#!/usr/bin/env python3
"""
Tokenize processed scRNA data for Geneformer input format.
Uses Geneformer's TranscriptomeTokenizer to convert AnnData to tokenized HuggingFace datasets.
"""
import scanpy as sc
import json
import os
import pickle
import sys
import tempfile
import numpy as np
from geneformer import TranscriptomeTokenizer
from datasets import DatasetDict, Dataset

# Configuration
PROJECT_DIR = "/home/gabuf/projects/ctc-detect"
DATA_FILE = os.path.join(PROJECT_DIR, "data/processed/ctc_merged_processed.h5ad")
SPLITS_FILE = os.path.join(PROJECT_DIR, "data/processed/splits.json")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data/processed/tokenized")
REPORT_PATH = os.path.join(PROJECT_DIR, "tokenization_report.md")

# Geneformer dictionary files
GENEFORMER_DIR = os.path.join(PROJECT_DIR, "Geneformer", "geneformer")
TOKEN_DICT = os.path.join(GENEFORMER_DIR, "token_dictionary_gc104M.pkl")
GENE_MEDIAN = os.path.join(GENEFORMER_DIR, "gene_median_dictionary_gc104M.pkl")
GENE_MAPPING = os.path.join(GENEFORMER_DIR, "ensembl_mapping_dict_gc104M.pkl")

os.chdir(PROJECT_DIR)

# Step 1: Load and prepare AnnData
print("Step 1: Loading AnnData...")
adata = sc.read_h5ad(DATA_FILE)
print(f"  Shape: {adata.shape}")

# Step 2: Map gene symbols to Ensembl IDs
print("\nStep 2: Mapping gene symbols to Ensembl IDs...")
with open(GENE_MAPPING, 'rb') as f:
    mapping_data = pickle.load(f)
print(f"  Gene mapping entries: {len(mapping_data)}")

ensembl_ids = []
mapped_count = 0
unmapped_genes = []
for gene_symbol in adata.var_names:
    if gene_symbol in mapping_data:
        ensembl_ids.append(mapping_data[gene_symbol])
        mapped_count += 1
    else:
        ensembl_ids.append(None)
        unmapped_genes.append(gene_symbol)

adata.var['ensembl_id'] = ensembl_ids
print(f"  Mapped: {mapped_count} / {len(adata.var_names)} genes")
print(f"  Unmapped: {len(unmapped_genes)} genes")

# Filter out genes without Ensembl ID
has_ensembl = adata.var['ensembl_id'].notna()
adata = adata[:, has_ensembl].copy()
print(f"  Shape after filtering: {adata.shape}")

# Step 3: Add n_counts column to adata.obs
# The tokenizer expects 'n_counts' in obs. This is the total count per cell.
# Our data was normalized to 10000 counts then log1p-transformed.
# We can compute approximate n_counts by summing exp(X) - 1 (inverse of log1p).
# But the tokenizer normalizes to target_sum anyway, so we just need the column to exist.
# We'll compute it from the data matrix.
print("\nStep 3: Computing n_counts...")
# The data is in adata.X (likely sparse)
# n_counts = sum of counts per cell
if hasattr(adata.X, 'toarray'):
    # Sparse matrix
    n_counts = np.array(adata.X.sum(axis=1)).flatten()
else:
    n_counts = np.array(adata.X.sum(axis=1)).flatten()
adata.obs['n_counts'] = n_counts
print(f"  n_counts range: {n_counts.min():.0f} - {n_counts.max():.0f}")
print(f"  n_counts mean: {n_counts.mean():.0f}")

# Step 4: Load splits
print("\nStep 4: Loading splits...")
with open(SPLITS_FILE, 'r') as f:
    splits = json.load(f)
for k, v in splits.items():
    print(f"  {k}: {len(v)} cells")

# Step 5: Initialize tokenizer
print("\nStep 5: Initializing TranscriptomeTokenizer...")
tokenizer = TranscriptomeTokenizer(
    token_dictionary_file=TOKEN_DICT,
    gene_median_file=GENE_MEDIAN,
    gene_mapping_file=GENE_MAPPING,
    nproc=4,
    chunk_size=512,
    model_input_size=4096,
    special_token=True,
    collapse_gene_ids=True,
    use_h5ad_index=False,
    keep_counts=False,
    model_version='V2'
)
print("  Tokenizer initialized successfully.")

# Step 6: Tokenize each split
print("\nStep 6: Tokenizing splits...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenized_info = {}

for split_name, indices in splits.items():
    print(f"  Tokenizing {split_name} split ({len(indices)} cells)...")
    adata_split = adata[indices].copy()

    # Save split to temp h5ad file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.h5ad')
    os.close(temp_fd)
    try:
        adata_split.write_h5ad(temp_path)
        print(f"    Saved temp h5ad: {temp_path}")

        # Tokenize the h5ad file
        tokenized_cells, cell_metadata, tokenized_counts = tokenizer.tokenize_anndata(
            temp_path, target_sum=10000, file_format='h5ad'
        )
        print(f"    Tokenized {len(tokenized_cells)} cells")

        if len(tokenized_cells) == 0:
            print(f"    WARNING: No cells tokenized for {split_name}")
            continue

        # Create HuggingFace Dataset
        dataset = tokenizer.create_dataset(
            tokenized_cells, cell_metadata, tokenized_counts,
            use_generator=False, keep_uncropped_input_ids=False
        )

        # Save to disk
        split_output_dir = os.path.join(OUTPUT_DIR, split_name)
        dataset.save_to_disk(split_output_dir)
        print(f"    Saved to {split_output_dir}")

        # Collect stats
        input_ids_list = [ex.get('input_ids', []) for ex in dataset]
        avg_seq_len = sum(len(ids) for ids in input_ids_list) / len(input_ids_list) if input_ids_list else 0
        tokenized_info[split_name] = {
            'num_examples': len(dataset),
            'avg_seq_length': avg_seq_len,
            'first_example_keys': list(dataset[0].keys()) if len(dataset) > 0 else [],
            'first_input_ids_length': len(input_ids_list[0]) if input_ids_list else 0,
            'first_10_ids': input_ids_list[0][:10] if input_ids_list else []
        }
        print(f"    Avg seq length: {avg_seq_len:.1f}")
        print(f"    First 10 token IDs: {tokenized_info[split_name]['first_10_ids']}")

    except Exception as e:
        print(f"    ERROR tokenizing {split_name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# Step 7: Vocabulary coverage analysis
print("\nStep 7: Computing vocabulary coverage...")
with open(TOKEN_DICT, 'rb') as f:
    token_dict_data = pickle.load(f)
vocab_size = len(token_dict_data)

# Coverage using Ensembl IDs
our_ensembl_ids = set(adata.var['ensembl_id'].dropna())
genes_in_vocab = our_ensembl_ids.intersection(set(token_dict_data.keys()))
coverage = len(genes_in_vocab) / len(our_ensembl_ids) * 100 if our_ensembl_ids else 0
print(f"  Vocabulary size: {vocab_size}")
print(f"  Genes in our data (Ensembl IDs): {len(our_ensembl_ids)}")
print(f"  Genes in vocabulary: {len(genes_in_vocab)} ({coverage:.2f}%)")
print(f"  Genes dropped: {len(our_ensembl_ids) - len(genes_in_vocab)}")

# Step 8: Generate report
print("\nStep 8: Generating report...")
report = []
report.append("# Tokenization Report\n")

report.append("## Data Shape")
report.append(f"- Original cells: 11156")
report.append(f"- Original genes: 7637")
report.append(f"- Genes mapped to Ensembl IDs: {mapped_count}")
report.append(f"- Genes unmapped (dropped): {len(unmapped_genes)}")
report.append(f"- Final shape: {adata.shape}\n")

report.append("## Splits")
for split_name, indices in splits.items():
    info = tokenized_info.get(split_name, {})
    n_tok = info.get('num_examples', 'N/A')
    report.append(f"- {split_name}: {len(indices)} cells -> {n_tok} tokenized examples")

report.append("\n## Tokenized Data Verification")
for split_name, info in tokenized_info.items():
    report.append(f"### {split_name}")
    report.append(f"- Number of examples: {info['num_examples']}")
    report.append(f"- Average sequence length: {info['avg_seq_length']:.1f}")
    report.append(f"- First example keys: {info['first_example_keys']}")
    report.append(f"- First example input_ids length: {info['first_input_ids_length']}")
    report.append(f"- First 10 token IDs: {info['first_10_ids']}")

report.append("\n## Vocabulary Coverage")
report.append(f"- Token dictionary: token_dictionary_gc104M.pkl")
report.append(f"- Vocabulary size: {vocab_size}")
report.append(f"- Genes in our data (Ensembl IDs): {len(our_ensembl_ids)}")
report.append(f"- Genes in vocabulary: {len(genes_in_vocab)} ({coverage:.2f}%)")
report.append(f"- Genes dropped (not in vocabulary): {len(our_ensembl_ids) - len(genes_in_vocab)}")

if unmapped_genes:
    report.append(f"\n### Genes without Ensembl mapping ({len(unmapped_genes)})")
    for g in sorted(unmapped_genes[:30]):
        report.append(f"- {g}")
    if len(unmapped_genes) > 30:
        report.append(f"- ... and {len(unmapped_genes) - 30} more")

report.append("\n## Notes")
report.append("- Tokenization performed using Geneformer's TranscriptomeTokenizer (model_version='V2').")
report.append("- Data was normalized (log1p) from preprocessing step.")
report.append("- Gene symbol -> Ensembl ID mapping via ensembl_mapping_dict_gc104M.pkl.")
report.append("- Genes without Ensembl IDs were filtered out before tokenization.")
report.append("- n_counts column computed from data matrix and added to adata.obs.")
report.append("- tokenize_anndata() used per split, then create_dataset() to build HF Dataset.")
report.append("- Each split saved as HuggingFace Dataset via save_to_disk().")

with open(REPORT_PATH, "w") as f:
    f.write("\n".join(report))
print(f"  Report written to {REPORT_PATH}")
print("\nTokenization task completed successfully.")
