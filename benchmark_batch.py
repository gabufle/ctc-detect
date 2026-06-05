#!/usr/bin/env python3
"""Benchmark batch inference with preloaded data."""
import time
import numpy as np
import torch
from transformers import BertModel

model_path = "/home/gabuf/.hermes/profiles/ml-engineer/home/.cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots/04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5"
model = BertModel.from_pretrained(model_path, trust_remote_code=True)
model.eval()

max_len = 128
n_cells = 11537

# Pre-load all data
from datasets import load_from_disk
ds = load_from_disk("/home/gabuf/projects/ctc-detect/data/processed/tokenized_pbmc_10k/")
all_ids = np.zeros((n_cells, max_len), dtype=np.int32)
for j in range(n_cells):
    ids = ds[j]['input_ids'][:max_len]
    all_ids[j, :len(ids)] = ids

# Create mask
lengths = np.array([np.count_nonzero(row) for row in all_ids])
mask = np.zeros((n_cells, max_len), dtype=np.int32)
for i, l in enumerate(lengths):
    mask[i, :l] = 1

# Convert to tensors
all_ids_t = torch.from_numpy(all_ids).long()
mask_t = torch.from_numpy(mask).long()

print(f"Data preloaded: {n_cells} cells, {max_len} max_len")

# Test different batch sizes
for bs in [4, 8, 16, 32, 64]:
    if bs > n_cells:
        continue
    # Warmup
    with torch.no_grad():
        _ = model(input_ids=all_ids_t[:bs], attention_mask=mask_t[:bs])

    n_batches = min(5, n_cells // bs)
    t0 = time.time()
    with torch.no_grad():
        for b in range(n_batches):
            _ = model(input_ids=all_ids_t[b*bs:(b+1)*bs], attention_mask=mask_t[b*bs:(b+1)*bs])
    elapsed = (time.time() - t0) / n_batches
    per_cell = elapsed / bs
    est_total = per_cell * n_cells / 60
    print(f"BS={bs}: {elapsed:.3f}s/batch, {per_cell:.4f}s/cell, est={est_total:.1f}min for {n_cells}")
