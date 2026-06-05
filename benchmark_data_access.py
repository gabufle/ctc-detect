#!/usr/bin/env python3
"""Benchmark different data access patterns for HF datasets."""
import time
import numpy as np
import torch
from datasets import load_from_disk

ds = load_from_disk("/home/gabuf/projects/ctc-detect/data/processed/tokenized_pbmc_10k/")
n = len(ds)
print(f"Dataset size: {n}")

# Method 1: dataset[j] random access
t0 = time.time()
for j in range(8):
    _ = ds[j]['input_ids'][:128]
t1 = time.time()
print(f"Method 1 (dataset[j]): {(t1-t0)/8:.4f}s/cell")

# Method 2: dataset[i:j] slice
t0 = time.time()
batch = ds[0:8]
t1 = time.time()
ids = [x[:128] for x in batch['input_ids']]
t2 = time.time()
print(f"Method 2 (ds[0:8] slice): {(t1-t0):.4f}s for slice, {(t2-t1):.4f}s for process")

# Method 3: ds.with_format('numpy')
ds_np = ds.with_format('numpy')
t0 = time.time()
for j in range(8):
    _ = ds_np[j]['input_ids'][:128]
t1 = time.time()
print(f"Method 3 (numpy format): {(t1-t0)/8:.4f}s/cell")

# Method 4: Preload all to list
t0 = time.time()
all_ids = [ds[j]['input_ids'] for j in range(100)]
t1 = time.time()
print(f"Method 4 (preload 100 to list): {(t1-t0):.4f}s for 100, {(t1-t0)/100:.4f}s/cell")

# Method 5: Preload all to numpy array (padded)
t0 = time.time()
max_len = 128
all_ids_padded = np.zeros((n, max_len), dtype=np.int32)
for j in range(n):
    ids = ds[j]['input_ids'][:max_len]
    all_ids_padded[j, :len(ids)] = ids
t1 = time.time()
print(f"Method 5 (preload all n={n} to numpy): {(t1-t0):.1f}s total, {(t1-t0)/n:.4f}s/cell")

# Test: preloaded numpy batch inference speed
from transformers import BertModel
model_path = "/home/gabuf/.hermes/profiles/ml-engineer/home/.cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots/04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5"
model = BertModel.from_pretrained(model_path, trust_remote_code=True)
model.eval()

# Batch from preloaded numpy
batch_np = all_ids_padded[:8]
lengths = np.array([np.count_nonzero(row) for row in batch_np])
mask_np = np.zeros((8, max_len), dtype=np.int32)
for i, l in enumerate(lengths):
    mask_np[i, :l] = 1

ids_t = torch.from_numpy(batch_np).long()
mask_t = torch.from_numpy(mask_np).long()

# warmup
with torch.no_grad():
    _ = model(input_ids=ids_t, attention_mask=mask_t)

t0 = time.time()
for _ in range(5):
    with torch.no_grad():
        out = model(input_ids=ids_t, attention_mask=mask_t)
        cls = out.last_hidden_state[:, 0, :]
elapsed = (time.time() - t0) / 5
print(f"\nPreloaded numpy batch inference (bs=8, seq=128): {elapsed:.3f}s/batch, {elapsed/8:.4f}s/cell")
print(f"Estimated for {n} cells: {elapsed/8 * n / 60:.1f} min")
