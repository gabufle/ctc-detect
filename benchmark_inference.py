#!/usr/bin/env python3
"""Quick benchmark for inference speed on CPU."""
import torch
from transformers import BertModel
from datasets import load_from_disk
import numpy as np
import time

torch.set_num_threads(6)

model_path = "/home/gabuf/.hermes/profiles/ml-engineer/home/.cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots/04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5"
print("Loading model...")
t0 = time.time()
model = BertModel.from_pretrained(model_path, trust_remote_code=True)
model.eval()
print(f"Loaded in {time.time()-t0:.1f}s")

ds = load_from_disk("/home/gabuf/projects/ctc-detect/data/processed/tokenized_pbmc_10k/")
print(f"PBMC cells: {len(ds)}")

# Test different configs
for max_len in [128, 256]:
    for bs in [1, 4]:
        n_test = min(bs * 3, len(ds))
        t0 = time.time()
        for i in range(0, n_test, bs):
            batch_ids = []
            for j in range(i, min(i+bs, n_test)):
                ids = ds[j]['input_ids'][:max_len]
                batch_ids.append(ids)
            max_l = max(len(x) for x in batch_ids)
            padded = [x + [0]*(max_l - len(x)) for x in batch_ids]
            masks = [[1]*len(x) + [0]*(max_l - len(x)) for x in batch_ids]
            ids_t = torch.tensor(padded, dtype=torch.long)
            mask_t = torch.tensor(masks, dtype=torch.long)
            with torch.no_grad():
                out = model(input_ids=ids_t, attention_mask=mask_t)
                cls = out.last_hidden_state[:, 0, :]
                scores = torch.norm(cls, dim=1)
        elapsed = time.time() - t0
        per_cell = elapsed / n_test
        est_total = per_cell * 11537 / 60
        print(f"max_len={max_len}, bs={bs}: {per_cell:.3f}s/cell, est={est_total:.1f}min for 11537 cells")
