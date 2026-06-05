#!/usr/bin/env python3
"""Quick benchmark: single batch inference speed."""
import time
import torch
from transformers import BertModel

model_path = "/home/gabuf/.hermes/profiles/ml-engineer/home/.cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots/04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5"
print("Loading model...")
model = BertModel.from_pretrained(model_path, trust_remote_code=True)
model.eval()
print("Model loaded")

max_len = 128
bs = 8

dummy_ids = torch.randint(0, 1000, (bs, max_len)).long()
dummy_mask = torch.ones(bs, max_len).long()

# Warmup
print("Warmup...")
with torch.no_grad():
    for _ in range(3):
        out = model(input_ids=dummy_ids, attention_mask=dummy_mask)
        cls = out.last_hidden_state[:, 0, :]
        scores = torch.norm(cls, dim=1)
print("Warmup done")

# Benchmark
n_iters = 10
t0 = time.time()
with torch.no_grad():
    for _ in range(n_iters):
        out = model(input_ids=dummy_ids, attention_mask=dummy_mask)
        cls = out.last_hidden_state[:, 0, :]
        scores = torch.norm(cls, dim=1)
elapsed = (time.time() - t0) / n_iters
print(f"BS={bs}, seq={max_len}: {elapsed:.3f}s/batch, {elapsed/bs:.4f}s/cell")
print(f"For 11537 cells: {elapsed/bs * 11537 / 60:.1f} min")

# Try torch.compile
print("\nTrying torch.compile...")
try:
    compiled = torch.compile(model, mode="reduce-overhead")
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            out = compiled(input_ids=dummy_ids, attention_mask=dummy_mask)
            cls = out.last_hidden_state[:, 0, :]
    
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            out = compiled(input_ids=dummy_ids, attention_mask=dummy_mask)
            cls = out.last_hidden_state[:, 0, :]
    elapsed = (time.time() - t0) / n_iters
    print(f"Compiled BS={bs}: {elapsed:.3f}s/batch, {elapsed/bs:.4f}s/cell")
except Exception as e:
    print(f"torch.compile failed: {e}")
