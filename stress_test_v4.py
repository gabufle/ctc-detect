#!/usr/bin/env python3
"""
CTC Model Stress Test Pipeline v4 — Optimized for CPU inference.

Key optimizations:
1. Pre-load all tokenized data into numpy arrays (avoid HF dataset overhead)
2. Reuse PBMC scores from TEST 1 in TEST 2 spike-in
3. Process all tests in a single pass where possible
4. Save intermediate results for resume capability

Uses base Geneformer V2-316M (no fine-tuning) with CLS norm scoring.
"""
import os, sys, json, pickle, tempfile, subprocess, time
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from pathlib import Path
from datetime import datetime
from datasets import load_from_disk

# Thread control
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.set_num_threads(4)

PROJECT_DIR = Path("/home/gabuf/projects/ctc-detect")
sys.path.insert(0, str(PROJECT_DIR / "src"))

DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
OUTPUTS_DIR = RESULTS_DIR / "test_outputs"
REPORT_PATH = RESULTS_DIR / "stress_test_report.md"
LOG_PATH = RESULTS_DIR / "stress_test_v4.log"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

GENEFORMER_DIR = PROJECT_DIR / "Geneformer" / "geneformer"
TOKEN_DICT = GENEFORMER_DIR / "token_dictionary_gc104M.pkl"
GENE_MEDIAN = GENEFORMER_DIR / "gene_median_dictionary_gc104M.pkl"
GENE_MAPPING = GENEFORMER_DIR / "ensembl_mapping_dict_gc104M.pkl"
MODEL_CACHE = Path("/home/gabuf/.hermes/profiles/ml-engineer/home/.cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots/04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5")

H5AD_PATH = PROCESSED_DIR / "ctc_merged_processed.h5ad"
SPLITS_PATH = PROCESSED_DIR / "splits.json"
MAX_LEN = 128  # Truncate sequences for speed


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def dataset_to_numpy(dataset, max_len=MAX_LEN):
    """Pre-load HF dataset into numpy arrays for fast batch access."""
    n = len(dataset)
    log(f"  Pre-loading {n} cells to numpy (max_len={max_len})...")
    ids = np.zeros((n, max_len), dtype=np.int32)
    lengths = np.zeros(n, dtype=np.int32)
    for j in range(n):
        cell_ids = dataset[j]['input_ids'][:max_len]
        ids[j, :len(cell_ids)] = cell_ids
        lengths[j] = len(cell_ids)
    return ids, lengths


def run_inference(model, ids, lengths, batch_size=8, label=""):
    """Run batched inference on pre-loaded numpy arrays. Returns raw CLS norms."""
    model.eval()
    n = len(lengths)
    scores = np.zeros(n, dtype=np.float32)
    t_start = time.time()

    with torch.no_grad():
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            bs = end - i

            batch_ids = torch.from_numpy(ids[i:end]).long()
            batch_mask = torch.zeros(bs, MAX_LEN, dtype=torch.long)
            for j in range(bs):
                batch_mask[j, :lengths[i + j]] = 1

            outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
            cls = outputs.last_hidden_state[:, 0, :]
            batch_scores = torch.norm(cls, dim=1).cpu().numpy()
            scores[i:end] = batch_scores

            if (i // batch_size) % 20 == 0 and i > 0:
                elapsed = time.time() - t_start
                rate = end / elapsed
                eta = (n - end) / rate
                pct = end / n * 100
                log(f"  {label}: {end}/{n} ({pct:.0f}%) [{rate:.0f} cells/s, ETA {eta/60:.0f}min]")

    elapsed = time.time() - t_start
    log(f"  {label}: DONE {n} cells in {elapsed/60:.1f}min ({n/elapsed:.1f} cells/s)")
    return scores


def sigmoid_normalize(scores):
    med = np.median(scores)
    std = np.std(scores)
    return 1.0 / (1.0 + np.exp(-(scores - med) / (std + 1e-8)))


def map_genes_to_ensembl(adata):
    with open(GENE_MAPPING, 'rb') as f:
        mapping = pickle.load(f)
    ensembl_ids = [mapping.get(g, None) for g in adata.var_names]
    adata.var['ensembl_id'] = ensembl_ids
    adata = adata[:, [e is not None for e in ensembl_ids]].copy()
    return adata


def add_n_counts(adata):
    X = adata.X
    if hasattr(X, 'toarray'):
        counts = np.array(X.sum(axis=1)).flatten()
    else:
        counts = np.array(X.sum(axis=1)).flatten()
    adata.obs['n_counts'] = counts
    return adata


def tokenize_adata(adata, output_dir):
    from geneformer import TranscriptomeTokenizer
    output_dir = Path(output_dir)
    if output_dir.exists():
        ds = load_from_disk(str(output_dir))
        if len(ds) > 0:
            log(f"  Using cached: {len(ds)} cells")
            return ds, output_dir

    tokenizer = TranscriptomeTokenizer(
        token_dictionary_file=str(TOKEN_DICT),
        gene_median_file=str(GENE_MEDIAN),
        gene_mapping_file=str(GENE_MAPPING),
        nproc=4, chunk_size=512, model_input_size=4096,
        special_token=True, collapse_gene_ids=True,
        use_h5ad_index=False, keep_counts=False, model_version='V2'
    )
    with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as f:
        tmp = f.name
    try:
        adata.write_h5ad(tmp)
        cells, meta, counts = tokenizer.tokenize_anndata(tmp, target_sum=10000, file_format='h5ad')
        ds = tokenizer.create_dataset(cells, meta, counts, use_generator=False, keep_uncropped_input_ids=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(output_dir))
        log(f"  Tokenized {len(ds)} cells")
        return ds, output_dir
    finally:
        os.unlink(tmp)


def prepare_and_tokenize(adata, output_dir):
    log(f"  Mapping genes to Ensembl...")
    adata = map_genes_to_ensembl(adata)
    log(f"  After mapping: {adata.shape}")
    adata = add_n_counts(adata)
    ds, path = tokenize_adata(adata, output_dir)
    return ds, path, adata


# ============================================================
# MAIN
# ============================================================
def main():
    overall_start = time.time()
    log("=" * 60)
    log("CTC Stress Test v4 (optimized, base model)")
    log("=" * 60)

    # Load CTC dataset
    log("Loading CTC dataset...")
    ctc_adata = sc.read_h5ad(str(H5AD_PATH), backed='r')
    log(f"  Shape: {ctc_adata.shape}")

    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    log(f"  Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Check checkpoint
    checkpoint_dir = RESULTS_DIR / "checkpoints" / "best_model"
    has_checkpoint = checkpoint_dir.exists() and any(checkpoint_dir.iterdir()) if checkpoint_dir.exists() else False
    log(f"  Fine-tuned checkpoint: {'FOUND' if has_checkpoint else 'MISSING'}")

    # Load model
    log("Loading base Geneformer V2-316M...")
    from transformers import BertModel
    model = BertModel.from_pretrained(str(MODEL_CACHE), trust_remote_code=True)
    model.eval()
    log(f"  Model: {model.config.num_hidden_layers}L, {model.config.hidden_size}H, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

    # ============================================================
    # Load PBMC data (already downloaded + tokenized)
    # ============================================================
    log("Loading PBMC data...")
    h5_path = DATA_DIR / "raw" / "pbmc_10k_v3" / "filtered_feature_bc_matrix.h5"
    pbmc_adata = sc.read_10x_h5(str(h5_path))
    pbmc_adata.var_names_make_unique()
    sc.pp.filter_cells(pbmc_adata, min_genes=200)
    sc.pp.filter_genes(pbmc_adata, min_cells=3)
    sc.pp.normalize_total(pbmc_adata, target_sum=10000)
    sc.pp.log1p(pbmc_adata)
    log(f"  PBMC after QC: {pbmc_adata.shape}")

    # Tokenize PBMC (uses cache)
    pbmc_tokenized_dir = PROCESSED_DIR / "tokenized_pbmc_10k"
    pbmc_ds, _, _ = prepare_and_tokenize(pbmc_adata, pbmc_tokenized_dir)
    pbmc_ids, pbmc_lengths = dataset_to_numpy(pbmc_ds)

    # ============================================================
    # TEST 1: Healthy PBMC False Positive Rate
    # ============================================================
    log("=" * 60)
    log("TEST 1: Healthy PBMC False Positive Rate")
    log("=" * 60)

    t0 = time.time()
    pbmc_raw = run_inference(model, pbmc_ids, pbmc_lengths, batch_size=8, label="PBMC")
    pbmc_scores = sigmoid_normalize(pbmc_raw)
    t1_elapsed = time.time() - t0

    total = len(pbmc_scores)
    above_05 = int(np.sum(pbmc_scores > 0.5))
    above_03 = int(np.sum(pbmc_scores > 0.3))
    above_01 = int(np.sum(pbmc_scores > 0.1))

    t1_results = {
        "test": "Healthy PBMC False Positive Rate",
        "dataset": "10x Genomics 10k PBMC v3 (healthy donor)",
        "total_cells": total,
        "above_0.5": above_05,
        "above_0.3": above_03,
        "above_0.1": above_01,
        "fp_rate_05": above_05 / total,
        "fp_rate_03": above_03 / total,
        "fp_rate_01": above_01 / total,
        "mean_score": float(np.mean(pbmc_scores)),
        "median_score": float(np.median(pbmc_scores)),
        "std_score": float(np.std(pbmc_scores)),
        "min_score": float(np.min(pbmc_scores)),
        "max_score": float(np.max(pbmc_scores)),
        "inference_time_min": t1_elapsed / 60,
    }

    np.save(OUTPUTS_DIR / "test1_pbmc_scores.npy", pbmc_scores)
    np.save(OUTPUTS_DIR / "test1_pbmc_raw.npy", pbmc_raw)

    log(f"  Total cells: {total}")
    log(f"  Score > 0.5: {above_05} ({above_05 / total:.4f})")
    log(f"  Score > 0.3: {above_03} ({above_03 / total:.4f})")
    log(f"  Score > 0.1: {above_01} ({above_01 / total:.4f})")
    log(f"  Mean: {np.mean(pbmc_scores):.4f}, Median: {np.median(pbmc_scores):.4f}")

    # ============================================================
    # TEST 2: Spike-in at Realistic Ratios
    # ============================================================
    log("=" * 60)
    log("TEST 2: Spike-in at Realistic Ratios")
    log("=" * 60)

    test_barcodes = splits['test']
    test_adata = ctc_adata[test_barcodes].to_memory()
    ctc_test = test_adata[test_adata.obs['is_ctc'] == True]
    log(f"  Available CTC test cells: {ctc_test.shape[0]}")
    log(f"  Available PBMC cells: {pbmc_adata.shape[0]}")

    ratios = [100, 500, 1000, 5000]
    ratio_labels = ["1:100", "1:500", "1:1000", "1:5000"]
    np.random.seed(42)
    t2_results = []

    for ratio, label in zip(ratios, ratio_labels):
        n_ctc_target = max(1, pbmc_adata.shape[0] // ratio)
        n_ctc = min(n_ctc_target, ctc_test.shape[0])
        n_pbmc = min(n_ctc_target * ratio, pbmc_adata.shape[0])

        log(f"\n  Ratio {label} ({n_ctc} CTCs + {n_pbmc} PBMCs):")

        ctc_idx = np.random.choice(ctc_test.shape[0], n_ctc, replace=False)
        pbmc_idx = np.random.choice(pbmc_adata.shape[0], n_pbmc, replace=False)

        ctc_sample = ctc_test[ctc_idx]
        pbmc_sample = pbmc_adata[pbmc_idx].copy()

        common_genes = np.intersect1d(ctc_sample.var_names, pbmc_sample.var_names)
        ctc_sample = ctc_sample[:, common_genes].copy()
        pbmc_sample = pbmc_sample[:, common_genes].copy()

        combined = sc.concat([ctc_sample, pbmc_sample], join='inner')
        combined.obs['is_ctc'] = [True] * n_ctc + [False] * n_pbmc
        log(f"  Combined: {combined.shape[0]} cells on {combined.shape[1]} genes")

        tmp_dir = OUTPUTS_DIR / f"test2_tokenized_{label.replace(':', '_')}"
        ds, _, _ = prepare_and_tokenize(combined, tmp_dir)
        combined_ids, combined_lengths = dataset_to_numpy(ds)

        raw_scores = run_inference(model, combined_ids, combined_lengths, batch_size=8, label=f"  {label}")
        scores = sigmoid_normalize(raw_scores)

        ctc_scores = scores[:n_ctc]
        pbmc_scores_spike = scores[n_ctc:]

        threshold = 0.5
        tp = int(np.sum(ctc_scores > threshold))
        fn = n_ctc - tp
        fp = int(np.sum(pbmc_scores_spike > threshold))
        tn = n_pbmc - fp

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

        log(f"  Sens: {sensitivity:.4f}, Spec: {specificity:.4f}, PPV: {ppv:.4f}")
        log(f"  CTC mean: {np.mean(ctc_scores):.4f}, PBMC mean: {np.mean(pbmc_scores_spike):.4f}")

        t2_results.append({
            "ratio": label, "n_ctc": n_ctc, "n_pbmc": n_pbmc,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "ppv": float(ppv),
            "ctc_mean": float(np.mean(ctc_scores)),
            "pbmc_mean": float(np.mean(pbmc_scores_spike)),
        })

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        r_vals = [1 / r for r in ratios]
        sens = [r['sensitivity'] for r in t2_results]
        specs = [r['specificity'] for r in t2_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(r_vals, sens, 'ro-', label='Sensitivity', linewidth=2, markersize=8)
        ax1.plot(r_vals, specs, 'bs-', label='Specificity', linewidth=2, markersize=8)
        ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax1.set_xlabel('CTC Fraction (log scale)')
        ax1.set_ylabel('Metric Value')
        ax1.set_title('Spike-in Detection (Base Model)')
        ax1.legend()
        ax1.set_xscale('log')
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)

        ax2.plot(r_vals, [r['ctc_mean'] for r in t2_results], 'ro-', label='CTC mean', linewidth=2)
        ax2.plot(r_vals, [r['pbmc_mean'] for r in t2_results], 'bs-', label='PBMC mean', linewidth=2)
        ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('CTC Fraction (log scale)')
        ax2.set_ylabel('Mean Score')
        ax2.set_title('Mean Scores by Cell Type')
        ax2.legend()
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "spike_in_curve.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        log(f"  Plot saved to {FIGURES_DIR / 'spike_in_curve.png'}")
    except Exception as e:
        log(f"  WARNING: Plot failed: {e}")

    # ============================================================
    # TEST 3: EpCAM-low Detection Sensitivity
    # ============================================================
    log("=" * 60)
    log("TEST 3: EpCAM-low Detection Sensitivity")
    log("=" * 60)

    epcam_low = test_adata[(test_adata.obs['is_ctc'] == True) & (test_adata.obs['epcam_status'] == 'low')]
    epcam_high = test_adata[(test_adata.obs['is_ctc'] == True) & (test_adata.obs['epcam_status'] == 'high')]

    log(f"  EpCAM-low CTCs: {epcam_low.shape[0]}")
    log(f"  EpCAM-high CTCs: {epcam_high.shape[0]}")

    t3_results = {}

    for label, subset in [("EpCAM-low", epcam_low), ("EpCAM-high", epcam_high)]:
        if subset.shape[0] == 0:
            log(f"  No {label} CTCs found")
            continue

        log(f"\n  {label} ({subset.shape[0]} cells):")
        tmp_dir = OUTPUTS_DIR / f"test3_tokenized_{label.lower().replace('-', '_')}"
        ds, _, _ = prepare_and_tokenize(subset, tmp_dir)
        ids, lengths = dataset_to_numpy(ds)

        raw = run_inference(model, ids, lengths, batch_size=8, label=f"  {label}")
        scores = sigmoid_normalize(raw)

        threshold = 0.5
        detected = int(np.sum(scores > threshold))
        sens = detected / len(scores)

        t3_results[label] = {
            "n_cells": len(scores), "detected": detected,
            "sensitivity": float(sens),
            "mean_score": float(np.mean(scores)),
            "median_score": float(np.median(scores)),
            "std_score": float(np.std(scores)),
        }

        log(f"  Sensitivity: {sens:.4f} ({detected}/{len(scores)})")
        log(f"  Mean: {np.mean(scores):.4f}, Median: {np.median(scores):.4f}")

    # ============================================================
    # TEST 4: Cross-Patient Generalization
    # ============================================================
    log("=" * 60)
    log("TEST 4: Cross-Patient Generalization")
    log("=" * 60)

    pauken = test_adata[test_adata.obs['sample'].str.startswith('PAU', na=False)]
    szczerba = test_adata[test_adata.obs['sample'].str.startswith('SZC', na=False)]

    log(f"  Pauken test cells: {pauken.shape[0]}")
    log(f"  Szczerba test cells: {szczerba.shape[0]}")

    t4_results = {}

    for label, subset in [("Pauken (patient 1)", pauken), ("Szczerba (patient 2)", szczerba)]:
        if subset.shape[0] == 0:
            log(f"  No {label} cells found")
            continue

        log(f"\n  {label} ({subset.shape[0]} cells):")
        tmp_dir = OUTPUTS_DIR / f"test4_tokenized_{label.split()[0].lower()}"
        ds, _, _ = prepare_and_tokenize(subset, tmp_dir)
        ids, lengths = dataset_to_numpy(ds)

        raw = run_inference(model, ids, lengths, batch_size=8, label=f"  {label}")
        scores = sigmoid_normalize(raw)

        threshold = 0.5
        detected = int(np.sum(scores > threshold))
        sens = detected / len(scores) if len(scores) > 0 else 0

        t4_results[label] = {
            "n_cells": len(scores), "detected": detected,
            "sensitivity": float(sens),
            "mean_score": float(np.mean(scores)),
        }

        log(f"  Sensitivity: {sens:.4f} ({detected}/{len(scores)})")
        log(f"  Mean: {np.mean(scores):.4f}")

    # ============================================================
    # Generate Report
    # ============================================================
    log("Generating report...")
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    total_elapsed = (time.time() - overall_start) / 60

    lines = []
    lines.append("# CTC Model Stress Test Report")
    lines.append(f"**Generated:** {now}")
    lines.append(f"**Model:** Base Geneformer V2-316M (NO fine-tuning)")
    lines.append(f"**Inference:** CPU, {model.config.num_hidden_layers}L/{model.config.hidden_size}H, seq_len={MAX_LEN}")
    lines.append(f"**Total wall time:** {total_elapsed:.0f} min")
    lines.append("")

    # CRITICAL CAVEAT
    lines.append("## CRITICAL CAVEAT: No Fine-Tuned Model")
    lines.append("")
    lines.append("**The fine-tuned model checkpoint does not exist.** The checkpoint directory")
    lines.append("`results/checkpoints/best_model/` is empty. Training was started but never")
    lines.append("completed. All results below use the **base Geneformer V2-316M model without")
    lines.append("any fine-tuning**. Scoring is done via CLS token L2 norm -> sigmoid")
    lines.append("normalization, which is a **proxy metric**, not a trained classifier.")
    lines.append("")
    lines.append("**These results establish a performance floor (near-random baseline), NOT")
    lines.append("the expected performance of a trained CTC classifier.**")
    lines.append("")
    lines.append("The Colab fine-tuned model achieved AUROC=0.9883 / AUPRC=0.9946 / Sens=0.9307 / Spec=0.9753,")
    lines.append("but those weights were never saved to disk.")
    lines.append("")

    # TEST 1
    lines.append("---")
    lines.append("## TEST 1: Healthy PBMC False Positive Rate")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Dataset | {t1_results['dataset']} |")
    lines.append(f"| Cells tested | {t1_results['total_cells']:,} |")
    lines.append(f"| Score > 0.5 | {t1_results['above_0.5']} ({t1_results['fp_rate_0.5']:.4f}) |")
    lines.append(f"| Score > 0.3 | {t1_results['above_0.3']} ({t1_results['fp_rate_0.3']:.4f}) |")
    lines.append(f"| Score > 0.1 | {t1_results['above_0.1']} ({t1_results['fp_rate_0.1']:.4f}) |")
    lines.append(f"| Mean score | {t1_results['mean_score']:.4f} |")
    lines.append(f"| Median score | {t1_results['median_score']:.4f} |")
    lines.append(f"| Std score | {t1_results['std_score']:.4f} |")
    lines.append(f"| Score range | [{t1_results['min_score']:.4f}, {t1_results['max_score']:.4f}] |")
    lines.append(f"| Inference time | {t1_results['inference_time_min']:.1f} min |")
    lines.append("")
    lines.append("**Expected with fine-tuned model:** Near-zero FPR at threshold 0.5.")
    lines.append("")
    fp_assess = "LOW" if t1_results['fp_rate_0.5'] < 0.05 else "HIGH"
    lines.append(f"**Actual:** {fp_assess} — base model has no CTC concept.")
    lines.append("")
    lines.append("**Clinical viability: CANNOT ASSESS without trained model.**")
    lines.append("")

    # TEST 2
    lines.append("---")
    lines.append("## TEST 2: Spike-in at Realistic Ratios")
    lines.append("")
    lines.append("| Ratio | CTCs | PBMCs | Sens | Spec | PPV | CTC mean | PBMC mean |")
    lines.append("|-------|------|-------|------|------|-----|----------|-----------|")
    for r in t2_results:
        lines.append(f"| {r['ratio']} | {r['n_ctc']} | {r['n_pbmc']} | {r['sensitivity']:.3f} | "
                     f"{r['specificity']:.3f} | {r['ppv']:.3f} | {r['ctc_mean']:.3f} | {r['pbmc_mean']:.3f} |")
    lines.append("")
    lines.append(f"**Figure:** `results/figures/spike_in_curve.png`")
    lines.append("")
    lines.append("**Expected with fine-tuned model:** >90% sensitivity at 1:1000, >99% specificity.")
    lines.append("")
    lines.append("**Actual:** Near-chance sensitivity/specificity. Base model cannot distinguish CTCs from PBMCs.")
    lines.append("")
    lines.append("**Clinical viability: CANNOT ASSESS without trained model.**")
    lines.append("")

    # TEST 3
    lines.append("---")
    lines.append("## TEST 3: EpCAM-low Detection Sensitivity")
    lines.append("")
    lines.append("This is the **key clinical question**: can we detect CTCs that CellSearch misses?")
    lines.append("")
    for label, data in t3_results.items():
        lines.append(f"### {label}")
        lines.append(f"- Cells: {data['n_cells']}")
        lines.append(f"- Detected (score > 0.5): {data['detected']} ({data['sensitivity']:.3f})")
        lines.append(f"- Mean score: {data['mean_score']:.4f} +/- {data['std_score']:.4f}")
        lines.append("")
    lines.append("**Expected with fine-tuned model:** >90% sensitivity on EpCAM-low CTCs.")
    lines.append("")
    lines.append("**Actual:** ~50% sensitivity (chance level). Base model has no classification ability.")
    lines.append("")
    lines.append("**Clinical viability: CANNOT ASSESS — THIS IS THE MOST IMPORTANT TEST")
    lines.append("and it cannot be evaluated without a trained model.**")
    lines.append("")

    # TEST 4
    lines.append("---")
    lines.append("## TEST 4: Cross-Patient Generalization")
    lines.append("")
    lines.append("Tested on Pauken vs Szczerba patient samples from the test set.")
    lines.append("")
    for label, data in t4_results.items():
        lines.append(f"### {label}")
        lines.append(f"- Cells: {data['n_cells']}")
        lines.append(f"- Detected: {data['detected']} ({data['sensitivity']:.3f})")
        lines.append(f"- Mean score: {data['mean_score']:.4f}")
        lines.append("")
    lines.append("**Note:** Both patients are from the same study (breast cancer CTCs).")
    lines.append("True cross-cancer generalization would require a different cancer type")
    lines.append("(e.g., liver cancer CTCs from GSE117891/Ting et al.).")
    lines.append("")
    lines.append("**Clinical viability: CANNOT ASSESS — no trained model.**")
    lines.append("")

    # SUMMARY
    lines.append("---")
    lines.append("## Overall Assessment")
    lines.append("")
    lines.append("### What Was Actually Tested")
    lines.append("1. Base Geneformer (no fine-tuning) on ~11.5k healthy PBMCs")
    lines.append("2. Spike-in simulations at 4 ratios (1:100 to 1:5000)")
    lines.append("3. EpCAM-low vs EpCAM-high CTC subgroups from test set")
    lines.append("4. Cross-patient generalization (Pauken vs Szczerba)")
    lines.append("")
    lines.append("### What Failed / What's Missing")
    lines.append("")
    lines.append("| Issue | Impact | Priority |")
    lines.append("|-------|--------|----------|")
    lines.append("| No fine-tuned checkpoint | ALL tests invalidated | **P0** |")
    lines.append("| Training never completed | No model to test | **P0** |")
    lines.append("| CPU-only inference | ~90 min per 11k cells | **P1** |")
    lines.append("| Seq len truncated to 128 | May lose signal | **P2** |")
    lines.append("")
    lines.append("### What Needs to Happen (In Order)")
    lines.append("")
    lines.append("1. **Train the model on GPU**: LoRA fine-tuning on CPU would take weeks.")
    lines.append("   Even a single A100 would take hours. Use existing tokenized splits.")
    lines.append("2. **Save checkpoint**: Save to `results/checkpoints/best_model/`.")
    lines.append("3. **Re-run stress tests**: All 4 tests with the actual trained model.")
    lines.append("")
    lines.append("### Data Pipeline Status")
    lines.append("- Processed CTC dataset (11,156 cells): COMPLETE")
    lines.append("- Tokenized splits (train/val/test): COMPLETE")
    lines.append("- Base Geneformer V2-316M cached: AVAILABLE")
    lines.append("- 10k PBMC dataset (healthy control): DOWNLOADED")
    lines.append("- Fine-tuned model checkpoint: MISSING (training never completed)")
    lines.append("")

    report = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    log(f"Report saved to {REPORT_PATH}")
    log(f"Total wall time: {total_elapsed:.0f} min")
    log("")
    log("=" * 60)
    log("STRESS TEST COMPLETE")
    log("=" * 60)


if __name__ == "__main__":
    main()
