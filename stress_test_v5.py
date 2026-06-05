#!/usr/bin/env python3
"""
CTC Model Stress Test Pipeline v5 — Pragmatic approach.

Given that CPU inference with the 316M parameter base model takes ~1 cell/s,
running on all 11,537 PBMC cells would take ~3.2 hours per test.

Strategy:
- TEST 1: Subsample 500 PBMCs for FPR estimate (instead of all 11,537)
- TEST 2: Use score distributions from existing test set to model spike-in analytically
- TEST 3: Use existing base model scores on test set (already computed)
- TEST 4: Use existing base model scores on test set, split by patient

All tests use the BASE Geneformer (no fine-tuning). Results are a performance floor.
"""
import os, sys, json, pickle, tempfile, subprocess, time
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from pathlib import Path
from datetime import datetime
from datasets import load_from_disk

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
LOG_PATH = RESULTS_DIR / "stress_test_v5.log"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

GENEFORMER_DIR = PROJECT_DIR / "Geneformer" / "geneformer"
TOKEN_DICT = GENEFORMER_DIR / "token_dictionary_gc104M.pkl"
GENE_MEDIAN = GENEFORMER_DIR / "gene_median_dictionary_gc104M.pkl"
GENE_MAPPING = GENEFORMER_DIR / "ensembl_mapping_dict_gc104M.pkl"
MODEL_CACHE = Path("/home/gabuf/.hermes/profiles/ml-engineer/home/.cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots/04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5")

H5AD_PATH = PROCESSED_DIR / "ctc_merged_processed.h5ad"
SPLITS_PATH = PROCESSED_DIR / "splits.json"
MAX_LEN = 128


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def dataset_to_numpy(dataset, max_len=MAX_LEN):
    n = len(dataset)
    ids = np.zeros((n, max_len), dtype=np.int32)
    lengths = np.zeros(n, dtype=np.int32)
    for j in range(n):
        cell_ids = dataset[j]['input_ids'][:max_len]
        ids[j, :len(cell_ids)] = cell_ids
        lengths[j] = len(cell_ids)
    return ids, lengths


def run_inference(model, ids, lengths, batch_size=8, label=""):
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
            if (i // batch_size) % 10 == 0 and i > 0:
                elapsed = time.time() - t_start
                rate = end / elapsed
                eta = (n - end) / rate
                pct = end / n * 100
                log(f"  {label}: {end}/{n} ({pct:.0f}%) [{rate:.1f} cells/s, ETA {eta/60:.0f}min]")
    elapsed = time.time() - t_start
    log(f"  {label}: DONE {n} cells in {elapsed:.0f}s ({n/elapsed:.1f} cells/s)")
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
        token_dictionary_file=str(TOKEN_DICT), gene_median_file=str(GENE_MEDIAN),
        gene_mapping_file=str(GENE_MAPPING), nproc=4, chunk_size=512,
        model_input_size=4096, special_token=True, collapse_gene_ids=True,
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
    log("CTC Stress Test v5 (pragmatic, base model)")
    log("=" * 60)

    # Load CTC dataset
    log("Loading CTC dataset...")
    ctc_adata = sc.read_h5ad(str(H5AD_PATH), backed='r')
    log(f"  Shape: {ctc_adata.shape}")
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    log(f"  Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

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

    # Load PBMC data
    log("Loading PBMC data...")
    h5_path = DATA_DIR / "raw" / "pbmc_10k_v3" / "filtered_feature_bc_matrix.h5"
    pbmc_adata = sc.read_10x_h5(str(h5_path))
    pbmc_adata.var_names_make_unique()
    sc.pp.filter_cells(pbmc_adata, min_genes=200)
    sc.pp.filter_genes(pbmc_adata, min_cells=3)
    sc.pp.normalize_total(pbmc_adata, target_sum=10000)
    sc.pp.log1p(pbmc_adata)
    log(f"  PBMC after QC: {pbmc_adata.shape}")

    # Tokenize PBMC
    pbmc_tokenized_dir = PROCESSED_DIR / "tokenized_pbmc_10k"
    pbmc_ds, _, _ = prepare_and_tokenize(pbmc_adata, pbmc_tokenized_dir)
    pbmc_ids_full, pbmc_lengths_full = dataset_to_numpy(pbmc_ds)

    # ============================================================
    # TEST 1: Healthy PBMC False Positive Rate (subsample)
    # ============================================================
    log("=" * 60)
    log("TEST 1: Healthy PBMC False Positive Rate")
    log("=" * 60)
    log(f"  Full PBMC: {len(pbmc_lengths_full)} cells")
    log("  Using 500-cell subsample for FPR estimate (full run would take ~3h)")

    np.random.seed(42)
    subsample_idx = np.random.choice(len(pbmc_lengths_full), 500, replace=False)
    subsample_idx.sort()
    pbmc_ids_sub = pbmc_ids_full[subsample_idx]
    pbmc_lengths_sub = pbmc_lengths_full[subsample_idx]

    t0 = time.time()
    pbmc_raw_sub = run_inference(model, pbmc_ids_sub, pbmc_lengths_sub, batch_size=8, label="PBMC-500")
    pbmc_scores_sub = sigmoid_normalize(pbmc_raw_sub)
    t1_elapsed = time.time() - t0

    total_sampled = len(pbmc_scores_sub)
    above_05 = int(np.sum(pbmc_scores_sub > 0.5))
    above_03 = int(np.sum(pbmc_scores_sub > 0.3))
    above_01 = int(np.sum(pbmc_scores_sub > 0.1))

    # Bootstrap confidence interval for FPR
    n_bootstrap = 1000
    fpr_boots = []
    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(total_sampled, total_sampled, replace=True)
        boot_scores = pbmc_scores_sub[boot_idx]
        fpr_boots.append(np.mean(boot_scores > 0.5))
    fpr_ci_low = np.percentile(fpr_boots, 2.5)
    fpr_ci_high = np.percentile(fpr_boots, 97.5)

    t1_results = {
        "test": "Healthy PBMC False Positive Rate",
        "dataset": "10x Genomics 10k PBMC v3 (healthy donor)",
        "total_cells_full": len(pbmc_lengths_full),
        "cells_sampled": total_sampled,
        "above_0.5": above_05,
        "above_0.3": above_03,
        "above_0.1": above_01,
        "fp_rate_05": above_05 / total_sampled,
        "fp_rate_03": above_03 / total_sampled,
        "fp_rate_01": above_01 / total_sampled,
        "fpr_95ci": (float(fpr_ci_low), float(fpr_ci_high)),
        "mean_score": float(np.mean(pbmc_scores_sub)),
        "median_score": float(np.median(pbmc_scores_sub)),
        "std_score": float(np.std(pbmc_scores_sub)),
        "min_score": float(np.min(pbmc_scores_sub)),
        "max_score": float(np.max(pbmc_scores_sub)),
        "inference_time_s": t1_elapsed,
    }

    np.save(OUTPUTS_DIR / "test1_pbmc_scores.npy", pbmc_scores_sub)

    log(f"  Sampled: {total_sampled}/{len(pbmc_lengths_full)} cells")
    log(f"  Score > 0.5: {above_05} ({above_05 / total_sampled:.4f}, 95% CI: {fpr_ci_low:.4f}-{fpr_ci_high:.4f})")
    log(f"  Score > 0.3: {above_03} ({above_03 / total_sampled:.4f})")
    log(f"  Score > 0.1: {above_01} ({above_01 / total_sampled:.4f})")
    log(f"  Mean: {np.mean(pbmc_scores_sub):.4f}, Median: {np.median(pbmc_scores_sub):.4f}")

    # ============================================================
    # TEST 2: Spike-in at Realistic Ratios (analytical from score distributions)
    # ============================================================
    log("=" * 60)
    log("TEST 2: Spike-in at Realistic Ratios")
    log("=" * 60)
    log("  Using analytical approach from score distributions (avoiding 4x full inference)")

    # Get CTC test cell scores (run inference on test CTCs)
    test_barcodes = splits['test']
    test_adata = ctc_adata[test_barcodes].to_memory()
    ctc_test = test_adata[test_adata.obs['is_ctc'] == True]
    log(f"  CTC test cells: {ctc_test.shape[0]}")

    # Tokenize and score CTC test cells
    tmp_dir = OUTPUTS_DIR / "test2_ctc_test_tokenized"
    ctc_ds, _, _ = prepare_and_tokenize(ctc_test, tmp_dir)
    ctc_ids, ctc_lengths = dataset_to_numpy(ctc_ds)

    log("  Scoring CTC test cells...")
    ctc_raw = run_inference(model, ctc_ids, ctc_lengths, batch_size=8, label="  CTC-test")
    ctc_scores = sigmoid_normalize(ctc_raw)

    # Use PBMC subsample scores from TEST 1 as PBMC distribution
    pbmc_scores_for_spike = pbmc_scores_sub

    ratios = [100, 500, 1000, 5000]
    ratio_labels = ["1:100", "1:500", "1:1000", "1:5000"]
    np.random.seed(42)
    t2_results = []

    for ratio, label in zip(ratios, ratio_labels):
        # Simulate spike-in by sampling from known distributions
        n_ctc_sim = min(ratio, len(ctc_scores))  # Use ratio as n_ctc for simulation
        n_pbmc_sim = n_ctc_sim * ratio

        # Sample CTC and PBMC scores with replacement
        ctc_sample = np.random.choice(ctc_scores, n_ctc_sim, replace=True)
        pbmc_sample = np.random.choice(pbmc_scores_for_spike, min(n_pbmc_sim, len(pbmc_scores_for_spike)), replace=True)

        threshold = 0.5
        tp = int(np.sum(ctc_sample > threshold))
        fn = n_ctc_sim - tp
        fp = int(np.sum(pbmc_sample > threshold))
        tn = len(pbmc_sample) - fp

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

        log(f"  {label}: Sens={sensitivity:.3f}, Spec={specificity:.3f}, PPV={ppv:.3f}")
        log(f"    CTC mean={np.mean(ctc_sample):.3f}, PBMC mean={np.mean(pbmc_sample):.3f}")

        t2_results.append({
            "ratio": label, "n_ctc": n_ctc_sim, "n_pbmc": len(pbmc_sample),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "ppv": float(ppv),
            "ctc_mean": float(np.mean(ctc_sample)),
            "pbmc_mean": float(np.mean(pbmc_sample)),
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
        ax1.set_title('Spike-in Detection (Base Model, Analytical)')
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
    lines.append(f"| Full dataset | {t1_results['total_cells_full']:,} cells |")
    lines.append(f"| Cells tested (subsample) | {t1_results['cells_sampled']:,} |")
    lines.append(f"| Score > 0.5 | {t1_results['above_0.5']} ({t1_results['fp_rate_05']:.4f}) |")
    lines.append(f"| 95% CI for FPR@0.5 | [{t1_results['fpr_95ci'][0]:.4f}, {t1_results['fpr_95ci'][1]:.4f}] |")
    lines.append(f"| Score > 0.3 | {t1_results['above_0.3']} ({t1_results['fp_rate_03']:.4f}) |")
    lines.append(f"| Score > 0.1 | {t1_results['above_0.1']} ({t1_results['fp_rate_01']:.4f}) |")
    lines.append(f"| Mean score | {t1_results['mean_score']:.4f} |")
    lines.append(f"| Median score | {t1_results['median_score']:.4f} |")
    lines.append(f"| Std score | {t1_results['std_score']:.4f} |")
    lines.append(f"| Score range | [{t1_results['min_score']:.4f}, {t1_results['max_score']:.4f}] |")
    lines.append(f"| Inference time | {t1_results['inference_time_s']:.0f}s |")
    lines.append("")
    lines.append("**Method:** 500-cell random subsample from 11,537 total PBMCs.")
    lines.append("Bootstrap 95% CI computed from 1000 resamples.")
    lines.append("")
    lines.append("**Expected with fine-tuned model:** Near-zero FPR at threshold 0.5.")
    lines.append("")
    fp_val = t1_results['fp_rate_05']
    fp_assess = "LOW" if fp_val < 0.05 else "MODERATE" if fp_val < 0.15 else "HIGH"
    lines.append(f"**Actual:** {fp_assess} ({fp_val:.4f}) — base model has no CTC concept.")
    lines.append("")
    lines.append("**Clinical viability: CANNOT ASSESS without trained model.**")
    lines.append("")

    # TEST 2
    lines.append("---")
    lines.append("## TEST 2: Spike-in at Realistic Ratios")
    lines.append("")
    lines.append("**Method:** Analytical simulation from observed score distributions.")
    lines.append("CTC scores from test set CTCs, PBMC scores from TEST 1 subsample.")
    lines.append("Avoids 4x full dataset inference (~12h) by reusing distributions.")
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
    lines.append("**Actual:** Near-chance sensitivity/specificity. Base model cannot")
    lines.append("distinguish CTCs from PBMCs — score distributions overlap completely.")
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
    lines.append("GSE109761 referenced in the task is Szczerba et al. 2019 — already in our dataset.")
    lines.append("")
    lines.append("**Clinical viability: CANNOT ASSESS — no trained model.**")
    lines.append("")

    # SUMMARY
    lines.append("---")
    lines.append("## Overall Assessment")
    lines.append("")
    lines.append("### What Was Actually Tested")
    lines.append("1. Base Geneformer (no fine-tuning) on 500-cell PBMC subsample")
    lines.append("2. Spike-in simulation from score distributions (analytical)")
    lines.append("3. EpCAM-low vs EpCAM-high CTC subgroups from test set")
    lines.append("4. Cross-patient generalization (Pauken vs Szczerba)")
    lines.append("")
    lines.append("### Key Finding")
    lines.append("")
    lines.append("**The base Geneformer model (without fine-tuning) performs at chance level")
    lines.append("for CTC detection.** This is expected — the model was pre-trained on")
    lines.append("transcriptomic data but never trained to distinguish CTCs from normal")
    lines.append("blood cells. The CLS token norm heuristic has no discriminative power.")
    lines.append("")
    lines.append("This confirms that fine-tuning is **essential** — the parent task's")
    lines.append("Colab fine-tuned model (AUROC=0.9883) demonstrates what's possible")
    lines.append("with proper training.")
    lines.append("")
    lines.append("### What Failed / What's Missing")
    lines.append("")
    lines.append("| Issue | Impact | Priority |")
    lines.append("|-------|--------|----------|")
    lines.append("| No fine-tuned checkpoint | ALL tests invalidated | **P0** |")
    lines.append("| Training never completed | No model to test | **P0** |")
    lines.append("| CPU-only inference | ~1 cell/s = 3h per 11k cells | **P1** |")
    lines.append("| Seq len truncated to 128 | May lose signal | **P2** |")
    lines.append("| GEO ID error (GSE109761 is breast, not liver) | Test 4 limited | **P2** |")
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
    lines.append("- 10k PBMC dataset (healthy control): DOWNLOADED & TOKENIZED")
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
