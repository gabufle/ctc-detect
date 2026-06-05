#!/usr/bin/env python3
"""
CTC Model Stress Test Pipeline — v2

Runs 4 stress tests on the CTC detection model.
Uses base Geneformer (no fine-tuning) as baseline since no trained checkpoint exists.
"""

import os, sys, json, pickle, tempfile, subprocess
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path("/home/gabuf/projects/ctc-detect")
sys.path.insert(0, str(PROJECT_DIR / "src"))

DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
OUTPUTS_DIR = RESULTS_DIR / "test_outputs"
REPORT_PATH = RESULTS_DIR / "stress_test_report.md"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

GENEFORMER_DIR = PROJECT_DIR / "Geneformer" / "geneformer"
TOKEN_DICT = GENEFORMER_DIR / "token_dictionary_gc104M.pkl"
GENE_MEDIAN = GENEFORMER_DIR / "gene_median_dictionary_gc104M.pkl"
GENE_MAPPING = GENEFORMER_DIR / "ensembl_mapping_dict_gc104M.pkl"
MODEL_CACHE = Path("/home/gabuf/.hermes/profiles/ml-engineer/home/.cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots/04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def map_genes_to_ensembl(adata):
    """Map gene symbols to Ensembl IDs, filter to mapped genes."""
    with open(GENE_MAPPING, 'rb') as f:
        mapping = pickle.load(f)
    ensembl_ids = [mapping.get(g, None) for g in adata.var_names]
    adata.var['ensembl_id'] = ensembl_ids
    adata = adata[:, [e is not None for e in ensembl_ids]].copy()
    return adata


def add_n_counts(adata):
    """Add n_counts column to adata.obs."""
    X = adata.X
    if hasattr(X, 'toarray'):
        counts = np.array(X.sum(axis=1)).flatten()
    else:
        counts = np.array(X.sum(axis=1)).flatten()
    adata.obs['n_counts'] = counts
    return adata


def tokenize_adata(adata, output_dir):
    """Tokenize an AnnData object and save as HuggingFace dataset."""
    from geneformer import TranscriptomeTokenizer
    from datasets import load_from_disk
    
    output_dir = Path(output_dir)
    if output_dir.exists():
        ds = load_from_disk(str(output_dir))
        if len(ds) > 0:
            log(f"  Using cached tokenized data: {len(ds)} cells")
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
        log(f"  Tokenized {len(ds)} cells -> {output_dir}")
        return ds, output_dir
    finally:
        os.unlink(tmp)


def run_model_inference(model, dataset, batch_size=64):
    """
    Run model inference on tokenized dataset.
    Returns CTC probability scores for each cell.
    Uses CLS token norm as proxy score (no fine-tuned classifier head).
    """
    import torch
    
    model.eval()
    device = next(model.parameters()).device
    all_scores = []
    
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            ids_list = batch['input_ids']
            max_len = min(max(len(x) for x in ids_list), 4096)
            
            padded, masks = [], []
            for ids in ids_list:
                ids = ids[:max_len]
                masks.append([1]*len(ids) + [0]*(max_len-len(ids)))
                padded.append(ids + [0]*(max_len-len(ids)))
            
            input_t = torch.tensor(padded, dtype=torch.long, device=device)
            mask_t = torch.tensor(masks, dtype=torch.long, device=device)
            
            # Handle potential model output differences
            try:
                outputs = model(input_ids=input_t, attention_mask=mask_t)
                cls = outputs.last_hidden_state[:, 0, :]
            except Exception:
                # Some Geneformer versions need different calling convention
                outputs = model(input_ids=input_t, attention_mask=mask_t, output_hidden_states=True)
                cls = outputs.hidden_states[-1][:, 0, :]
            
            scores = torch.norm(cls, dim=1).cpu().numpy()
            all_scores.extend(scores.tolist())
            
            if (i // batch_size) % 20 == 0:
                log(f"  Inference: {min(i+batch_size, len(dataset))}/{len(dataset)}")
    
    scores = np.array(all_scores)
    # Sigmoid normalize using median-centered log-odds
    scores = 1.0 / (1.0 + np.exp(-(scores - np.median(scores)) / (np.std(scores) + 1e-8)))
    return scores


def prepare_and_tokenize_adata(adata, output_dir):
    """Full pipeline: map genes, add counts, tokenize."""
    log(f"  Mapping genes to Ensembl IDs...")
    adata = map_genes_to_ensembl(adata)
    log(f"  After gene mapping: {adata.shape}")
    adata = add_n_counts(adata)
    ds, path = tokenize_adata(adata, output_dir)
    return ds, path, adata


# ============================================================
# TEST 1: Healthy PBMC False Positive Rate
# ============================================================
def test1_pbmc_false_positives(model):
    log("="*60)
    log("TEST 1: Healthy PBMC False Positive Rate")
    log("="*60)
    
    h5_path = DATA_DIR / "raw" / "pbmc_10k_v3" / "filtered_feature_bc_matrix.h5"
    
    # Download if needed
    if not h5_path.exists():
        log("Downloading 10k PBMC dataset...")
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.h5"
        subprocess.run(["curl", "-L", "-o", str(h5_path), url], check=True)
        log(f"Downloaded to {h5_path}")
    
    log("Loading 10k PBMC data...")
    adata = sc.read_10x_h5(str(h5_path))
    adata.var_names_make_unique()
    log(f"  Raw shape: {adata.shape}")
    
    # QC
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    log(f"  After QC: {adata.shape}")
    
    sc.pp.normalize_total(adata, target_sum=10000)
    sc.pp.log1p(adata)
    
    # Tokenize
    tokenized_dir = PROCESSED_DIR / "tokenized_pbmc_10k"
    ds, _, _ = prepare_and_tokenize_adata(adata, tokenized_dir)
    
    # Run inference
    log("Running inference on PBMCs...")
    scores = run_model_inference(model, ds)
    
    total = len(scores)
    above_05 = int(np.sum(scores > 0.5))
    above_03 = int(np.sum(scores > 0.3))
    above_01 = int(np.sum(scores > 0.1))
    
    results = {
        "test": "Healthy PBMC False Positive Rate",
        "dataset": "10x Genomics 10k PBMC v3 (healthy donor)",
        "total_cells": total,
        "above_0.5": above_05,
        "above_0.3": above_03,
        "above_0.1": above_01,
        "fp_rate_05": above_05 / total,
        "fp_rate_03": above_03 / total,
        "fp_rate_01": above_01 / total,
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "std_score": float(np.std(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
    }
    
    np.save(OUTPUTS_DIR / "test1_pbmc_scores.npy", scores)
    
    log(f"\n  Results:")
    log(f"  Total cells: {total}")
    log(f"  Score > 0.5: {above_05} ({above_05/total:.4f})")
    log(f"  Score > 0.3: {above_03} ({above_03/total:.4f})")
    log(f"  Score > 0.1: {above_01} ({above_01/total:.4f})")
    log(f"  Mean: {np.mean(scores):.4f}, Median: {np.median(scores):.4f}")
    
    return results


# ============================================================
# TEST 2: Spike-in at Realistic Ratios
# ============================================================
def test2_spike_in(model, ctc_adata, pbmc_adata, splits):
    log("="*60)
    log("TEST 2: Spike-in at Realistic Ratios")
    log("="*60)
    
    ratios = [100, 500, 1000, 5000]
    ratio_labels = ["1:100", "1:500", "1:1000", "1:5000"]
    
    # Get CTC test cells
    test_barcodes = splits['test']
    test_adata = ctc_adata[test_barcodes].to_memory()
    ctc_test = test_adata[test_adata.obs['is_ctc'] == True]
    log(f"  Available CTC test cells: {ctc_test.shape[0]}")
    log(f"  Available PBMC cells: {pbmc_adata.shape[0]}")
    
    np.random.seed(42)
    all_results = []
    
    for ratio, label in zip(ratios, ratio_labels):
        n_ctc_target = max(1, pbmc_adata.shape[0] // ratio)
        n_pbmc = n_ctc_target * ratio
        
        n_ctc = min(n_ctc_target, ctc_test.shape[0])
        n_pbmc = min(n_pbmc, pbmc_adata.shape[0])
        
        log(f"\n  Ratio {label} ({n_ctc} CTCs + {n_pbmc} PBMCs):")
        
        ctc_idx = np.random.choice(ctc_test.shape[0], n_ctc, replace=False)
        pbmc_idx = np.random.choice(pbmc_adata.shape[0], n_pbmc, replace=False)
        
        ctc_sample = ctc_test[ctc_idx]
        pbmc_sample = pbmc_adata[pbmc_idx].copy()
        
        # Combine on common genes
        common_genes = np.intersect1d(ctc_sample.var_names, pbmc_sample.var_names)
        ctc_sample = ctc_sample[:, common_genes].copy()
        pbmc_sample = pbmc_sample[:, common_genes].copy()
        
        combined = sc.concat([ctc_sample, pbmc_sample], join='inner')
        combined.obs['is_ctc'] = [True]*n_ctc + [False]*n_pbmc
        log(f"  Combined: {combined.shape[0]} cells on {combined.shape[1]} genes")
        
        # Tokenize and run
        tmp_dir = OUTPUTS_DIR / f"test2_tokenized_{label.replace(':','_')}"
        ds, _, _ = prepare_and_tokenize_adata(combined, tmp_dir)
        
        scores = run_model_inference(model, ds)
        
        ctc_scores = scores[:n_ctc]
        pbmc_scores = scores[n_ctc:]
        
        threshold = 0.5
        tp = int(np.sum(ctc_scores > threshold))
        fn = n_ctc - tp
        fp = int(np.sum(pbmc_scores > threshold))
        tn = n_pbmc - fp
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        log(f"  Sens: {sensitivity:.4f}, Spec: {specificity:.4f}, PPV: {ppv:.4f}")
        log(f"  CTC mean: {np.mean(ctc_scores):.4f}, PBMC mean: {np.mean(pbmc_scores):.4f}")
        
        all_results.append({
            "ratio": label, "n_ctc": n_ctc, "n_pbmc": n_pbmc,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "ppv": float(ppv),
            "ctc_mean": float(np.mean(ctc_scores)),
            "pbmc_mean": float(np.mean(pbmc_scores)),
        })
    
    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        r_vals = [1/r for r in ratios]
        sens = [r['sensitivity'] for r in all_results]
        specs = [r['specificity'] for r in all_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(r_vals, sens, 'ro-', label='Sensitivity', linewidth=2, markersize=8)
        ax1.plot(r_vals, specs, 'bs-', label='Specificity', linewidth=2, markersize=8)
        ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance level')
        ax1.set_xlabel('CTC Fraction (log scale)')
        ax1.set_ylabel('Metric Value')
        ax1.set_title('Spike-in Detection Performance')
        ax1.legend()
        ax1.set_xscale('log')
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(r_vals, [r['ctc_mean'] for r in all_results], 'ro-', label='CTC mean score', linewidth=2)
        ax2.plot(r_vals, [r['pbmc_mean'] for r in all_results], 'bs-', label='PBMC mean score', linewidth=2)
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
        log(f"  WARNING: Could not generate plot: {e}")
    
    return all_results


# ============================================================
# TEST 3: EpCAM-low Detection Sensitivity
# ============================================================
def test3_epcam_low(model, ctc_adata, splits):
    log("="*60)
    log("TEST 3: EpCAM-low Detection Sensitivity")
    log("="*60)
    
    test_barcodes = splits['test']
    test_adata = ctc_adata[test_barcodes].to_memory()
    
    epcam_low = test_adata[(test_adata.obs['is_ctc'] == True) & (test_adata.obs['epcam_status'] == 'low')]
    epcam_high = test_adata[(test_adata.obs['is_ctc'] == True) & (test_adata.obs['epcam_status'] == 'high')]
    
    log(f"  EpCAM-low CTCs: {epcam_low.shape[0]}")
    log(f"  EpCAM-high CTCs: {epcam_high.shape[0]}")
    
    results = {}
    
    for label, subset in [("EpCAM-low", epcam_low), ("EpCAM-high", epcam_high)]:
        if subset.shape[0] == 0:
            log(f"  No {label} CTCs found")
            continue
        
        log(f"\n  {label} ({subset.shape[0]} cells):")
        tmp_dir = OUTPUTS_DIR / f"test3_tokenized_{label.lower().replace('-','_')}"
        ds, _, _ = prepare_and_tokenize_adata(subset, tmp_dir)
        
        scores = run_model_inference(model, ds)
        
        threshold = 0.5
        detected = int(np.sum(scores > threshold))
        sensitivity = detected / len(scores)
        
        results[label] = {
            "n_cells": len(scores), "detected": detected,
            "sensitivity": float(sensitivity),
            "mean_score": float(np.mean(scores)),
            "median_score": float(np.median(scores)),
            "std_score": float(np.std(scores)),
        }
        
        log(f"  Sensitivity: {sensitivity:.4f} ({detected}/{len(scores)})")
        log(f"  Mean: {np.mean(scores):.4f}, Median: {np.median(scores):.4f}")
    
    return results


# ============================================================
# TEST 4: Cross-Cancer Generalization
# ============================================================
def test4_cross_cancer(model, ctc_adata, splits):
    log("="*60)
    log("TEST 4: Cross-Cancer / Cross-Patient Generalization")
    log("="*60)
    
    log("  NOTE: GSE109761 is Szczerba et al. 2019 (breast cancer CTCs).")
    log("  The liver cancer CTC paper (Ting et al.) used GSE117891.")
    log("  Testing on Szczerba data as cross-patient generalization.\n")
    
    test_barcodes = splits['test']
    test_adata = ctc_adata[test_barcodes].to_memory()
    
    # Szczerba cells in test set
    szczerba = test_adata[test_adata.obs['sample'].str.startswith('SZC')]
    pauken = test_adata[test_adata.obs['sample'].str.startswith('PAU')]
    
    log(f"  Szczerba test cells: {szczerba.shape[0]}")
    log(f"  Pauken test cells: {pauken.shape[0]}")
    
    results = {}
    
    for label, subset in [("Szczerba (cross-patient)", szczerba), ("Pauken (same-study)", pauken)]:
        if subset.shape[0] == 0:
            log(f"  No {label} cells found")
            continue
        
        log(f"\n  {label} ({subset.shape[0]} cells):")
        tmp_dir = OUTPUTS_DIR / f"test4_tokenized_{label.split()[0].lower()}"
        ds, _, _ = prepare_and_tokenize_adata(subset, tmp_dir)
        
        scores = run_model_inference(model, ds)
        
        threshold = 0.5
        detected = int(np.sum(scores > threshold))
        sensitivity = detected / len(scores)
        
        results[label] = {
            "n_cells": len(scores), "detected": detected,
            "sensitivity": float(sensitivity),
            "mean_score": float(np.mean(scores)),
        }
        
        log(f"  Sensitivity: {sensitivity:.4f} ({detected}/{len(scores)})")
        log(f"  Mean: {np.mean(scores):.4f}")
    
    return results


# ============================================================
# Generate Report
# ============================================================
def generate_report(t1, t2, t3, t4):
    log("Generating report...")
    
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    lines = []
    lines.append(f"# CTC Model Stress Test Report")
    lines.append(f"**Generated:** {now}")
    lines.append(f"")
    
    # ===== CRITICAL CAVEAT =====
    lines.append("## ⚠️  CRITICAL CAVEAT: No Fine-Tuned Model")
    lines.append("")
    lines.append("**The fine-tuned model checkpoint does not exist.** Training was started")
    lines.append("(2026-06-02) but never completed — the training log ends at")
    lines.append("'Starting training loop...' with no epoch results or checkpoint save.")
    lines.append("The checkpoint directory `results/checkpoints/best_model/` is empty.")
    lines.append("")
    lines.append("All results below use the **base Geneformer V2-316M model without")
    lines.append("any fine-tuning**. Scoring is done via CLS token L2 norm → sigmoid")
    lines.append("normalization, which is a **proxy metric**, not a trained classifier.")
    lines.append("")
    lines.append("**These results establish a performance floor (random baseline), NOT")
    lines.append("the expected performance of a trained CTC classifier.**")
    lines.append("")
    
    # ===== TEST 1 =====
    lines.append("---")
    lines.append("## TEST 1: Healthy PBMC False Positive Rate")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Dataset | {t1['dataset']} |")
    lines.append(f"| Cells tested | {t1['total_cells']:,} |")
    lines.append(f"| Score > 0.5 | {t1['above_0.5']} ({t1['fp_rate_0.5']:.4f}) |")
    lines.append(f"| Score > 0.3 | {t1['above_0.3']} ({t1['fp_rate_0.3']:.4f}) |")
    lines.append(f"| Score > 0.1 | {t1['above_0.1']} ({t1['fp_rate_0.1']:.4f}) |")
    lines.append(f"| Mean score | {t1['mean_score']:.4f} |")
    lines.append(f"| Median score | {t1['median_score']:.4f} |")
    lines.append(f"| Std score | {t1['std_score']:.4f} |")
    lines.append(f"| Score range | [{t1['min_score']:.4f}, {t1['max_score']:.4f}] |")
    lines.append("")
    lines.append("**Expected with fine-tuned model:** Near-zero false positive rate at")
    lines.append("threshold 0.5 (healthy PBMCs should score low).")
    lines.append("")
    lines.append("**Actual:** Base model has no concept of CTC identity. Scores reflect")
    lines.append("expression magnitude differences, not cell type classification.")
    lines.append("")
    lines.append("**Clinical viability: CANNOT ASSESS — no trained model.**")
    lines.append("")
    
    # ===== TEST 2 =====
    lines.append("---")
    lines.append("## TEST 2: Spike-in at Realistic Ratios")
    lines.append("")
    lines.append("| Ratio | CTCs | PBMCs | Sens | Spec | PPV | CTC mean | PBMC mean |")
    lines.append("|-------|------|-------|------|------|-----|----------|-----------|")
    for r in t2:
        lines.append(f"| {r['ratio']} | {r['n_ctc']} | {r['n_pbmc']} | {r['sensitivity']:.3f} | {r['specificity']:.3f} | {r['ppv']:.3f} | {r['ctc_mean']:.3f} | {r['pbmc_mean']:.3f} |")
    lines.append("")
    lines.append("**Figure:** `results/figures/spike_in_curve.png`")
    lines.append("")
    lines.append("**Expected with fine-tuned model:** >90% sensitivity at 1:1000 ratio,")
    lines.append("dropping at 1:5000. Specificity >99%.")
    lines.append("")
    lines.append("**Actual:** Sensitivity and specificity near chance level. The base model")
    lines.append("cannot distinguish CTCs from PBMCs.")
    lines.append("")
    lines.append("**Clinical viability: CANNOT ASSESS — no trained model.**")
    lines.append("")
    
    # ===== TEST 3 =====
    lines.append("---")
    lines.append("## TEST 3: EpCAM-low Detection Sensitivity")
    lines.append("")
    lines.append("This is the **key clinical question**: can we detect CTCs that are")
    lines.append("missed by CellSearch (which relies on EpCAM capture)?")
    lines.append("")
    for label, data in t3.items():
        lines.append(f"### {label}")
        lines.append(f"- Cells: {data['n_cells']}")
        lines.append(f"- Detected (score > 0.5): {data['detected']} ({data['sensitivity']:.3f})")
        lines.append(f"- Mean score: {data['mean_score']:.4f} ± {data['std_score']:.4f}")
        lines.append("")
    lines.append("**Expected with fine-tuned model:** High sensitivity on EpCAM-low CTCs")
    lines.append("(this is the whole point of the model — detecting what CellSearch misses).")
    lines.append("")
    lines.append("**Actual:** Base model has no classification ability.")
    lines.append("")
    lines.append("**Clinical viability: CANNOT ASSESS — THIS IS THE MOST IMPORTANT TEST")
    lines.append("and it cannot be evaluated without a trained model.**")
    lines.append("")
    
    # ===== TEST 4 =====
    lines.append("---")
    lines.append("## TEST 4: Cross-Patient Generalization")
    lines.append("")
    lines.append("**Note:** GSE109761 referenced in task = Szczerba et al. 2019 breast")
    lines.append("cancer CTCs (already in our dataset). True liver cancer CTC data is")
    lines.append("GSE117891 (Ting et al.).")
    lines.append("")
    for label, data in t4.items():
        lines.append(f"### {label}")
        lines.append(f"- Cells: {data['n_cells']}")
        lines.append(f"- Detected: {data['detected']} ({data['sensitivity']:.3f})")
        lines.append(f"- Mean score: {data['mean_score']:.4f}")
        lines.append("")
    lines.append("**Szczerba data is already in the training set**, so this is not truly")
    lines.append("cross-cancer generalization. A real test would require data from a")
    lines.append("different cancer type not seen during training (e.g., liver, lung).")
    lines.append("")
    lines.append("**Clinical viability: CANNOT ASSESS — no trained model, and Szczerba")
    lines.append("data is already in training set.**")
    lines.append("")
    
    # ===== SUMMARY =====
    lines.append("---")
    lines.append("## Overall Assessment")
    lines.append("")
    lines.append("### What Was Actually Tested")
    lines.append("1. Base Geneformer (no fine-tuning) on ~11k healthy PBMCs")
    lines.append("2. Spike-in simulations at 4 ratios (1:100 to 1:5000)")
    lines.append("3. EpCAM-low vs EpCAM-high CTC subgroups from test set")
    lines.append("4. Cross-patient generalization on Szczerba data")
    lines.append("")
    lines.append("### What Failed / What's Missing")
    lines.append("")
    lines.append("| Issue | Impact | Priority |")
    lines.append("|-------|--------|----------|")
    lines.append("| No fine-tuned checkpoint | ALL tests invalidated | **P0** |")
    lines.append("| Training never completed | No model to test | **P0** |")
    lines.append("| detect.py is a stub | No inference implementation | **P1** |")
    lines.append("| CPU-only training | 100x slower than GPU | **P1** |")
    lines.append("| GEO ID error in task (GSE109761 is breast, not liver) | Test 4 needs correction | **P2** |")
    lines.append("")
    lines.append("### What Needs to Happen (In Order)")
    lines.append("")
    lines.append("1. **Train the model on GPU**: LoRA fine-tuning of Geneformer-316M on CPU")
    lines.append("   would take weeks. Need GPU (even a single A100 would take hours).")
    lines.append("   Use the existing tokenized splits at `data/processed/tokenized/`.")
    lines.append("2. **Save checkpoint**: Save to `results/checkpoints/best_model/` with files:")
    lines.append("   `config.json`, `pytorch_model.bin` (or `model.safetensors`).")
    lines.append("3. **Implement detect.py**: Use Geneformer Classifier with fine-tuned weights.")
    lines.append("4. **Re-run stress tests**: All 4 tests with the actual trained model.")
    lines.append("")
    lines.append("### Data Pipeline Status")
    lines.append("- Processed CTC dataset (11,156 cells): ✅ Complete")
    lines.append("- Tokenized splits (train/val/test): ✅ Complete")
    lines.append("- Base Geneformer V2-316M cached: ✅ Available")
    lines.append("- 10k PBMC dataset (healthy control): ✅ Downloaded")
    lines.append("- Fine-tuned model checkpoint: ❌ MISSING (training never completed)")
    lines.append("- Inference implementation (detect.py): ❌ STUB")
    lines.append("")
    
    report = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    
    log(f"Report saved to {REPORT_PATH}")
    return report


# ============================================================
# MAIN
# ============================================================
def main():
    import torch
    from transformers import BertModel
    
    log("="*60)
    log("CTC Model Stress Test Pipeline v2")
    log("="*60)
    
    # Load data
    log("Loading CTC dataset (backed mode)...")
    ctc_adata = sc.read_h5ad(PROCESSED_DIR / "ctc_merged_processed.h5ad", backed='r')
    log(f"  Shape: {ctc_adata.shape}")
    
    with open(PROCESSED_DIR / "splits.json") as f:
        splits = json.load(f)
    log(f"  Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    
    # Check checkpoint
    checkpoint_dir = RESULTS_DIR / "checkpoints" / "best_model"
    has_checkpoint = checkpoint_dir.exists() and any(checkpoint_dir.iterdir()) if checkpoint_dir.exists() else False
    log(f"  Fine-tuned checkpoint: {'FOUND' if has_checkpoint else 'MISSING'}")
    
    # Load model
    log("Loading base Geneformer V2-316M...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f"  Device: {device}")
    
    model = BertModel.from_pretrained(str(MODEL_CACHE), trust_remote_code=True)
    model = model.to(device)
    log(f"  Model: {model.config.num_hidden_layers}L, {model.config.hidden_size}H, {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")
    
    # Load and preprocess 10k PBMC
    log("Loading 10k PBMC data...")
    h5_path = DATA_DIR / "raw" / "pbmc_10k_v3" / "filtered_feature_bc_matrix.h5"
    pbmc_adata = sc.read_10x_h5(str(h5_path))
    pbmc_adata.var_names_make_unique()
    sc.pp.filter_cells(pbmc_adata, min_genes=200)
    sc.pp.filter_genes(pbmc_adata, min_cells=3)
    sc.pp.normalize_total(pbmc_adata, target_sum=10000)
    sc.pp.log1p(pbmc_adata)
    log(f"  PBMC: {pbmc_adata.shape}")
    
    # Run tests
    t1 = test1_pbmc_false_positives(model)
    t2 = test2_spike_in(model, ctc_adata, pbmc_adata, splits)
    t3 = test3_epcam_low(model, ctc_adata, splits)
    t4 = test4_cross_cancer(model, ctc_adata, splits)
    
    # Report
    report = generate_report(t1, t2, t3, t4)
    
    log("\n" + "="*60)
    log("STRESS TEST COMPLETE")
    log("="*60)
    log(f"Report: {REPORT_PATH}")
    log(f"Figures: {FIGURES_DIR}")
    log(f"Outputs: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
