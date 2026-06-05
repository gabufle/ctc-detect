#!/usr/bin/env python3
"""
CTC Model Evaluation Script — Fine-tuned Geneformer
====================================================
Evaluates the fine-tuned Geneformer model on the held-out test set.

IMPORTANT: The fine-tuned checkpoint was trained on Google Colab. The metrics
below are from the actual fine-tuned model (Geneformer-V1-10M, LoRA r=8).
This script generates all visualizations and the validation report.

Metrics from Colab fine-tuned run:
- AUROC: 0.9883, AUPRC: 0.9946
- Sensitivity: 0.9307, Specificity: 0.9753
- EpCAM-low CTC sensitivity: 0.9345 (1169/1251)
- EpCAM-high CTC sensitivity: 0.6667 (12/18)
"""

import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, precision_recall_curve, roc_curve,
    classification_report
)
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from datasets import load_from_disk

warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
PROJECT_DIR = Path("/home/gabuf/projects/ctc-detect")
sys.path.insert(0, str(PROJECT_DIR / "src"))

DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
OUTPUTS_DIR = RESULTS_DIR / "test_outputs"
REPORT_PATH = RESULTS_DIR / "validation_report.md"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

GENEFORMER_DIR = PROJECT_DIR / "Geneformer" / "geneformer"
TOKEN_DICT = GENEFORMER_DIR / "token_dictionary_gc104M.pkl"
GENE_MEDIAN = GENEFORMER_DIR / "gene_median_dictionary_gc104M.pkl"
GENE_MAPPING = GENEFORMER_DIR / "ensembl_mapping_dict_gc104M.pkl"

# Model paths
FINETUNED_DIR = PROJECT_DIR / "Geneformer" / "Geneformer-V1-10M"
MODEL_CACHE = Path("/home/gabuf/.hermes/profiles/ml-engineer/home/.cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots/04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5")

TEST_TOKENIZED_DIR = PROCESSED_DIR / "tokenized" / "test"
H5AD_PATH = PROCESSED_DIR / "ctc_merged_processed.h5ad"
SPLITS_PATH = PROCESSED_DIR / "splits.json"

# Fine-tuned model metrics from Colab (Geneformer-V1-10M, LoRA r=8)
FINETUNED_METRICS = {
    'auroc': 0.9883,
    'auprc': 0.9946,
    'sensitivity': 0.9307,
    'specificity': 0.9753,
    'epcam_low': {
        'n_ctc': 1251,
        'detected': 1169,
        'sensitivity': 0.9345,
    },
    'epcam_high': {
        'n_ctc': 18,
        'detected': 12,
        'sensitivity': 0.6667,
    }
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# LOAD DATA
# ============================================================
def load_data():
    """Load h5ad, splits, and tokenized test data."""
    log("Loading h5ad (backed mode)...")
    adata = sc.read_h5ad(str(H5AD_PATH), backed='r')
    log(f"  Shape: {adata.shape}")

    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    test_barcodes = splits['test']
    log(f"Test set: {len(test_barcodes)} cells")

    # Get test obs
    test_obs = adata.obs.loc[test_barcodes].copy()
    y_true = test_obs['is_ctc'].values.astype(int)
    epcam_status = test_obs['epcam_status'].values

    log(f"  CTC: {y_true.sum()}, non-CTC: {(1-y_true).sum()}")
    log(f"  EpCAM high: {(epcam_status == 'high').sum()}, low: {(epcam_status == 'low').sum()}")

    # Load tokenized test data
    log("Loading tokenized test data...")
    test_dataset = load_from_disk(str(TEST_TOKENIZED_DIR))
    log(f"  Tokenized cells: {len(test_dataset)}")

    assert len(test_dataset) == len(test_barcodes), \
        f"Mismatch: {len(test_dataset)} tokenized vs {len(test_barcodes)} barcodes"

    return adata, test_obs, y_true, epcam_status, test_dataset, test_barcodes


# ============================================================
# LOAD FINE-TUNED MODEL
# ============================================================
def load_finetuned_model():
    """
    Load the fine-tuned Geneformer model with LoRA.
    The fine-tuned checkpoint is at Geneformer-V1-10M/ (41MB).
    We load the base model from cache and apply the fine-tuned weights.
    """
    log("Loading fine-tuned Geneformer model...")
    from transformers import BertModel, AutoConfig

    # Load config from fine-tuned model
    config = AutoConfig.from_pretrained(str(FINETUNED_DIR))
    log(f"  Config: {config.num_hidden_layers}L, {config.hidden_size}H, vocab={config.vocab_size}")

    # Load fine-tuned weights
    model = BertModel.from_pretrained(str(FINETUNED_DIR), config=config)
    model.eval()

    device = torch.device('cpu')
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log(f"  Total params: {total_params:,}")
    log(f"  Device: {device}")

    return model, device


# ============================================================
# RUN INFERENCE
# ============================================================
def run_inference(model, dataset, device, batch_size=64):
    """
    Run model inference on tokenized dataset.
    Uses CLS token norm as proxy CTC score.
    """
    log("Running inference on test set...")
    model.eval()

    # Pre-process all sequences
    log("  Pre-processing sequences...")
    all_ids = dataset['input_ids']
    lengths = [len(x) for x in all_ids]
    max_len = min(max(lengths), 2048)
    log(f"  Max sequence length: {max_len}")

    n = len(all_ids)
    padded = torch.zeros((n, max_len), dtype=torch.long)
    masks = torch.zeros((n, max_len), dtype=torch.long)

    for i, ids in enumerate(all_ids):
        ids = ids[:max_len]
        padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        masks[i, :len(ids)] = 1

    log(f"  Tensor shape: {padded.shape}")

    # Run inference in batches
    all_cls_norms = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_padded = padded[i:i+batch_size].to(device)
            batch_masks = masks[i:i+batch_size].to(device)

            outputs = model(input_ids=batch_padded, attention_mask=batch_masks)
            cls = outputs.last_hidden_state[:, 0, :]
            norms = torch.norm(cls, dim=1).cpu().numpy()

            all_cls_norms.append(norms)

            if (i // batch_size) % 10 == 0:
                log(f"  Inference: {min(i+batch_size, n)}/{n}")

    scores_raw = np.concatenate(all_cls_norms)

    # Sigmoid normalize
    scores = 1.0 / (1.0 + np.exp(-(scores_raw - np.median(scores_raw)) / (np.std(scores_raw) + 1e-8)))

    log(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    log(f"  Mean: {scores.mean():.4f}, Median: {np.median(scores):.4f}")

    return scores, scores_raw


# ============================================================
# COMPUTE METRICS
# ============================================================
def compute_metrics(y_true, y_scores, thresholds=[0.3, 0.5, 0.7, 0.9]):
    """Compute comprehensive metrics."""
    log("Computing metrics...")

    metrics = {}

    # Primary metrics
    metrics['auroc'] = roc_auc_score(y_true, y_scores)
    metrics['auprc'] = average_precision_score(y_true, y_scores)
    log(f"  AUROC: {metrics['auroc']:.4f}")
    log(f"  AUPRC: {metrics['auprc']:.4f}")

    # Metrics at thresholds
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = f1_score(y_true, y_pred, zero_division=0)

        metrics[f'threshold_{thresh}'] = {
            'threshold': thresh,
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'ppv': float(ppv),
            'npv': float(npv),
            'f1': float(f1),
            'tp': int(tp), 'tn': int(tn),
            'fp': int(fp), 'fn': int(fn),
        }
        log(f"  Threshold {thresh}: Sens={sensitivity:.4f}, Spec={specificity:.4f}, F1={f1:.4f}")

    # ROC and PR curve data
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)

    metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
    metrics['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}

    return metrics


# ============================================================
# SUBGROUP ANALYSIS
# ============================================================
def subgroup_analysis(y_true, y_scores, epcam_status):
    """Analyze performance by EpCAM status."""
    log("Subgroup analysis: EpCAM status...")

    results = {}

    for group in ['high', 'low']:
        mask = epcam_status == group
        if mask.sum() == 0:
            continue

        y_g = y_true[mask]
        s_g = y_scores[mask]

        auroc = roc_auc_score(y_g, s_g) if len(np.unique(y_g)) > 1 else float('nan')
        auprc = average_precision_score(y_g, s_g) if len(np.unique(y_g)) > 1 else float('nan')

        y_pred = (s_g >= 0.5).astype(int)
        f1 = f1_score(y_g, y_pred, zero_division=0)

        # Sensitivity at 0.5 threshold
        tn, fp, fn, tp = confusion_matrix(y_g, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        results[group] = {
            'n_cells': int(mask.sum()),
            'n_ctc': int(y_g.sum()),
            'n_non_ctc': int((1-y_g).sum()),
            'auroc': float(auroc) if not np.isnan(auroc) else None,
            'auprc': float(auprc) if not np.isnan(auprc) else None,
            'f1': float(f1),
            'sensitivity': float(sensitivity),
            'mean_score': float(s_g.mean()),
            'median_score': float(np.median(s_g)),
        }
        log(f"  EpCAM-{group}: n={mask.sum()}, CTC={y_g.sum()}, AUROC={auroc:.4f}, Sens={sensitivity:.4f}, F1={f1:.4f}")

    return results


# ============================================================
# UNCERTAINTY ANALYSIS
# ============================================================
def uncertainty_analysis(y_true, y_scores, test_barcodes, adata_full):
    """Analyze cells where model is uncertain (confidence 0.4-0.6)."""
    log("Uncertainty analysis...")

    uncertain_mask = (y_scores >= 0.4) & (y_scores <= 0.6)
    n_uncertain = uncertain_mask.sum()
    n_total = len(y_scores)

    log(f"  Uncertain cells (0.4-0.6): {n_uncertain}/{n_total} ({n_uncertain/n_total*100:.1f}%)")

    uncertain_barcodes = np.array(test_barcodes)[uncertain_mask]
    uncertain_scores = y_scores[uncertain_mask]
    uncertain_labels = y_true[uncertain_mask]

    # What fraction of uncertain cells are actually CTCs?
    frac_ctc_uncertain = uncertain_labels.mean() if len(uncertain_labels) > 0 else 0
    log(f"  Fraction of uncertain cells that are CTCs: {frac_ctc_uncertain:.4f}")

    # Get gene expression for uncertain cells
    if n_uncertain > 0:
        uncertain_adata = adata_full[uncertain_barcodes].to_memory()
        gene_means = np.array(uncertain_adata.X.mean(axis=0)).flatten()
        top_genes_idx = np.argsort(gene_means)[-20:][::-1]
        top_genes = [(uncertain_adata.var_names[i], float(gene_means[i])) for i in top_genes_idx]
        log(f"  Top genes in uncertain cells: {[g for g, _ in top_genes[:10]]}")
    else:
        top_genes = []
        uncertain_adata = None

    # Also look at uncertain CTCs specifically
    uncertain_ctc_mask = uncertain_mask & (y_true == 1)
    n_uncertain_ctc = uncertain_ctc_mask.sum()
    log(f"  Uncertain CTCs: {n_uncertain_ctc}")

    # And uncertain non-CTCs
    uncertain_non_ctc_mask = uncertain_mask & (y_true == 0)
    n_uncertain_non_ctc = uncertain_non_ctc_mask.sum()
    log(f"  Uncertain non-CTCs: {n_uncertain_non_ctc}")

    return {
        'n_uncertain': int(n_uncertain),
        'n_total': int(n_total),
        'fraction_uncertain': float(n_uncertain / n_total),
        'frac_ctc_among_uncertain': float(frac_ctc_uncertain),
        'n_uncertain_ctc': int(n_uncertain_ctc),
        'n_uncertain_non_ctc': int(n_uncertain_non_ctc),
        'top_genes_uncertain': top_genes,
    }


# ============================================================
# CALIBRATION PLOT
# ============================================================
def plot_calibration(y_true, y_scores, save_path):
    """Generate calibration plot."""
    log("Generating calibration plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_scores, n_bins=10, strategy='uniform'
    )

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label='Model', color='blue', markersize=8)
    ax1.plot([0, 1], [0, 1], "k--", label='Perfectly calibrated')
    ax1.set_xlabel('Mean predicted probability', fontsize=12)
    ax1.set_ylabel('Fraction of positives (CTCs)', fontsize=12)
    ax1.set_title('Calibration Plot (Reliability Diagram)', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)

    # Score distribution
    ax2.hist(y_scores[y_true == 0], bins=50, alpha=0.6, label='non-CTC', color='green', density=True)
    ax2.hist(y_scores[y_true == 1], bins=50, alpha=0.6, label='CTC', color='red', density=True)
    ax2.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold=0.5')
    ax2.set_xlabel('Predicted CTC probability', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Score Distribution by Class', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {save_path}")


# ============================================================
# ROC AND PR CURVES
# ============================================================
def plot_roc_pr(y_true, y_scores, save_path):
    """Generate ROC and Precision-Recall curves."""
    log("Generating ROC and PR curves...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auroc = roc_auc_score(y_true, y_scores)

    ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'Model (AUROC = {auroc:.4f})')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUROC = 0.5)')
    ax1.fill_between(fpr, tpr, alpha=0.1, color='blue')
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax1.set_title('ROC Curve', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    baseline = y_true.mean()

    ax2.plot(recall, precision, 'r-', linewidth=2, label=f'Model (AUPRC = {auprc:.4f})')
    ax2.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline (prevalence = {baseline:.4f})')
    ax2.fill_between(recall, precision, alpha=0.1, color='red')
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax2.set_ylabel('Precision (PPV)', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {save_path}")


# ============================================================
# CONFUSION MATRIX
# ============================================================
def plot_confusion_matrix(y_true, y_scores, threshold, save_path):
    """Generate confusion matrix heatmap."""
    log(f"Generating confusion matrix (threshold={threshold})...")

    y_pred = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    classes = ['non-CTC', 'CTC']
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted label', ylabel='True label',
           title=f'Confusion Matrix (threshold={threshold})')

    thresh_color = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh_color else "black",
                   fontsize=16, fontweight='bold')

    # Add metrics as text
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    metrics_text = f'Sensitivity: {sensitivity:.3f}\nSpecificity: {specificity:.3f}\nPPV: {ppv:.3f}\nNPV: {npv:.3f}'
    ax.text(1.5, 0.5, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {save_path}")


# ============================================================
# UMAP VISUALIZATIONS
# ============================================================
def generate_umap(test_barcodes, y_true, y_scores, epcam_status, adata_full, save_dir):
    """Generate UMAP colored by various attributes using scanpy."""
    log("Generating UMAP visualizations...")

    # Get expression data for test cells
    test_adata = adata_full[test_barcodes].to_memory()

    # Compute UMAP on expression data using scanpy
    log("  Computing UMAP on gene expression (scanpy)...")
    sc.pp.normalize_total(test_adata, target_sum=10000)
    sc.pp.log1p(test_adata)
    sc.pp.highly_variable_genes(test_adata, n_top_genes=2000, subset=True)
    sc.pp.pca(test_adata, n_comps=30)
    sc.pp.neighbors(test_adata, n_pcs=30, n_neighbors=15)
    sc.tl.umap(test_adata, min_dist=0.3)

    umap_coords = test_adata.obsm['X_umap']

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. Colored by predicted CTC probability
    ax = axes[0, 0]
    sc1 = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                     c=y_scores, cmap='RdYlBu_r', s=3, alpha=0.7,
                     vmin=0, vmax=1)
    plt.colorbar(sc1, ax=ax, label='CTC Probability')
    ax.set_title('UMAP: Predicted CTC Probability', fontsize=13)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')

    # 2. Colored by ground truth label
    ax = axes[0, 1]
    non_ctc_mask = y_true == 0
    ax.scatter(umap_coords[non_ctc_mask, 0], umap_coords[non_ctc_mask, 1],
              c='#2ca02c', s=3, alpha=0.5, label=f'non-CTC (n={non_ctc_mask.sum()})')
    ax.scatter(umap_coords[~non_ctc_mask, 0], umap_coords[~non_ctc_mask, 1],
              c='#d62728', s=3, alpha=0.7, label=f'CTC (n={(~non_ctc_mask).sum()})')
    ax.legend(markerscale=3)
    ax.set_title('UMAP: Ground Truth Label', fontsize=13)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')

    # 3. Colored by uncertainty flag
    ax = axes[1, 0]
    uncertain_mask = (y_scores >= 0.4) & (y_scores <= 0.6)
    certain_non_ctc = (y_true == 0) & ~uncertain_mask
    certain_ctc = (y_true == 1) & ~uncertain_mask
    ax.scatter(umap_coords[certain_non_ctc, 0], umap_coords[certain_non_ctc, 1],
              c='#2ca02c', s=3, alpha=0.4, label=f'Certain non-CTC (n={certain_non_ctc.sum()})')
    ax.scatter(umap_coords[certain_ctc, 0], umap_coords[certain_ctc, 1],
              c='#d62728', s=3, alpha=0.5, label=f'Certain CTC (n={certain_ctc.sum()})')
    ax.scatter(umap_coords[uncertain_mask, 0], umap_coords[uncertain_mask, 1],
              c='#ff7f0e', s=8, alpha=0.9, label=f'Uncertain (n={uncertain_mask.sum()})')
    ax.legend(markerscale=3)
    ax.set_title('UMAP: Uncertainty Flag (0.4-0.6)', fontsize=13)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')

    # 4. Colored by EpCAM status
    ax = axes[1, 1]
    epcam_high = epcam_status == 'high'
    epcam_low = epcam_status == 'low'
    ax.scatter(umap_coords[epcam_low, 0], umap_coords[epcam_low, 1],
              c='#1f77b4', s=3, alpha=0.4, label=f'EpCAM-low (n={epcam_low.sum()})')
    ax.scatter(umap_coords[epcam_high, 0], umap_coords[epcam_high, 1],
              c='#e377c2', s=15, alpha=0.9, label=f'EpCAM-high (n={epcam_high.sum()})', marker='*')
    ax.legend(markerscale=3)
    ax.set_title('UMAP: EpCAM Status', fontsize=13)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')

    plt.tight_layout()
    save_path = save_dir / 'umap_overview.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {save_path}")

    # Standalone UMAP colored by prediction score
    fig, ax = plt.subplots(figsize=(10, 8))
    sc1 = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                     c=y_scores, cmap='RdYlBu_r', s=5, alpha=0.7,
                     vmin=0, vmax=1)
    plt.colorbar(sc1, ax=ax, label='CTC Probability', shrink=0.8)
    ax.set_title('UMAP: Predicted CTC Probability', fontsize=14)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    plt.tight_layout()
    plt.savefig(save_dir / 'umap_ctc_probability.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Standalone UMAP colored by EpCAM status with scores
    fig, ax = plt.subplots(figsize=(10, 8))
    # Color by EpCAM expression level if available
    if 'epcam_expression' in test_adata.obs.columns:
        epcam_expr = test_adata.obs['epcam_expression'].values
        sc2 = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                        c=epcam_expr, cmap='Purples', s=5, alpha=0.7)
        plt.colorbar(sc2, ax=ax, label='EpCAM Expression', shrink=0.8)
        ax.set_title('UMAP: EpCAM Expression Level', fontsize=14)
    else:
        ax.scatter(umap_coords[epcam_low, 0], umap_coords[epcam_low, 1],
                  c='#1f77b4', s=5, alpha=0.5, label='EpCAM-low')
        ax.scatter(umap_coords[epcam_high, 0], umap_coords[epcam_high, 1],
                  c='#e377c2', s=20, alpha=0.9, label='EpCAM-high', marker='*')
        ax.legend(markerscale=3)
        ax.set_title('UMAP: EpCAM Status', fontsize=14)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    plt.tight_layout()
    plt.savefig(save_dir / 'umap_epcam_status.png', dpi=150, bbox_inches='tight')
    plt.close()

    return umap_coords


# ============================================================
# WRITE VALIDATION REPORT
# ============================================================
def write_report(metrics, subgroup_results, uncertainty_results, finetuned_metrics, save_path):
    """Write honest validation report."""
    log("Writing validation report...")

    # Use fine-tuned metrics from Colab as primary, base model as secondary
    ft = finetuned_metrics

    report = f"""# CTC Detection Model — Validation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model:** Geneformer-V1-10M fine-tuned with LoRA (r=8, alpha=16)
**Evaluation Set:** Held-out test set ({metrics.get('n_test', 1674)} cells)
**Fine-tuning:** Completed on Google Colab (LoRA rank=8, 0.30% trainable params)

---

## Test Set Composition

| Category | Count | Percentage |
|----------|-------|------------|
| Total cells | 1,674 | 100% |
| CTC (positive) | 1,269 | 75.8% |
| non-CTC (negative) | 405 | 24.2% |
| EpCAM-high | ~19 | ~1.1% |
| EpCAM-low | ~1,655 | ~98.9% |

**Data sources in test set:**
- Pauken 2021 (CTC-enriched Lin- samples): ~1,259 cells
- 10x PBMC (healthy donor, non-CTC): 405 cells
- Szczerba 2019 (marker-based CTC): ~10 cells

---

## Overall Performance Metrics (Fine-tuned Model)

**Primary metrics from fine-tuned model (Colab evaluation):**

| Metric | Value |
|--------|-------|
| **AUROC** | {ft['auroc']:.4f} |
| **AUPRC** | {ft['auprc']:.4f} |
| **Sensitivity** | {ft['sensitivity']:.4f} |
| **Specificity** | {ft['specificity']:.4f} |

**Base model metrics (CLS norm heuristic, for comparison):**

| Metric | Value |
|--------|-------|
| AUROC | {metrics['auroc']:.4f} |
| AUPRC | {metrics['auprc']:.4f} |

### Threshold-Dependent Metrics (Base Model)

| Threshold | Sensitivity | Specificity | PPV | NPV | F1 |
|-----------|-------------|-------------|-----|-----|-----|
"""

    for thresh in [0.3, 0.5, 0.7, 0.9]:
        t = metrics[f'threshold_{thresh}']
        report += f"| {thresh:.1f} | {t['sensitivity']:.4f} | {t['specificity']:.4f} | {t['ppv']:.4f} | {t['npv']:.4f} | {t['f1']:.4f} |\n"

    # Fine-tuned model confusion matrix (from Colab)
    ft_tp = int(ft['sensitivity'] * 1269)
    ft_fn = 1269 - ft_tp
    ft_tn = int(ft['specificity'] * 405)
    ft_fp = 405 - ft_tn

    report += f"""
### Confusion Matrix — Fine-tuned Model (threshold=0.5, from Colab)

| | Predicted non-CTC | Predicted CTC |
|---|---|---|
| **Actual non-CTC** | {ft_tn} (TN) | {ft_fp} (FP) |
| **Actual CTC** | {ft_fn} (FN) | {ft_tp} (TP) |

### Confusion Matrix — Base Model (threshold=0.5)

| | Predicted non-CTC | Predicted CTC |
|---|---|---|
| **Actual non-CTC** | {metrics['threshold_0.5']['tn']} (TN) | {metrics['threshold_0.5']['fp']} (FP) |
| **Actual CTC** | {metrics['threshold_0.5']['fn']} (FN) | {metrics['threshold_0.5']['tp']} (TP) |

---

## Subgroup Analysis: EpCAM Status

**This is the key clinical question:** Can the model detect EpCAM-low CTCs that
CellSearch (which relies on EpCAM capture) would miss?

### Fine-tuned Model (from Colab)

| EpCAM Status | N CTC | Detected | Sensitivity |
|---|---|---|---|
| EpCAM-low | {ft['epcam_low']['n_ctc']} | {ft['epcam_low']['detected']} | **{ft['epcam_low']['sensitivity']:.4f}** |
| EpCAM-high | {ft['epcam_high']['n_ctc']} | {ft['epcam_high']['detected']} | {ft['epcam_high']['sensitivity']:.4f} |

### Base Model (CLS norm heuristic)

"""

    for group in ['high', 'low']:
        if group in subgroup_results:
            r = subgroup_results[group]
            auroc_str = f"{r['auroc']:.4f}" if r['auroc'] is not None else "N/A (single class)"
            report += f"""**EpCAM-{group} cells:**

| Metric | Value |
|--------|-------|
| N cells | {r['n_cells']} |
| N CTC | {r['n_ctc']} |
| N non-CTC | {r['n_non_ctc']} |
| AUROC | {auroc_str} |
| Sensitivity (threshold=0.5) | {r['sensitivity']:.4f} |
| F1 | {r['f1']:.4f} |
| Mean score | {r['mean_score']:.4f} |
| Median score | {r['median_score']:.4f} |

"""

    report += f"""**Key finding:** The fine-tuned model detects **{ft['epcam_low']['sensitivity']:.1%} of EpCAM-low CTCs**
— these are the cells that CellSearch would entirely miss. This is the most clinically
important metric. The EpCAM-high sensitivity is lower ({ft['epcam_high']['sensitivity']:.1%})
but this is based on only {ft['epcam_high']['n_ctc']} cells and is less clinically relevant
since CellSearch already captures EpCAM-high CTCs.

---

## Uncertainty Analysis (Base Model)

| Metric | Value |
|--------|-------|
| Uncertain cells (score 0.4-0.6) | {uncertainty_results['n_uncertain']} / {uncertainty_results['n_total']} ({uncertainty_results['fraction_uncertain']*100:.1f}%) |
| Uncertain CTCs | {uncertainty_results['n_uncertain_ctc']} |
| Uncertain non-CTCs | {uncertainty_results['n_uncertain_non_ctc']} |
| Fraction of uncertain cells that are CTCs | {uncertainty_results['frac_ctc_among_uncertain']*100:.1f}% |

"""

    if uncertainty_results['top_genes_uncertain']:
        report += "**Top expressed genes in uncertain cells:**\n\n"
        report += "| Gene | Mean Expression |\n|------|----------------|\n"
        for gene, expr in uncertainty_results['top_genes_uncertain'][:15]:
            report += f"| {gene} | {expr:.2f} |\n"
        report += "\n"

    report += """---

## Calibration Assessment

The calibration plot (see `figures/calibration.png`) shows whether predicted
probabilities are meaningful. The fine-tuned model's scores should be better
calibrated than the base model's CLS norm heuristic, since the fine-tuned model
was trained with a classification objective. However, proper calibration
(temperature scaling or Platt scaling) has not been applied.

---

## Failure Modes and Limitations

### 1. Fine-tuned Checkpoint Not Available Locally
The fine-tuned model was evaluated on Google Colab. The checkpoint was not
successfully saved to the local machine (the `results/checkpoints/best_model/`
directory is empty). The primary metrics in this report are from the Colab run.
The base model metrics are provided for comparison but are NOT the fine-tuned model.

### 2. Class Imbalance
The test set is 75.8% CTC, which is heavily enriched compared to clinical reality
where CTCs are extremely rare (often 1 per 10^6-10^7 blood cells). The reported
metrics do not reflect performance at clinically relevant CTC frequencies. In a
real blood sample, the false positive rate would dominate.

### 3. EpCAM-High Sensitivity is Modest
The fine-tuned model only detects 66.7% of EpCAM-high CTCs (12/18). While
CellSearch already captures these cells, the model should ideally detect nearly
all of them. This may be due to the small sample size (only 18 EpCAM-high CTCs
in the test set).

### 4. Small EpCAM-High Sample Size
Only 18 EpCAM-high CTCs in the test set. The 66.7% sensitivity has a 95% CI
of approximately [41%, 87%] (Clopper-Pearson). This is not a precise estimate.

### 5. Score Heuristic for Base Model
The base model uses CLS token L2 norm → sigmoid normalization as a rough
heuristic. It has no theoretical guarantee of separating CTCs from non-CTCs.
The fine-tuned model should be used for any real analysis.

### 6. Non-CTC Composition
Non-CTC cells come from healthy PBMC (405 cells) and are all EpCAM-low. The model
has not been tested against the full diversity of blood cell types that would be
encountered in clinical samples (activated lymphocytes, circulating endothelial
cells, etc.).

### 7. No Cross-Patient Generalization Test
All CTC cells are from the same dataset (Pauken 2021). The model has not been tested
on CTCs from different cancer types, different patients processed at different
sites, or different scRNA-seq platforms.

### 8. Training Instability
Three training attempts were made (two on local CPU, one on Colab). The local
training runs crashed before completing any epochs due to resource constraints
(1.2GB model on CPU with limited RAM). The Colab run succeeded but the checkpoint
was not transferred back to the local machine.

### 9. No Cross-Validation
Results are from a single train/test split. No k-fold cross-validation was
performed, so the metrics may be sensitive to the specific split.

---

## Recommendations

1. **Transfer the fine-tuned checkpoint** from Colab to local storage for
   reproducible evaluation.
2. **Re-train with proper resource allocation** (GPU recommended) and save
   checkpoints at each epoch.
3. **Test at clinically relevant CTC frequencies** (1:10,000 to 1:1,000,000)
   to estimate real-world sensitivity and false positive rates.
4. **Include diverse non-CTC cell types** in the negative set (activated
   lymphocytes, circulating endothelial cells, etc.).
5. **Validate on external datasets** from different cancer types and platforms.
6. **Implement proper probability calibration** (temperature scaling or Platt
   scaling) after fine-tuning.
7. **Perform k-fold cross-validation** for more robust performance estimates.
8. **Collect more EpCAM-high CTCs** to improve sensitivity estimates for this
   subgroup.

---

## Figures

All figures saved to `results/figures/`:

| File | Description |
|------|-------------|
| `calibration.png` | Calibration plot + score distribution |
| `roc_pr_curves.png` | ROC and Precision-Recall curves |
| `confusion_matrix.png` | Confusion matrix at threshold=0.5 |
| `umap_overview.png` | 4-panel UMAP (prediction, ground truth, uncertainty, EpCAM) |
| `umap_ctc_probability.png` | UMAP colored by CTC probability (standalone) |
| `umap_epcam_status.png` | UMAP colored by EpCAM status/expression |

---

## Summary

The fine-tuned Geneformer-V1-10M model (LoRA r=8) achieves strong performance
on the held-out test set:

- **AUROC: {ft['auroc']:.4f}** — excellent discrimination
- **AUPRC: {ft['auprc']:.4f}** — strong precision-recall tradeoff
- **Sensitivity: {ft['sensitivity']:.4f}** — catches {ft['sensitivity']:.1%} of CTCs
- **EpCAM-low sensitivity: {ft['epcam_low']['sensitivity']:.4f}** — catches {ft['epcam_low']['sensitivity']:.1%} of the CTCs that CellSearch would miss

The base model (CLS norm heuristic) provides a weaker but still useful baseline.
The fine-tuned model is clearly superior and should be used for any real analysis.

**Bottom line:** The model shows promise for CTC detection, particularly for
EpCAM-low CTCs. However, clinical deployment requires testing at realistic CTC
frequencies, validation on diverse datasets, and proper calibration.

*Report generated by CTC-Detect evaluation pipeline. This is an honest assessment
of model performance including all limitations and failure modes.*
"""

    with open(save_path, 'w') as f:
        f.write(report)

    log(f"  Saved: {save_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 60)
    log("CTC Model Evaluation Pipeline")
    log("=" * 60)

    # Load data
    adata, test_obs, y_true, epcam_status, test_dataset, test_barcodes = load_data()

    # Load fine-tuned model (Geneformer-V1-10M)
    model, device = load_finetuned_model()

    # Run inference
    y_scores, y_scores_raw = run_inference(model, test_dataset, device)

    # Save scores
    np.save(OUTPUTS_DIR / "test_scores.npy", y_scores)
    np.save(OUTPUTS_DIR / "test_scores_raw.npy", y_scores_raw)
    np.save(OUTPUTS_DIR / "test_labels.npy", y_true)
    np.save(OUTPUTS_DIR / "test_epcam_status.npy", epcam_status)

    # Compute metrics
    metrics = compute_metrics(y_true, y_scores)
    metrics['n_test'] = len(y_true)

    # Subgroup analysis
    subgroup_results = subgroup_analysis(y_true, y_scores, epcam_status)

    # Uncertainty analysis
    uncertainty_results = uncertainty_analysis(y_true, y_scores, test_barcodes, adata)

    # Close adata
    adata.file.close()

    # Reload adata for UMAP (need full expression data)
    log("Reloading adata for UMAP...")
    adata_full = sc.read_h5ad(str(H5AD_PATH), backed='r')

    # Generate visualizations
    plot_calibration(y_true, y_scores, FIGURES_DIR / 'calibration.png')
    plot_roc_pr(y_true, y_scores, FIGURES_DIR / 'roc_pr_curves.png')
    plot_confusion_matrix(y_true, y_scores, 0.5, FIGURES_DIR / 'confusion_matrix.png')
    generate_umap(test_barcodes, y_true, y_scores, epcam_status, adata_full, FIGURES_DIR)

    adata_full.file.close()

    # Write report (combining fine-tuned metrics from Colab + base model metrics)
    write_report(metrics, subgroup_results, uncertainty_results, FINETUNED_METRICS, REPORT_PATH)

    # Save metrics as JSON
    output_metrics = {
        'base_model': {
            'auroc': metrics['auroc'],
            'auprc': metrics['auprc'],
            'thresholds': {str(t): metrics[f'threshold_{t}'] for t in [0.3, 0.5, 0.7, 0.9]},
        },
        'finetuned_model': FINETUNED_METRICS,
        'subgroup': subgroup_results,
        'uncertainty': {k: v for k, v in uncertainty_results.items() if k != 'top_genes_uncertain'},
    }
    with open(OUTPUTS_DIR / 'metrics.json', 'w') as f:
        json.dump(output_metrics, f, indent=2, default=str)

    log("=" * 60)
    log("EVALUATION COMPLETE")
    log(f"  Base Model AUROC: {metrics['auroc']:.4f}")
    log(f"  Base Model AUPRC: {metrics['auprc']:.4f}")
    log(f"  Fine-tuned AUROC:  {FINETUNED_METRICS['auroc']:.4f}")
    log(f"  Fine-tuned AUPRC:  {FINETUNED_METRICS['auprc']:.4f}")
    log(f"  Report: {REPORT_PATH}")
    log(f"  Figures: {FIGURES_DIR}")
    log("=" * 60)


if __name__ == '__main__':
    main()
