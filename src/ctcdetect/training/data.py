"""Training data preparation for CTC-Detect.

Handles dataset loading, tokenization, and splitting for Geneformer fine-tuning.
"""

from pathlib import Path
from typing import Tuple, Optional
import json

import numpy as np
import pandas as pd
import scanpy as sc
from datasets import Dataset
from sklearn.model_selection import train_test_split


def load_combined_dataset(data_path: Path) -> sc.AnnData:
    """Load the combined training dataset.

    Args:
        data_path: Path to the combined training h5ad file.

    Returns:
        AnnData object with obs['is_ctc'] and obs['patient_id'] columns.
    """
    adata = sc.read_h5ad(str(data_path))
    print(f"Loaded dataset: {adata.shape[0]} cells x {adata.shape[1]} genes")
    return adata


def split_data(
    adata: sc.AnnData,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify_by: str = "patient_id",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/val/test with patient-level grouping.

    Args:
        adata: AnnData with obs['is_ctc'] and obs['patient_id'] columns.
        test_size: Fraction for test set.
        val_size: Fraction for validation set (from remaining train).
        random_state: Random seed for reproducibility.
        stratify_by: Column in obs to use for grouped stratification.

    Returns:
        Tuple of (train_idx, val_idx, test_idx) arrays.
    """
    n = adata.shape[0]
    indices = np.arange(n)

    # Group by patient_id to ensure no patient appears in multiple splits
    patient_ids = adata.obs[stratify_by].values
    unique_patients = np.unique(patient_ids)

    # First split: train+val vs test
    train_val_patients, test_patients = train_test_split(
        unique_patients,
        test_size=test_size,
        random_state=random_state,
        stratify=[1 if str(p).startswith("CTC") else 0 for p in unique_patients],
    )

    # Second split: train vs val
    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
    )

    # Build indices
    train_idx = np.where(np.isin(patient_ids, train_patients))[0]
    val_idx = np.where(np.isin(patient_ids, val_patients))[0]
    test_idx = np.where(np.isin(patient_ids, test_patients))[0]

    print(f"Train: {len(train_idx)} cells ({len(train_patients)} patients)")
    print(f"Val:   {len(val_idx)} cells ({len(val_patients)} patients)")
    print(f"Test:  {len(test_idx)} cells ({len(test_patients)} patients)")

    return train_idx, val_idx, test_idx


def prepare_training_data(
    adata: sc.AnnData,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    token_dict_path: Path,
    gene_median_path: Path,
    gene_mapping_path: Path,
    output_dir: Path,
    model_input_size: int = 2048,
    nproc: int = 4,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Prepare tokenized datasets for training.

    Args:
        adata: Full AnnData object.
        train_idx, val_idx, test_idx: Split indices.
        token_dict_path: Path to Geneformer token dictionary.
        gene_median_path: Path to gene median dictionary.
        gene_mapping_path: Path to Ensembl mapping dictionary.
        output_dir: Directory to save tokenized datasets.
        model_input_size: Max sequence length for Geneformer.
        nproc: Number of processes for tokenization.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    from geneformer import TranscriptomeTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save split AnnData files
    train_adata = adata[train_idx].copy()
    val_adata = adata[val_idx].copy()
    test_adata = adata[test_idx].copy()

    for name, a in [("train", train_adata), ("val", val_adata), ("test", test_adata)]:
        a.write_h5ad(str(output_dir / f"{name}.h5ad"))

    # Initialize tokenizer
    tk = TranscriptomeTokenizer(
        token_dictionary_file=str(token_dict_path),
        gene_median_file=str(gene_median_path),
        gene_mapping_file=str(gene_mapping_path),
        nproc=nproc,
        model_input_size=model_input_size,
        special_token=False,
        collapse_gene_ids=True,
        model_version="V1",
    )

    # Tokenize each split
    datasets = {}
    for name in ["train", "val", "test"]:
        input_h5ad = str(output_dir / f"{name}.h5ad")
        tokenized_cells, cell_metadata, tokenized_counts = tk.tokenize_anndata(
            input_h5ad, target_sum=10000, file_format="h5ad"
        )

        if len(tokenized_cells) == 0:
            raise ValueError(f"No cells tokenized for {name} split")

        dataset = tk.create_dataset(
            tokenized_cells, cell_metadata, tokenized_counts,
            use_generator=False, keep_uncropped_input_ids=False
        )
        datasets[name] = dataset
        print(f"  {name}: {len(dataset)} cells")

    # Save datasets
    for name, ds in datasets.items():
        ds.save_to_disk(str(output_dir / f"{name}_dataset"))

    return datasets["train"], datasets["val"], datasets["test"]


__all__ = [
    "load_combined_dataset",
    "split_data",
    "prepare_training_data",
]