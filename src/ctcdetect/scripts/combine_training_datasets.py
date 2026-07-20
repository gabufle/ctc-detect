"""Combine training datasets into one file, with a patient_id column derived
per-source so the training notebook's split can be patient-grouped rather
than random.

Deliberately excludes the Allen Institute atlas — that's held out as an
external test-only negative control, not folded into training (see prior
discussion: different gene panel modality, and shouldn't be diluted into
the training distribution).

Usage:
  python scripts/combine_training_datasets.py \
      --datasets zhang=data/external/zhang/data.h5ad \
                 gse109761=data/external/gse109761/data.h5ad \
                 gse67980=data/external/gse67980/data.h5ad \
      --output data/combined_training_set.h5ad
"""

import argparse
import re
from pathlib import Path

import anndata as ad
import scanpy as sc


def extract_patient_id(barcode: str, source: str) -> str:
    """Best-effort patient ID extraction, source-specific."""
    if source == "gse109761":
        # e.g. Br16_AC12 -> Br16, LM2_A81 -> LM2 (xenograft line, treated as one group),
        # CD_LM2_27 -> LM2, CD_Br16_51 -> Br16, PLT_Br41 -> Br41, HV1_Bcell -> HV1
        b = re.sub(r"^CD_", "", barcode)
        m = re.match(r"^(PLT_)?([A-Za-z]+\d+)", b)
        return m.group(2) if m else b
    elif source == "gse67980":
        # e.g. Pr10.1.2 -> Pr10
        m = re.match(r"^([A-Za-z]+\d+)", barcode)
        return m.group(1) if m else barcode
    elif source == "zhang":
        # Adjust this pattern once you confirm Zhang et al.'s actual barcode convention —
        # placeholder assumes a similar prefix pattern
        m = re.match(r"^([A-Za-z]+\d+)", barcode)
        return m.group(1) if m else barcode
    return barcode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True,
                         help="One or more name=path.h5ad pairs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    datasets = {}
    for pair in args.datasets:
        name, path = pair.split("=", 1)
        datasets[name] = path

    loaded = {}
    for name, path in datasets.items():
        a = sc.read_h5ad(path)
        a.obs["source_dataset"] = name
        a.obs["patient_id"] = [f"{name}_{extract_patient_id(b, name)}" for b in a.obs_names]
        loaded[name] = a
        print(f"{name}: {a.shape[0]} cells x {a.shape[1]} genes, "
              f"{a.obs['patient_id'].nunique()} unique patient groups, "
              f"is_ctc distribution: {a.obs['is_ctc'].value_counts().to_dict()}")

    combined = ad.concat(list(loaded.values()), join="outer", fill_value=0)
    combined.obs_names_make_unique()

    print(f"\nCombined: {combined.shape[0]} cells x {combined.shape[1]} genes")
    print(f"Total unique patient groups: {combined.obs['patient_id'].nunique()}")
    print(f"Source breakdown:\n{combined.obs['source_dataset'].value_counts()}")
    print(f"Label breakdown:\n{combined.obs['is_ctc'].value_counts()}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(str(out_path))
    print(f"\nWrote {out_path}")
    print("\nSpot-check combined.obs[['source_dataset','patient_id','is_ctc']].sample(20) "
          "before uploading, to confirm patient_id extraction looks right for each source.")


if __name__ == "__main__":
    main()
