"""Convert any downloaded dataset into the ctc-detect h5ad + ground-truth format.

Supports two ways of assigning ground-truth labels, so no per-dataset
Python script is ever needed:

  1. --label-source file
     A separate CSV mapping barcode -> label column (original behavior).

  2. --label-source colname-regex
     Labels are derived directly from the column/cell names using regex
     rules stored in a small JSON config (see configs/labels/*.json for
     examples) — this covers datasets like GSE109761 where cell identity
     is encoded in the sample name itself (e.g. "Br16_AC12" vs "HV1_Bcell")
     rather than provided as a separate file.

Counts formats supported: .h5ad, .csv, .tsv, .txt (genes x cells, tab or
comma separated), or a 10x mtx directory.

Usage (file-based labels):
  python scripts/prepare_external_dataset.py \
      --counts data/raw/ge_et_al_counts.h5ad \
      --label-source file \
      --labels data/raw/ge_et_al_labels.csv \
      --barcode-col cell_id --label-col cell_type \
      --positive-values "tumor,CTC" \
      --output-dir data/external/ge_et_al

Usage (regex-based labels, no separate file needed):
  python scripts/prepare_external_dataset.py \
      --counts data/raw/GSE109761_processed_normalized_matrix_hs.txt \
      --label-source colname-regex \
      --label-config configs/labels/gse109761.json \
      --output-dir data/external/gse109761
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import scanpy as sc


def load_counts(counts_path: Path):
    if counts_path.suffix == ".h5ad":
        adata = sc.read_h5ad(str(counts_path))
    elif counts_path.suffix in (".csv", ".tsv", ".txt"):
        sep = "\t" if counts_path.suffix in (".tsv", ".txt") else ","
        df = pd.read_csv(counts_path, sep=sep, index_col=0)
        # Assume genes x cells (common GEO supplementary convention); transpose to cells x genes
        adata = sc.AnnData(df.T)
    elif counts_path.is_dir():
        adata = sc.read_10x_mtx(str(counts_path), var_names="gene_symbols")
    else:
        raise ValueError(f"Unrecognized counts format: {counts_path}")
    print(f"Loaded counts: {adata.shape[0]} cells x {adata.shape[1]} genes")
    return adata


def labels_from_file(adata, labels_path, barcode_col, label_col, positive_values):
    labels_df = pd.read_csv(labels_path).set_index(barcode_col)

    missing = set(adata.obs_names) - set(labels_df.index)
    if missing:
        print(
            f"Warning: {len(missing)} cells in counts matrix have no label — dropping them"
        )
        adata = adata[adata.obs_names.isin(labels_df.index)].copy()

    labels_df = labels_df.loc[adata.obs_names]
    raw_labels = labels_df[label_col].astype(str).str.lower()
    positive_set = {v.strip().lower() for v in positive_values.split(",")}
    true_label = raw_labels.isin(positive_set).astype(int)
    return adata, true_label


def labels_from_colname_regex(adata, config_path):
    """Derive labels from cell/column names using a JSON config of regex rules.

    Config format:
    {
      "positive_patterns": ["^LM2", "^Br16(?!_.*(Bcell|Tcell|NK|Mono|Gra|plt))", "^CD_Br16"],
      "negative_patterns": ["(Bcell|Tcell|NK|Mono|Gra|plts?)$"],
      "unmatched": "exclude"   // or "error"
    }

    Patterns are checked with re.search, case-insensitive. A name matching
    a negative pattern is label=0; matching a positive pattern (and no
    negative pattern) is label=1. Anything matching neither (or both) is
    "unmatched" and handled per the `unmatched` setting.
    """
    with open(config_path) as f:
        config = json.load(f)

    pos_patterns = [re.compile(p, re.IGNORECASE) for p in config["positive_patterns"]]
    neg_patterns = [re.compile(p, re.IGNORECASE) for p in config["negative_patterns"]]
    unmatched_policy = config.get("unmatched", "exclude")

    labels = {}
    unmatched = []
    for name in adata.obs_names:
        is_neg = any(p.search(name) for p in neg_patterns)
        is_pos = any(p.search(name) for p in pos_patterns)
        if is_neg and not is_pos:
            labels[name] = 0
        elif is_pos and not is_neg:
            labels[name] = 1
        else:
            unmatched.append(name)

    if unmatched:
        print(f"WARNING: {len(unmatched)} cell names matched neither/both patterns:")
        for n in unmatched[:20]:
            print(f"  {n}")
        if unmatched_policy == "error":
            raise ValueError(
                f"{len(unmatched)} unmatched cell names and unmatched policy is 'error'. "
                f"Update the regex patterns in {config_path} to cover them."
            )
        print(f"Excluding these {len(unmatched)} cells (unmatched policy = 'exclude').")

    keep = [n for n in adata.obs_names if n in labels]
    adata = adata[adata.obs_names.isin(keep)].copy()
    true_label = pd.Series([labels[n] for n in adata.obs_names], index=adata.obs_names)
    return adata, true_label


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--counts", required=True)
    parser.add_argument(
        "--label-source", choices=["file", "colname-regex"], required=True
    )
    parser.add_argument("--output-dir", required=True)

    # file-based label args
    parser.add_argument(
        "--labels", help="[file mode] Path to CSV with barcode + label columns"
    )
    parser.add_argument(
        "--barcode-col", default="barcode", help="[file mode] Barcode column name"
    )
    parser.add_argument(
        "--label-col", default="label", help="[file mode] Label column name"
    )
    parser.add_argument(
        "--positive-values",
        help="[file mode] Comma-separated label values counted as positive",
    )

    # regex-based label args
    parser.add_argument(
        "--label-config", help="[colname-regex mode] Path to JSON config of regex rules"
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = load_counts(Path(args.counts))

    if args.label_source == "file":
        if not (args.labels and args.positive_values):
            parser.error("--label-source file requires --labels and --positive-values")
        adata, true_label = labels_from_file(
            adata, args.labels, args.barcode_col, args.label_col, args.positive_values
        )
    else:
        if not args.label_config:
            parser.error("--label-source colname-regex requires --label-config")
        adata, true_label = labels_from_colname_regex(adata, args.label_config)

    print(
        f"Label distribution: {true_label.sum()} positive / {(true_label == 0).sum()} negative "
        f"({true_label.mean() * 100:.2f}% prevalence)"
    )

    h5ad_path = out_dir / "data.h5ad"
    adata.write_h5ad(str(h5ad_path))
    print(f"Wrote {h5ad_path}")

    gt_df = pd.DataFrame({"barcode": adata.obs_names, "true_label": true_label.values})
    gt_path = out_dir / "ground_truth.csv"
    gt_df.to_csv(gt_path, index=False)
    print(f"Wrote {gt_path}")

    print(f"\nNext steps:")
    print(
        f"  python scripts/run_and_eval.py --input {h5ad_path} --ground-truth {gt_path} --output results/{out_dir.name}"
    )


if __name__ == "__main__":
    main()
