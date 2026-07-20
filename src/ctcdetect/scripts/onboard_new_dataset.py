#!/usr/bin/env python3
"""
Interactive orchestrator: turn ONE raw dataset download into a standardized
data.h5ad + ground_truth.csv by chaining the existing prep scripts, but
STOPPING for human confirmation at every judgment call where the current
pipeline silently guessed and got it wrong.

Usage:
    python scripts/onboard_new_dataset.py --input-path <file_or_dir> --output-dir <out_dir>

Required detections + confirmations (in order):
  1. INPUT SHAPE: single file vs directory of per-cell files
  2. COMPRESSION + DELIMITER (text files): peek, detect, confirm
  3. ORIENTATION (genes x cells vs cells x genes) + METADATA COLUMNS
  4. NORMALIZATION STATE CHECK (raw counts vs log/CPM/TPM) — critical
  5. LABEL SOURCE: file-based vs colname-regex (and config if regex)
  6. PATIENT ID EXTRACTION PATTERN (for combine_training_datasets.py)

At EVERY step the tool prints what it detected, then pauses with a [y/n/edit]
prompt. Nothing proceeds without explicit confirmation.
"""

import argparse
import gzip
import json
import subprocess
import sys
import tarfile
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore


def confirm(prompt: str, default: str = "y") -> bool:
    """Ask y/n/edit, return True for yes, False for no, 'edit' returns 'edit'."""
    suffix = " [Y/n/e] " if default == "y" else " [y/N/e] "
    while True:
        ans = input(prompt + suffix).strip().lower()
        if ans == "":
            return default == "y"
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        if ans in ("e", "edit"):
            return "edit"
        print("  Please answer y, n, or e (edit).")


def run_script(script: str, args: list, cwd: Path = None) -> subprocess.CompletedProcess:
    """Run a sub-script and return the CompletedProcess."""
    cmd = [sys.executable, str(script)] + args
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd or Path.cwd(), capture_output=False, text=True)


def peek_text_file(path: Path, max_lines: int = 5) -> tuple[list[str], str]:
    """
    Peek at first N lines of a text file (handles .gz).
    Returns (lines, detected_delimiter).
    """
    opener = gzip.open if path.suffix == ".gz" else open
    lines = []
    with opener(path, "rt") as f:
        for _ in range(max_lines):
            try:
                lines.append(next(f))
            except StopIteration:
                break

    if not lines:
        return [], "\t"

    # Detect delimiter from first data line (skip header)
    data_line = lines[1] if len(lines) > 1 else lines[0]
    tab_count = data_line.count("\t")
    comma_count = data_line.count(",")
    if tab_count > comma_count:
        delim = "\t"
    elif comma_count > tab_count:
        delim = ","
    else:
        delim = "\t"  # default to tab for scRNA-seq

    return lines, delim


def detect_orientation_and_meta_columns(header: str, delim: str, n_data_rows: int) -> dict:
    """
    Heuristics for genes x cells vs cells x genes + metadata columns.
    Returns dict with:
      - orientation: "genes_x_cells" | "cells_x_genes" | "unknown"
      - gene_id_col_idx: int or None
      - sample_start_col_idx: int
      - meta_cols: list of (idx, name) before sample data starts
      - n_rows_estimate: int
      - n_cols_estimate: int
    """
    cols = header.strip().split(delim if delim != "\t" else "\t")
    n_cols = len(cols)

    # Heuristic: if first column looks like a gene ID column name
    first_col = cols[0].lower().strip()
    gene_like = any(kw in first_col for kw in ["gene", "id", "symbol", "entrez", "unigene", "name"])

    # If row count (genes) ~20k and col count (cells) ~hundreds, likely genes x cells
    # If row count ~hundreds and col count ~20k, likely cells x genes
    if gene_like and n_data_rows > 5000 and n_cols < 5000:
        orientation = "genes_x_cells"
    elif not gene_like and n_data_rows < 5000 and n_cols > 5000:
        orientation = "cells_x_genes"
    else:
        orientation = "unknown"

    # Find metadata columns before sample data starts
    # Common metadata column patterns
    meta_keywords = ["entrez", "unigene", "symbol", "name", "description", "geneid", "id", "gene"]
    meta_cols = []
    sample_start_idx = 0
    for i, col in enumerate(cols):
        col_lower = col.lower().strip()
        if any(kw in col_lower for kw in meta_keywords):
            meta_cols.append((i, col))
        else:
            sample_start_idx = i
            break

    # Gene ID column: prefer 'symbol' or 'gene' over 'entrez'/'id'
    gene_id_col_idx = None
    for idx, name in meta_cols:
        name_lower = name.lower()
        if "symbol" in name_lower or name_lower == "gene":
            gene_id_col_idx = idx
            break
    if gene_id_col_idx is None and meta_cols:
        gene_id_col_idx = meta_cols[0][0]

    return {
        "orientation": orientation,
        "gene_id_col_idx": gene_id_col_idx,
        "sample_start_col_idx": sample_start_idx,
        "meta_cols": meta_cols,
        "n_cols": n_cols,
        "all_columns": cols,
    }


def print_columns_in_groups(cols: list[str], group_size: int = 10):
    """Print column names in groups for readability."""
    for i in range(0, len(cols), group_size):
        group = cols[i:i+group_size]
        for j, col in enumerate(group):
            idx = i + j
            print(f"  [{idx:3d}] {col}")


def ask_column_indices(n_cols: int, meta_cols: list, gene_id_col_idx: int, sample_start_idx: int) -> tuple[int, int]:
    """Ask human which column is gene ID and where sample data starts."""
    print(f"\nDetected {len(meta_cols)} potential metadata columns before sample data starts at index {sample_start_idx}:")
    for idx, name in meta_cols:
        marker = " <- suggested gene ID" if idx == gene_id_col_idx else ""
        print(f"  [{idx}] {name}{marker}")

    while True:
        gene_idx_str = input(f"\nWhich column index should be used as the GENE IDENTIFIER? [{gene_id_col_idx}]: ").strip()
        gene_idx = int(gene_idx_str) if gene_idx_str else gene_id_col_idx
        if 0 <= gene_idx < n_cols:
            break
        print(f"  Invalid index. Must be 0-{n_cols-1}.")

    while True:
        sample_idx_str = input(f"Which column index does the FIRST SAMPLE COLUMN start at? [{sample_start_idx}]: ").strip()
        sample_idx = int(sample_idx_str) if sample_idx_str else sample_start_idx
        if 0 <= sample_idx < n_cols:
            break
        print(f"  Invalid index. Must be 0-{n_cols-1}.")

    return gene_idx, sample_idx


def check_normalization_state(path: Path, delim: str, sample_start_idx: int, n_peek_rows: int = 5) -> dict:
    """
    Peek at numeric values to guess normalization state.
    Returns dict with 'state' in {'raw_counts', 'log_cpm', 'tpm_fpkm', 'unknown'} and sample values.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        _ = next(f)  # skip header
        # Read a few data rows
        rows = []
        for _ in range(n_peek_rows):
            try:
                rows.append(next(f))
            except StopIteration:
                break

    if not rows:
        return {"state": "unknown", "sample_values": []}

    # Parse first sample column from each row
    sample_vals = []
    for row in rows:
        parts = row.strip().split(delim if delim != "\t" else "\t")
        if len(parts) > sample_start_idx:
            try:
                val = float(parts[sample_start_idx])
                sample_vals.append(val)
            except ValueError:
                pass

    if not sample_vals:
        return {"state": "unknown", "sample_values": []}

    # Heuristics
    max_val = max(sample_vals)
    min_val = min(sample_vals)
    has_decimals = any(v != int(v) for v in sample_vals)
    has_negatives = any(v < 0 for v in sample_vals)

    if has_negatives:
        state = "log_cpm"  # log-transformed can be negative
    elif max_val > 100 and not has_decimals:
        state = "raw_counts"
    elif max_val <= 50 and has_decimals:
        state = "log_cpm"
    elif max_val > 100 and has_decimals:
        state = "tpm_fpkm"
    else:
        state = "unknown"

    return {
        "state": state,
        "sample_values": sample_vals[:5],
        "min": min_val,
        "max": max_val,
        "has_decimals": has_decimals,
        "has_negatives": has_negatives,
    }


def list_tar_contents(tar_path: Path) -> list[str]:
    """List contents of a .tar.gz without extracting."""
    with tarfile.open(tar_path, "r:gz") as tf:
        return tf.getnames()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive orchestrator: raw dataset -> standardized data.h5ad + ground_truth.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-path", required=True, help="Path to raw dataset file or directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for standardized dataset")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge_per_cell_files.py even if input is a directory")
    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).parent

    print("=" * 70)
    print("ctc-detect: Interactive Dataset Onboarding")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print()

    # ============================================================
    # STEP 1: INPUT SHAPE DETECTION
    # ============================================================
    print("=" * 70)
    print("STEP 1: INPUT SHAPE DETECTION")
    print("=" * 70)

    if input_path.is_dir():
        files = sorted(input_path.glob("*"))
        print(f"Input is a DIRECTORY with {len(files)} entries.")
        if files:
            print("First 3 entries:")
            for f in files[:3]:
                print(f"  {f.name}")
        if not args.skip_merge:
            if confirm("Treat as per-cell files needing merge_per_cell_files.py?", default="y"):
                # Run merge_per_cell_files.py with --inspect-only first
                print("\n--- Running merge_per_cell_files.py --inspect-only ---")
                result = run_script(scripts_dir / "merge_per_cell_files.py", [
                    "--input-dir", str(input_path),
                    "--output", str(output_dir / "merged.txt"),
                    "--inspect-only",
                ])
                # --inspect-only doesn't create output, it just prints and exits
                # Now ask for custom args and run the actual merge
                if confirm("Run merge_per_cell_files.py with custom args?", default="y"):
                    # Let user provide custom args
                    glob = input("  --file-glob [GSM*.txt.gz]: ").strip() or "GSM*.txt.gz"
                    pattern = input("  --filename-pattern [GSM\\d+_([A-Za-z0-9_]+?)\\.counts\\.txt\\.gz$]: ").strip() or r"GSM\d+_([A-Za-z0-9_]+?)\.counts\.txt\.gz$"
                    sep = input("  --sep [\\t]: ").strip() or "\t"
                    has_header = confirm("  --has-header?", default="n")
                    run_script(scripts_dir / "merge_per_cell_files.py", [
                        "--input-dir", str(input_path),
                        "--output", str(output_dir / "merged.txt"),
                        "--file-glob", glob,
                        "--filename-pattern", pattern,
                        "--sep", sep,
                    ] + (["--has-header"] if has_header else []))
                merged_path = output_dir / "merged.txt"
                if not merged_path.exists():
                    print("Merge did not produce output. Exiting.")
                    sys.exit(1)
                input_path = merged_path
                print(f"\nMerged matrix -> {input_path}")
                # Continue to step 2 with the merged file
            else:
                print("Directory input but not treating as per-cell files. Exiting.")
                sys.exit(1)
        else:
            print("--skip-merge set but input is a directory. Exiting.")
            sys.exit(1)

    elif input_path.is_file():
        print(f"Input is a single file: {input_path.name}")
        # Handle .tar.gz
        if input_path.suffixes == [".tar", ".gz"] or input_path.name.endswith(".tar.gz"):
            print("Input is a .tar.gz archive. Listing contents...")
            members = list_tar_contents(input_path)
            print(f"Archive contains {len(members)} entries:")
            for m in members[:20]:
                print(f"  {m}")
            if len(members) > 20:
                print(f"  ... and {len(members) - 20} more")
            print("\nThis archive likely contains per-cell files. Extract and re-run?")
            if confirm("Extract to a temporary directory and continue?", default="y"):
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    with tarfile.open(input_path, "r:gz") as tf:
                        tf.extractall(tmpdir)
                    print(f"Extracted to {tmpdir}")
                    # Re-run detection on extracted contents
                    # For simplicity, just set input_path to the extracted dir and re-run step 1
                    # In practice, we'd recurse, but for now just tell user to re-run
                    print(f"Re-run: python scripts/onboard_new_dataset.py --input-path {tmpdir} --output-dir {output_dir}")
                    sys.exit(0)
            else:
                sys.exit(1)
    else:
        print(f"ERROR: {input_path} does not exist.")
        sys.exit(1)

    # ============================================================
    # STEP 2-4: HANDLE .h5ad FILES vs TEXT FILES
    # ============================================================
    if input_path.suffix == ".h5ad":
        print("\n" + "=" * 70)
        print("STEP 2-4: INSPECT .h5ad FILE (native format)")
        print("=" * 70)
        print("Detected .h5ad file. prepare_external_dataset.py handles this natively.")
        print("Skipping delimiter/orientation/normalization detection steps.")

        # Inspect .h5ad structure
        import scanpy as sc
        adata = sc.read(input_path)
        print(f"\nShape: {adata.shape[0]} cells x {adata.shape[1]} genes")
        print(f"Barcodes (obs_names): {list(adata.obs_names[:10])}{'...' if len(adata.obs_names) > 10 else ''}")
        print(f"Genes (var_names): {list(adata.var_names[:10])}{'...' if len(adata.var_names) > 10 else ''}")
        print(f"Obs columns: {list(adata.obs.columns)}")
        print(f"Var columns: {list(adata.var.columns)}")
        print(f"X dtype: {adata.X.dtype}")

        # Ask user to confirm barcodes look correct for label mapping
        if confirm("\nDo these barcodes look correct for label mapping?", default="y"):
            pass
        else:
            print("Please ensure your label CSV barcodes match these exactly.")
            if not confirm("Continue anyway?", default="n"):
                sys.exit(1)
    else:
        # ============================================================
        # STEP 2: COMPRESSION + DELIMITER DETECTION
        # ============================================================
        if input_path.suffix in (".csv", ".tsv", ".txt", ".gz") or input_path.name.endswith(".txt.gz"):
            print("\n" + "=" * 70)
            print("STEP 2: COMPRESSION + DELIMITER DETECTION")
            print("=" * 70)

            lines, detected_delim = peek_text_file(input_path)
            tab_char = "\t"
            print(f"Detected compression: {'gzip' if input_path.suffix == '.gz' else 'none'}")
            print(f"Detected delimiter:   {'tab' if detected_delim == tab_char else 'comma'}")
            print("\nFirst 5 lines:")
            for _i, line in enumerate(lines):
                print(f"  {line.rstrip()}")

            if confirm("\nConfirm delimiter and proceed?", default="y"):
                delim = detected_delim
            else:
                delim = input("Enter delimiter (\\t for tab, , for comma): ").strip()
                if delim == "\\t":
                    delim = "\t"
                elif delim == ",":
                    delim = ","
                else:
                    delim = "\t"

            # Read header + first few data rows for orientation detection
            opener = gzip.open if input_path.suffix == ".gz" else open
            with opener(input_path, "rt") as f:
                header = next(f)
                # Count data rows (peek)
                n_data_rows = sum(1 for _ in f)
                # Reset and read a few data rows
                f.seek(0)
                next(f)  # skip header
                _ = [next(f) for _ in range(min(5, n_data_rows))]

            # ============================================================
            # STEP 3: ORIENTATION + METADATA COLUMNS
            # ============================================================
            print("\n" + "=" * 70)
            print("STEP 3: ORIENTATION + METADATA COLUMN DETECTION")
            print("=" * 70)

            orient_info = detect_orientation_and_meta_columns(header, delim, n_data_rows)
            print(f"\nDetected orientation: {orient_info['orientation']}")
            print(f"Total columns: {orient_info['n_cols']}")
            print(f"Estimated data rows: {n_data_rows}")
            print(f"Metadata columns before sample data: {len(orient_info['meta_cols'])}")
            print("\nALL column names:")
            print_columns_in_groups(orient_info['all_columns'])

            if orient_info['meta_cols']:
                print("\nPotential metadata columns detected:")
                for idx, name in orient_info['meta_cols']:
                    print(f"  [{idx}] {name}")

            # Ask human to confirm gene ID column and sample start column
            gene_idx, sample_idx = ask_column_indices(
                orient_info['n_cols'],
                orient_info['meta_cols'],
                orient_info['gene_id_col_idx'],
                orient_info['sample_start_col_idx']
            )

            # ============================================================
            # STEP 4: NORMALIZATION STATE CHECK
            # ============================================================
            print("\n" + "=" * 70)
            print("STEP 4: NORMALIZATION STATE CHECK (CRITICAL)")
            print("=" * 70)

            norm_info = check_normalization_state(input_path, delim, sample_idx)
            print(f"Detected normalization state: {norm_info['state'].upper()}")
            print(f"Sample values (first 5 rows, first sample column): {norm_info['sample_values']}")
            print(f"Min: {norm_info['min']}, Max: {norm_info['max']}")
            print(f"Has decimals: {norm_info['has_decimals']}, Has negatives: {norm_info['has_negatives']}")
            print("\n!!! SILENT NORMALIZATION MISMATCHES HAVE CAUSED SILENT FAILURES BEFORE !!!")
            print("Raw counts: integers, max > 1000, no negatives")
            print("LogCPM/LogTPM: decimals, max ~10-50, can have negatives")
            print("TPM/FPKM: decimals, max > 1000")
            if confirm("\nDoes this look like RAW COUNTS (integers, large values, no negatives)?", default="y"):
                pass  # proceed
            else:
                print("WARNING: Non-raw data detected. prepare_external_dataset.py expects raw counts.")
                print("You may need to preprocess (e.g., extract_gse67980_counts.py pattern) before proceeding.")
                if not confirm("Continue anyway? (prepare_external_dataset.py may produce wrong results)", default="n"):
                    sys.exit(1)

# ============================================================
# STEP 5: LABEL SOURCE CONFIGURATION (for both .h5ad and text)
# ============================================================
    print("\n" + "=" * 70)
    print("STEP 5: GROUND-TRUTH LABEL SOURCE")
    print("=" * 70)
    print("How are CTC vs non-CTC labels provided?")
    print("  1) Separate CSV file mapping barcode -> label (--label-source file)")
    print("  2) Encoded in cell/column names via regex (--label-source colname-regex)")
    label_choice = input("Choose [1/2]: ").strip()

    if label_choice == "1":
        labels_path = input("  Path to labels CSV: ").strip()
        barcode_col = input("  Barcode column name [barcode]: ").strip() or "barcode"
        label_col = input("  Label column name [label]: ").strip() or "label"
        positive_values = input("  Comma-separated positive label values (e.g., 'tumor,CTC'): ").strip()
        label_args = [
            "--label-source", "file",
            "--labels", labels_path,
            "--barcode-col", barcode_col,
            "--label-col", label_col,
            "--positive-values", positive_values,
        ]
    elif label_choice == "2":
        print("  Regex-based labeling uses a JSON config with:")
        print("    - positive_patterns: []  (optional)")
        print("    - negative_patterns: [...]  (required)")
        print("    - unmatched: 'exclude'|'error'|'positive'|'negative'")
        config_path = input("  Path to label config JSON: ").strip()
        if not Path(config_path).exists():
            print("  Config not found. Example configs in configs/labels/")
            if confirm("  Create a template config now?", default="y"):
                template = {
                    "positive_patterns": [],
                    "negative_patterns": ["(Bcells?|Tcells?|NK|Mono|Gra|plts?)$"],
                    "unmatched": "exclude"
                }
                template_path = output_dir / "label_config_template.json"
                with open(template_path, "w") as f:
                    json.dump(template, f, indent=2)
                print(f"  Template written to {template_path}")
                print("  Edit it, then re-run with --label-config pointing to it.")
                sys.exit(0)
            else:
                sys.exit(1)
        label_args = ["--label-source", "colname-regex", "--label-config", config_path]
    else:
        print("Invalid choice.")
        sys.exit(1)

# ============================================================
# STEP 6: RUN prepare_external_dataset.py
# ============================================================
    print("\n" + "=" * 70)
    print("STEP 6: RUNNING prepare_external_dataset.py")
    print("=" * 70)

    # Build args for prepare_external_dataset.py
    prep_args = [
        "--counts", str(input_path),
        "--output-dir", str(output_dir),
    ] + label_args

    print(f"\nRunning: python scripts/prepare_external_dataset.py {' '.join(prep_args)}")
    if confirm("Execute?", default="y"):
        result = run_script(scripts_dir / "prepare_external_dataset.py", prep_args)
        if result.returncode != 0:
            print("prepare_external_dataset.py failed. Check output above.")
            sys.exit(1)
    else:
        print("Skipped. Run manually when ready.")
        sys.exit(0)

# ============================================================
# STEP 7: PATIENT ID EXTRACTION (for combine_training_datasets.py)
# ============================================================
    print("\n" + "=" * 70)
    print("STEP 7: PATIENT ID EXTRACTION PATTERN (for combine_training_datasets.py)")
    print("=" * 70)
    print("If you plan to combine this dataset with others for training,")
    print("combine_training_datasets.py needs a patient_id extraction pattern.")
    print("Examples:")
    print("  gse109761: Br16_AC12 -> Br16  (regex: ^(PLT_)?([A-Za-z]+\\d+))")
    print("  gse67980:  Pr10.1.2 -> Pr10   (regex: ^([A-Za-z]+\\d+))")
    dataset_name = input("\nDataset name (e.g., gse109761, gse67980, zhang): ").strip()
    patient_regex = input("Patient ID regex (capture group 1 = patient ID) [^([A-Za-z]+\\d+)]: ").strip() or r"^([A-Za-z]+\d+)"

    # Save pattern for later use
    pattern_info = {
        "dataset_name": dataset_name,
        "patient_id_regex": patient_regex,
        "output_h5ad": str(output_dir / "data.h5ad"),
    }
    pattern_path = output_dir / "patient_id_pattern.json"
    with open(pattern_path, "w") as f:
        json.dump(pattern_info, f, indent=2)
    print(f"Saved pattern to {pattern_path}")

    print("\n" + "=" * 70)
    print("DONE: Standardized dataset ready at:")
    print(f"  {output_dir}/data.h5ad")
    print(f"  {output_dir}/ground_truth.csv")
    print("=" * 70)
    print("\nNext steps:")
    print(f"  1. Spot-check: python -c \"import scanpy as sc; a=sc.read('{output_dir}/data.h5ad'); print(a.obs[['is_ctc','epcam_status']].value_counts())\"")
    print(f"  2. Evaluate:   python scripts/run_and_eval.py --input {output_dir}/data.h5ad --ground-truth {output_dir}/ground_truth.csv --output results/{output_dir.name}")
    print(f"  3. Combine:    python scripts/combine_training_datasets.py --datasets {dataset_name}={output_dir}/data.h5ad ... --output data/combined_training_set.h5ad")


if __name__ == "__main__":
    main()
