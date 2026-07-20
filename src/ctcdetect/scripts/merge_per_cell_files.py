"""Merge many per-cell/per-sample GEO count files into one genes x cells matrix.

Handles the common GEO pattern where raw counts are shipped as one gzipped
file per cell/sample (e.g. GSM2966406_LM2_A81...counts.txt.gz) rather than
a single combined matrix. Extracts the sample name from each filename and
produces one merged table ready for prepare_external_dataset.py.

Usage:
  python scripts/merge_per_cell_files.py \
      --input-dir data/raw/gse109761_raw \
      --output data/raw/gse109761_merged.txt \
      --filename-pattern "GSM\\d+_([A-Za-z0-9_]+?)\\.counts\\.txt\\.gz$" \
      --file-glob "GSM*.txt.gz"

Assumes each input file is a simple two-column table (gene, count), with
no header, tab-separated. Adjust --has-header / --sep if your files differ
— run with --inspect-only first to check one file's structure before
merging everything.
"""

import argparse
import gzip
import re
import sys
from pathlib import Path

import pandas as pd


def open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path)


def extract_sample_name(filename: str, pattern: str) -> str:
    m = re.search(pattern, filename)
    if m:
        return m.group(1)
    # Fallback: strip GSM accession prefix and common suffixes
    stem = re.sub(r"^GSM\d+_", "", filename)
    stem = re.sub(r"\.(counts?)?\.txt(\.gz)?$", "", stem, flags=re.IGNORECASE)
    return stem


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, help="Directory containing one file per cell/sample")
    parser.add_argument("--output", required=True, help="Output path for merged genes x cells .txt matrix")
    parser.add_argument("--file-glob", default="GSM*.txt.gz", help="Glob pattern to select input files")
    parser.add_argument("--filename-pattern", default=r"GSM\d+_([A-Za-z0-9_]+?)\.counts\.txt\.gz$",
                         help="Regex with one capture group extracting the sample name from each filename")
    parser.add_argument("--sep", default="\t", help="Field separator within each file (default: tab)")
    parser.add_argument("--has-header", action="store_true", help="Set if input files have a header row")
    parser.add_argument("--inspect-only", action="store_true",
                         help="Just print the first file's structure and exit, without merging anything")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.file_glob))
    if not files:
        sys.exit(f"No files matched {args.file_glob} in {input_dir}")

    print(f"Found {len(files)} files.")

    if args.inspect_only:
        f = files[0]
        print(f"\nInspecting: {f.name}")
        with open_maybe_gzip(f) as fh:
            for i, line in zip(range(10), fh):
                print(f"  {line.rstrip()}")
        return

    series = {}
    skipped = []
    for f in files:
        sample_name = extract_sample_name(f.name, args.filename_pattern)
        try:
            df = pd.read_csv(
                f, sep=args.sep, header=0 if args.has_header else None,
                names=None if args.has_header else ["gene", "count"],
                index_col=0,
            )
            # If a header was present, assume the (only, or first non-index) remaining column is the count
            count_col = df.columns[0]
            series[sample_name] = df[count_col]
        except Exception as e:
            skipped.append((f.name, str(e)))

    if skipped:
        print(f"\nWARNING: {len(skipped)} files failed to parse and were skipped:")
        for name, err in skipped[:10]:
            print(f"  {name}: {err}")

    if not series:
        sys.exit("No files parsed successfully — check --sep/--has-header/--filename-pattern with --inspect-only first.")

    merged = pd.DataFrame(series)
    merged = merged.fillna(0)
    merged.index.name = "Geneid"

    dupes = merged.columns[merged.columns.duplicated()].tolist()
    if dupes:
        print(f"\nWARNING: {len(dupes)} duplicate sample names extracted from filenames — check --filename-pattern:")
        for d in set(dupes):
            print(f"  {d}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, sep="\t")

    print(f"\nMerged {len(series)} samples -> {merged.shape[0]} genes x {merged.shape[1]} cells")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
