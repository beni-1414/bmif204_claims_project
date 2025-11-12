#!/usr/bin/env python3
"""
Join Parquet dataset with NDC→ATC mapping.

Usage:
    python join_ndc_to_atc.py input.parquet
"""

import sys
import pandas as pd

from pathlib import Path

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python join_ndc_to_atc.py input.parquet")
        sys.exit(1)

    parquet_path = Path(sys.argv[1])
    mapping_path = Path("/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50/NDC_to_ATC_levels1234_clean.csv")

    # Derive output path automatically
    output_path = parquet_path.with_stem(parquet_path.stem + "_ATC")

    # Load input Parquet file
    print(f"📂 Reading Parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if "ndc11code" not in df.columns:
        raise ValueError("Column 'ndc11code' not found in input parquet file.")

    # Load mapping file
    print(f"📄 Reading mapping CSV: {mapping_path}")
    map_df = pd.read_csv(mapping_path, dtype=str)  # read as string to preserve leading zeros

    # If there are duplicate ndc entries in mapping, raise an error
    if map_df["ndc"].duplicated().any():
        raise ValueError("Duplicate 'ndc' entries found in mapping CSV. Run the cleaner script first.")

    if "ndc" not in map_df.columns:
        raise ValueError("Column 'ndc' not found in mapping CSV.")

    # Normalize NDC formats to 11-digit strings
    df["ndc11code"] = df["ndc11code"].astype(str).str.zfill(11)
    map_df["ndc"] = map_df["ndc"].astype(str).str.zfill(11)

    # Perform left join
    print("🔗 Joining input data with mapping file...")
    merged_df = df.merge(map_df, how="left", left_on="ndc11code", right_on="ndc")

    # Save output parquet
    print(f"💾 Saving merged file to: {output_path}")
    merged_df.to_parquet(output_path, index=False)

    # Summary
    n_total = len(df)
    n_matched = merged_df["ndc"].notna().sum()
    print(f"✅ Done. Matched {n_matched:,} / {n_total:,} rows ({n_matched/n_total*100:.1f}%).")
    print(f"📁 Output file created: {output_path}")

if __name__ == "__main__":
    main()
