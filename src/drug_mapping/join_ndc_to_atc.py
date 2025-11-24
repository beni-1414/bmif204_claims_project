#!/usr/bin/env python3
"""
Chunked join of RX Parquet dataset with NDC→ATC mapping.

Keeps ALL columns from the input parquet and adds:
    RXCUI, rxnorm_description, atc_3_code, atc_3_name

Usage:
    python join_ndc_to_atc_chunked.py input.parquet
"""

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    if len(sys.argv) != 2:
        print("Usage: python join_ndc_to_atc_chunked.py input.parquet")
        sys.exit(1)

    parquet_path = Path(sys.argv[1])
    mapping_path = Path(
        "/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50/"
        "NDC_to_ATC_levels1234_clean.csv"
    )

    output_path = parquet_path.with_stem(parquet_path.stem + "_ATC")

    # -----------------------------
    # Load mapping (small, once)
    # -----------------------------
    usecols = ["ndc", "RXCUI", "rxnorm_description", "atc_3_code", "atc_3_name"]
    print(f"📄 Reading mapping CSV (limited columns): {mapping_path}")
    map_df = pd.read_csv(mapping_path, dtype=str, usecols=usecols)

    if "ndc" not in map_df.columns:
        raise ValueError("Column 'ndc' not found in mapping CSV.")

    if map_df["ndc"].duplicated().any():
        raise ValueError(
            "Duplicate 'ndc' entries found in mapping CSV. "
            "Run the cleaner script first."
        )

    # Normalize mapping NDC to 11 digits
    map_df["ndc"] = map_df["ndc"].astype(str).str.zfill(11)

    # Use index for efficient joins
    map_df = map_df.set_index("ndc")

    # -----------------------------
    # Stream RX parquet in chunks
    # -----------------------------
    print(f"📂 Opening Parquet file for chunked read: {parquet_path}")
    pf = pq.ParquetFile(parquet_path)

    writer = None
    total_rows = 0
    total_matched = 0

    # tune batch_size depending on memory; 1M is a decent starting point
    batch_size = 1_000_000

    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=batch_size)):
        print(f"🔄 Processing batch {batch_idx} ...")

        # Convert this batch to pandas
        df = batch.to_pandas()

        if "ndc11code" not in df.columns:
            raise ValueError("Column 'ndc11code' not found in input parquet file.")

        # Normalize NDC to 11 digits in this chunk
        df["ndc11code"] = df["ndc11code"].astype(str).str.zfill(11)

        # Join mapping onto this chunk
        # map_df index is 'ndc'; join with ndc11code column
        merged_df = df.join(map_df, on="ndc11code")

        # Stats
        n_rows = len(merged_df)
        n_matched = merged_df["RXCUI"].notna().sum()
        total_rows += n_rows
        total_matched += n_matched

        print(
            f"  Batch {batch_idx}: {n_rows:,} rows, "
            f"{n_matched:,} matched in this batch "
            f"({(n_matched / n_rows * 100) if n_rows else 0:.1f}%)"
        )

        # Write to output parquet incrementally
        table = pa.Table.from_pandas(merged_df, preserve_index=False)
        if writer is None:
            print(f"💾 Creating Parquet writer at: {output_path}")
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)

        # Explicitly free memory for this batch
        del df, merged_df, table

    if writer is not None:
        writer.close()
    else:
        print("⚠️ No batches processed; nothing written.")
        return

    print(
        f"✅ Done. Matched {total_matched:,} / {total_rows:,} rows "
        f"({(total_matched / total_rows * 100) if total_rows else 0:.1f}%)."
    )
    print(f"📁 Output file created: {output_path}")


if __name__ == "__main__":
    main()
