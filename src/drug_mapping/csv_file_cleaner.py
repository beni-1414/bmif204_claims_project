#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd

def main():

    csv_path = Path("/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50/NDC_to_ATC_final/RxNorm_full_10062025/NDC_to_ATC_levels1234.csv")
    out_path = Path("/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50/NDC_to_ATC_levels1234_clean.csv")

    # Read CSV; keep ndc as string to preserve leading zeros
    df = pd.read_csv(csv_path, dtype={"ndc": "string"})
    if "ndc" not in df.columns:
        raise SystemExit("Column 'ndc' not found.")

    # Normalize minor whitespace
    df["ndc"] = df["ndc"].str.strip()

    before = len(df)
    # Drop duplicate NDCs, keeping the first row seen
    df_clean = df[~df["ndc"].duplicated(keep="first")].copy()
    removed = before - len(df_clean)

    # Build output path: same folder, *_clean.csv
    df_clean.to_csv(out_path, index=False)

    print(f"Input rows:   {before}")
    print(f"Removed dups: {removed}")
    print(f"Output rows:  {len(df_clean)}")
    print(f"Wrote:        {out_path}")

if __name__ == "__main__":
    main()
