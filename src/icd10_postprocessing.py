from pathlib import Path
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Polypharmacy spell detection + AE labeling"
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="sample1M",
        help="Suffix used in filenames (e.g., '_sample1M' or '')",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50/",
        help="Base path to data files",
    )

    return parser.parse_args()


def process_icd10_file(suffix: str, base_path: str) -> pd.DataFrame:
    """Read ICD-10 parquet, normalize/filter codes, cluster, and return result."""
    path = Path(
        f"{base_path}   "
        f"icd10_codes_from_spells{suffix}.parquet"
    )

    print(f"Processing ICD-10 data from: {path}")

    # Read
    df = pd.read_parquet(
        path,
        columns=["MemberUID", "spell_id", "split_seq", "icd10_code"],
    )

    # Normalize codes (upper, strip non-alphanumerics)
    codes = (
        df["icd10_code"]
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )

    # ICD-10-CM (diagnoses): Letter + 2–6 more alphanumerics (3–7 total, no dot)
    cm_mask = codes.str.match(r"^[A-Z][0-9A-Z]{2,6}$")

    mask = cm_mask

    # Keep only valid ICD-10, make 3-char cluster
    df_valid = df.loc[mask].copy()
    df_valid["icd10_code_norm"] = codes[mask]
    df_valid["icd10_code3"] = df_valid["icd10_code_norm"].str[:3]

    # Group and collect unique sorted clusters per spell
    out = (
        df_valid.groupby(["MemberUID", "spell_id", "split_seq"], as_index=False)
        .agg(icd10_codes=("icd10_code3", lambda s: sorted(pd.unique(s))))
    )

    print("Processing complete. Processed", out.shape[0], "rows.")

    # Write output
    out_path = path.with_name(path.stem + "_clustered.parquet")
    out.to_parquet(out_path, index=False)
    # If you ever want CSV:
    # out.to_csv(path.with_name(path.stem + "_clustered.csv"), index=False)

    print(f"Wrote clustered data to: {out_path}")
    return out


def main():
    args = parse_args()
    process_icd10_file(args.suffix, args.path)


if __name__ == "__main__":
    main()
