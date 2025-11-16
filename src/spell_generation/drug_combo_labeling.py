from pathlib import Path
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attach concurrent drug combos at split start to split_spells parquet."
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="_opioid_sample1M_grace15_minspell7_ae_censoring",
        help="Suffix used in filenames (e.g., '_sample1M' or '')",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    SUFFIX = args.suffix
    data_dir = Path("/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50")

    split_path = data_dir / f"split_spells{SUFFIX}.parquet"
    drug_path = data_dir / f"drug_changes{SUFFIX}.parquet"

    print(f"Reading split spells from: {split_path}")
    split_spells = pd.read_parquet(split_path)

    print(f"Reading drug changes from: {drug_path}")
    drug_changes = pd.read_parquet(drug_path)

    # --- Derive concurrent drug combos at split start ---

    # 1) Minimal preps: keep only add/drop, and necessary columns
    dc = drug_changes.loc[
        drug_changes["change_type"].str.lower().isin(["add", "drop"]),
        ["spell_id", "atc_3_code", "date", "change_type"],
    ].copy()

    # Ensure date is datetime
    dc["date"] = pd.to_datetime(dc["date"])

    # status: 1 for add, 0 for drop
    dc["status"] = (dc["change_type"].str.lower() == "add").astype("Int8")

    # Split starts
    split_starts = split_spells[["spell_id", "split_seq", "entry_date"]].copy()
    split_starts["entry_date"] = pd.to_datetime(split_starts["entry_date"])

    # One row per (spell_id, atc_3_code)
    unique_atc_per_spell = dc[["spell_id", "atc_3_code"]].drop_duplicates()

    # 2) Probes: one probe row per (split_start × atc in that spell)
    pairs = split_starts.merge(unique_atc_per_spell, on="spell_id", how="left")
    pairs = pairs.dropna(subset=["atc_3_code"])

    probes = pairs.rename(columns={"entry_date": "date"}).copy()
    probes["is_probe"] = True
    probes["status"] = pd.NA  # will be filled from prior events

    # 3) Change events in unified schema
    events = dc[["spell_id", "atc_3_code", "date", "status"]].copy()
    events["is_probe"] = False
    events["split_seq"] = pd.NA  # not needed for events

    # 4) Combine events + probes and forward-fill status within (spell, drug)
    all_ev = pd.concat([events, probes], ignore_index=True)

    # Sort so that events come before probes on same day
    all_ev = all_ev.sort_values(["spell_id", "atc_3_code", "date", "is_probe"])

    all_ev["status"] = all_ev.groupby(
        ["spell_id", "atc_3_code"], sort=False
    )["status"].ffill()

    # 5) Active drugs at each probe (status == 1)
    active_at_probe = all_ev[all_ev["is_probe"] & (all_ev["status"] == 1)]

    # 6) Aggregate to combos at split start
    combos = (
        active_at_probe.groupby(["spell_id", "split_seq"])["atc_3_code"]
        .apply(lambda s: sorted(s.astype(str).unique()))  # list of unique sorted drugs
        .reset_index(name="drug_combo")
    )

    # Keep all split_starts, even if they have no active drugs
    split_enriched = split_starts.merge(
        combos, on=["spell_id", "split_seq"], how="left"
    )
    split_enriched["drug_combo"] = split_enriched["drug_combo"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # 7) Attach to full split_spells
    split_spells = split_spells.merge(
        split_enriched[["spell_id", "split_seq", "drug_combo"]],
        on=["spell_id", "split_seq"],
        how="left",
    )
    split_spells["drug_combo"] = split_spells["drug_combo"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # 8) Save out (non-destructive: add suffix)
    out_path = split_path.with_name(split_path.stem + "_with_drugcombo.parquet")
    split_spells.to_parquet(out_path, index=False)

    print("Done.")
    print(f"Input split_spells shape: {split_spells.shape}")
    print(f"Written enriched split_spells with drug_combo to: {out_path}")


if __name__ == "__main__":
    main()
