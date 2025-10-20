#!/usr/bin/env python3
"""
Polypharmacy spell detection + AE labeling (simplified, with logging)

Expected files in scratch_dir:
  - rx_fills.parquet         [MemberUID, filldate, ndc11code, supplydayscount]
  - adverse_events.parquet   [MemberUID, event_date, CodeType, CodeValue]
  - enrollment.parquet       [MemberUID, effectivedate, terminationdate]
  - demographics.parquet     [MemberUID, ...]

USAGE:

Usage example:
--------------
python spell_generation.py \
    --output_suffix "_sample1M" \
    --input_suffix "_sample1M" \
    --opioid_flag True \
    --min_concurrent 3 \
    --extend_days 21 \
    --min_spell_len 51
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import argparse
from multiprocessing import Pool, cpu_count
from itertools import islice

def parse_args():
    parser = argparse.ArgumentParser(description="Polypharmacy spell detection + AE labeling")

    parser.add_argument("--output_suffix", type=str, default="_sample1M",
                        help="Suffix used in output filenames (e.g., '_sample1M' or '')")
    parser.add_argument("--input_suffix", type=str, default="_sample1M",
                        help="Suffix used in input filenames (e.g., '_sample1M' or '')")
    parser.add_argument("--opioid_flag", type=lambda x: str(x).lower() in ("1", "true", "yes"),
                        default=False, help="If True, restrict to opioid spells only")
    parser.add_argument("--min_concurrent", type=int, default=3,
                        help="Minimum number of concurrent drugs defining a spell")
    parser.add_argument("--extend_days", type=int, default=21,
                        help="Extension period (days) after last fill below threshold")
    parser.add_argument("--min_spell_len", type=int, default=None,
                        help="Minimum total spell length (if not provided, 30 + extend_days)")

    return parser.parse_args()



args = parse_args()

INPUT_SUFFIX = args.input_suffix
OUTPUT_SUFFIX = args.output_suffix
OPIOID_FLAG = args.opioid_flag
MIN_CONCURRENT = args.min_concurrent
EXTEND_DAYS = args.extend_days
MIN_SPELL_LEN = args.min_spell_len or (30 + EXTEND_DAYS)


def log(msg: str):
    """Simple timestamped logger for cluster environments."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def build_spells_for_member(df):
    """Given all Rx fills for one member, return a list of (entry, raw_exit, extended_exit)."""
    # List to hold change points (date, +1 for start, -1 for end)
    cps = []
    # Group fills by drug code and merge overlapping intervals for each drug
    drug_changes_local = []  # track (MemberUID, date, change, drug)

    for ndc, g in df.groupby("ndc11code"):
        intervals = g[["start", "end"]].values
        intervals = intervals[intervals[:, 0].argsort()]
        merged = []
        cur_start, cur_end = intervals[0]
        for s, e in intervals[1:]:
            # Merge intervals if they overlap or are consecutive
            if s <= cur_end + timedelta(days=1):
                cur_end = max(cur_end, e)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end))

        # Mark start (+1) and end (-1) of each merged interval
        for s, e in merged:
            cps.append((s, +1))
            cps.append((e + timedelta(days=1), -1))
            drug_changes_local.append({
                "MemberUID": df["MemberUID"].iloc[0],
                "date": s,
                "change_type": "add",
                "ndc11code": ndc
            })
            drug_changes_local.append({
                "MemberUID": df["MemberUID"].iloc[0],
                "date": e + timedelta(days=1),
                "change_type": "drop",
                "ndc11code": ndc
            })
    if not cps:
        return []

    # Sort change points by date
    cps.sort()
    compressed = []
    day, delta = cps[0]
    # Compress change points for same day
    for d, dl in cps[1:]:
        if d == day:
            delta += dl
        else:
            compressed.append((day, delta))
            day, delta = d, dl
    compressed.append((day, delta))

    # Detect spells where concurrent drugs >= MIN_CONCURRENT
    spells = []
    count = 0
    in_spell = False
    entry = None
    below_days = 0
    low_start = None

    for i in range(len(compressed)):
        day, delta = compressed[i]
        # Calculate segment length to next change point
        if i < len(compressed) - 1:
            next_day = compressed[i + 1][0]
            seg_len = (next_day - day).days
        else:
            seg_len = 1  # tail

        # Start spell if concurrent count threshold met
        if not in_spell and count >= MIN_CONCURRENT:
            in_spell = True
            entry = day
            below_days = 0
            low_start = None

        if in_spell:
            # Track days below threshold to determine spell exit
            if count < MIN_CONCURRENT:
                if low_start is None:
                    low_start = day
                    below_days = 0
                below_days += seg_len
                if below_days >= EXTEND_DAYS:
                    raw_exit = low_start - timedelta(days=1)
                    extended_exit = raw_exit + timedelta(days=EXTEND_DAYS)
                    # Only keep spells of sufficient length
                    if (extended_exit - entry).days + 1 >= MIN_SPELL_LEN:
                        spells.append((entry, raw_exit, extended_exit))
                    in_spell = False
                    entry = None
                    below_days = 0
                    low_start = None
            else:
                below_days = 0
                low_start = None

        # Update concurrent drug count
        count += delta

    # Handle spell that runs to end of data
    if in_spell:
        raw_exit = compressed[-1][0]
        extended_exit = raw_exit + timedelta(days=EXTEND_DAYS)
        if (extended_exit - entry).days + 1 >= MIN_SPELL_LEN:
            spells.append((entry, raw_exit, extended_exit))

    return spells, drug_changes_local

def process_member(item):
    mid, g = item
    mem_spells, drug_changes_local = build_spells_for_member(g)
    out = []
    for sid, (entry, raw_exit, extended) in enumerate(mem_spells, 1):
        out.append({
            "MemberUID": mid,
            "spell_id": sid,
            "entry_date": entry,
            "raw_exit_date": raw_exit,
            "extended_exit_date": extended,
            "spell_length_days": (extended - entry).days + 1
        })
    return out, drug_changes_local


def main(scratch_dir="/n/scratch/users/b/bef299/polypharmacy_project/"):
    base = Path(scratch_dir)

    # ---------- Load data ----------
    global INPUT_SUFFIX
    if OPIOID_FLAG:
        INPUT_SUFFIX = "_opioid" + INPUT_SUFFIX
    log(f"Loading Parquet files from: {base}")
    rx = pd.read_parquet(base / f"rx_fills{INPUT_SUFFIX}.parquet")
    ae = pd.read_parquet(base / f"adverse_events{INPUT_SUFFIX}.parquet")
    enr = pd.read_parquet(base / f"enrollment{INPUT_SUFFIX}.parquet")
    log(f"Loaded rx_fills: {len(rx):,} rows | adverse_events: {len(ae):,} rows | enrollment: {len(enr):,} rows")

    # ---------- Prepare Rx intervals ----------
    log("Preparing Rx intervals...")
    rx["filldate"] = pd.to_datetime(rx["filldate"]).dt.date
    rx["start"] = rx["filldate"]
    rx["end"] = rx["filldate"] + pd.to_timedelta(rx["supplydayscount"] - 1, unit="D")

        # ---------- Build spells (parallelized) ----------
    log("Detecting polypharmacy spells per member (parallel)...")
    n_members = rx["MemberUID"].nunique()
    log(f"Unique members in Rx: {n_members:,}")

    members_iter = list(rx.groupby("MemberUID", sort=False))
    nproc = min(cpu_count(), 8)  # cap to 8 for cluster jobs

    log(f"Launching parallel pool with {nproc} workers...")
    with Pool(processes=nproc) as pool:
        results = []
        total = len(members_iter)
        log(f"Processing {total:,} members in parallel...")

        for i, r in enumerate(pool.imap_unordered(process_member, members_iter, chunksize=100), 1):
            results.append(r)
            if i % 5000 == 0 or i == total:
                log(f"  → Processed {i:,}/{total:,} members ({i/total:.1%})")

    # Split results into spells and drug changes
    all_spells = []
    all_changes = []
    for spell_list, change_list in results:
        all_spells.extend(spell_list)
        all_changes.extend(change_list)

    spells = pd.DataFrame(all_spells)
    drug_changes = pd.DataFrame(all_changes)
    log(f"Detected total spells: {len(spells):,}")
    log(f"Captured total drug change events: {len(drug_changes):,}")
    if spells.empty:
        log("No spells detected. Exiting.")
        return
    
    # ---------- Only keep spells with opioid ----------
    if OPIOID_FLAG:
        log("Flagging spells that contain at least one opioid NDC (optimized)...")

        opioid_df = pd.read_csv("opioid_ndc11_list.csv", dtype=str)
        opioid_set = set(opioid_df["ndc11"].astype(str).str.strip())

        # Tag opioid fills once
        rx["is_opioid"] = rx["ndc11code"].astype(str).isin(opioid_set)
        rx = rx[rx["is_opioid"]]  # keep only opioid fills (reduces data a lot)

        # Expand each spell into one row per opioid fill overlap
        log("Joining spells with opioid fills...")
        merged = spells.merge(
            rx[["MemberUID", "filldate"]],
            on="MemberUID",
            how="left"
        )

        # Keep only overlaps between fill date and spell window
        mask = (merged["filldate"] >= merged["entry_date"]) & (merged["filldate"] <= merged["extended_exit_date"])
        merged = merged[mask]

        # Keep unique spells that had any opioid overlap
        opioid_spell_ids = merged[["MemberUID", "spell_id"]].drop_duplicates()

        log(f"Spells with opioid overlap: {len(opioid_spell_ids):,} / {len(spells):,}")
        spells = spells.merge(opioid_spell_ids, on=["MemberUID", "spell_id"], how="inner")

    # Enrollment censoring: ensure the spell is within enrollment periods
    log("Applying enrollment censoring...")
    enr["effectivedate"] = pd.to_datetime(enr["effectivedate"]).dt.date
    enr["terminationdate"] = pd.to_datetime(enr["terminationdate"]).dt.date

    followup_end = []
    for _, r in spells.iterrows():
        e_start, e_end = r.entry_date, r.extended_exit_date
        enr_rows = enr[enr["MemberUID"] == r.MemberUID]
        overlap_end = None
        for _, er in enr_rows.iterrows():
            s, t = er.effectivedate, er.terminationdate
            if s <= e_end and t >= e_start:
                overlap_end = max(overlap_end or s, min(t, e_end))
        followup_end.append(overlap_end)
    spells["followup_end_date"] = followup_end
    spells = spells.dropna(subset=["followup_end_date"])
    log(f"Spells after enrollment censoring: {len(spells):,}")

    # Washout censoring: ensure at least 180 days enrollment before spell entry
    log("Applying washout censoring...")
    valid_spells = []
    for _, r in spells.iterrows():
        enr_rows = enr[enr["MemberUID"] == r.MemberUID]
        has_washout = False
        for _, er in enr_rows.iterrows():
            s, t = er.effectivedate, er.terminationdate
            if s <= r.entry_date - timedelta(days=180) and t >= r.entry_date:
                has_washout = True
                break
        if has_washout:
            valid_spells.append(r)

    log(f"Spells before washout: {len(spells):,}")
    log(f"Spells after washout: {len(valid_spells):,}")

    spells = pd.DataFrame(valid_spells)

    # ---------- Apply enrollment + washout censoring to drug_changes ----------
    log("Applying censoring to drug change events (vectorized)...")

    if not drug_changes.empty:
        drug_changes["date"] = pd.to_datetime(drug_changes["date"]).dt.date
        enr["effectivedate"] = pd.to_datetime(enr["effectivedate"]).dt.date
        enr["terminationdate"] = pd.to_datetime(enr["terminationdate"]).dt.date

        # --- Enrollment censoring (vectorized join instead of loops) ---
        merged_enr = pd.merge(
            drug_changes,
            enr[["MemberUID", "effectivedate", "terminationdate"]],
            on="MemberUID",
            how="left"
        )

        mask_enr = (merged_enr["date"] >= merged_enr["effectivedate"]) & (
            merged_enr["date"] <= merged_enr["terminationdate"]
        )

        drug_changes = merged_enr[mask_enr][
            ["MemberUID", "date", "change_type", "ndc11code"]
        ].drop_duplicates()

        # --- Spell window censoring (also vectorized join) ---
        spells["entry_date"] = pd.to_datetime(spells["entry_date"]).dt.date
        spells["extended_exit_date"] = pd.to_datetime(spells["extended_exit_date"]).dt.date

        merged_spell = pd.merge(
            drug_changes,
            spells[["MemberUID", "spell_id", "entry_date", "extended_exit_date"]],
            on="MemberUID",
            how="left"
        )

        mask_spell = (merged_spell["date"] >= merged_spell["entry_date"]) & (
            merged_spell["date"] <= merged_spell["extended_exit_date"]
        )

        drug_changes = merged_spell[mask_spell][
            ["MemberUID", "spell_id", "date", "change_type", "ndc11code"]
        ].drop_duplicates()

        log(f"Drug change events after censoring: {len(drug_changes):,}")
    else:
        log("No drug change events found, skipping censoring.")

    # ---------- AE labeling ----------
    log("Labeling spells with adverse events...")
    ae["event_date"] = pd.to_datetime(ae["event_date"]).dt.date
    ae_groups = {m: g for m, g in ae.groupby("MemberUID", sort=False)}

    had_ae, first_ae_date, ae_codes = [], [], []
    for idx, r in spells.iterrows():
        g = ae_groups.get(r.MemberUID)
        if g is None:
            had_ae.append(False); first_ae_date.append(pd.NaT); ae_codes.append([])
            continue

        mask_in_window = (g["event_date"] >= r.entry_date) & (g["event_date"] <= r.followup_end_date)
        sub = g[mask_in_window]

        if sub.empty:
            had_ae.append(False); first_ae_date.append(pd.NaT); ae_codes.append([])
        else:
            # Exclude AE codes already seen before this spell
            prior_events = g[g["event_date"] < r.entry_date]["CodeValue"].unique()
            new_codes = [c for c in sub["CodeValue"].unique() if c not in prior_events]

            if len(new_codes) == 0:
                had_ae.append(False); first_ae_date.append(pd.NaT); ae_codes.append([])
            else:
                had_ae.append(True)
                first_ae_date.append(sub[sub["CodeValue"].isin(new_codes)]["event_date"].min())
                ae_codes.append(new_codes)

    spells["had_ae"] = had_ae
    spells["first_ae_date"] = first_ae_date
    spells["ae_codes"] = ae_codes
    log("AE labeling completed.")

    # ---------- Save outputs ----------
    log("Saving output files...")
    spells.to_parquet(base / f"spells_with_labels_{EXTEND_DAYS}_days{OUTPUT_SUFFIX}.parquet", index=False)
    drug_changes.to_parquet(base / f"drug_changes_{EXTEND_DAYS}_days{OUTPUT_SUFFIX}.parquet", index=False)
    spells.head(500).to_csv(base / f"spells_debug_sample_{EXTEND_DAYS}_days{OUTPUT_SUFFIX}.csv", index=False)
    log(f"✅ Wrote {len(spells):,} spells to {base/f'spells_with_labels_{EXTEND_DAYS}_days{OUTPUT_SUFFIX}.parquet'}")
    log(f"✅ Wrote {len(drug_changes):,} drug change events to {base/f'drug_changes_{EXTEND_DAYS}_days{OUTPUT_SUFFIX}.parquet'}")


if __name__ == "__main__":
    main()
