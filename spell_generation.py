#!/usr/bin/env python3
"""
Polypharmacy spell detection + AE labeling (simplified, with logging)

Expected files in scratch_dir:
  - rx_fills.parquet         [MemberUID, filldate, ndc11code, supplydayscount]
  - adverse_events.parquet   [MemberUID, event_date, CodeType, CodeValue]
  - enrollment.parquet       [MemberUID, effectivedate, terminationdate]
  - demographics.parquet     [MemberUID, ...]
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

MIN_CONCURRENT = 5
EXIT_BELOW_DAYS = 15
EXTEND_DAYS = 15
MIN_SPELL_LEN = 30


def log(msg: str):
    """Simple timestamped logger for cluster environments."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def build_spells_for_member(df):
    """Given all Rx fills for one member, return a list of (entry, raw_exit, extended_exit)."""
    # List to hold change points (date, +1 for start, -1 for end)
    cps = []
    # Group fills by drug code and merge overlapping intervals for each drug
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
                if below_days >= EXIT_BELOW_DAYS:
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

    return spells


def main(scratch_dir="/n/scratch/users/b/bef299/polypharmacy_project/"):
    base = Path(scratch_dir)

    # ---------- Load data ----------
    log(f"Loading Parquet files from: {base}")
    rx = pd.read_parquet(base / "rx_fills.parquet")
    ae = pd.read_parquet(base / "adverse_events.parquet")
    enr = pd.read_parquet(base / "enrollment.parquet")
    log(f"Loaded rx_fills: {len(rx):,} rows | adverse_events: {len(ae):,} rows | enrollment: {len(enr):,} rows")

    # ---------- Prepare Rx intervals ----------
    log("Preparing Rx intervals...")
    rx["filldate"] = pd.to_datetime(rx["filldate"]).dt.date
    rx["start"] = rx["filldate"]
    rx["end"] = rx["filldate"] + pd.to_timedelta(rx["supplydayscount"] - 1, unit="D")

    # ---------- Build spells ----------
    log("Detecting polypharmacy spells per member...")
    spells = []
    n_members = rx["MemberUID"].nunique()
    log(f"Unique members in Rx: {n_members:,}")

    for idx, (mid, g) in enumerate(rx.groupby("MemberUID", sort=False), 1):
        mem_spells = build_spells_for_member(g)
        for sid, (entry, raw_exit, extended) in enumerate(mem_spells, 1):
            spells.append({
                "MemberUID": mid,
                "spell_id": sid,
                "entry_date": entry,
                "raw_exit_date": raw_exit,
                "extended_exit_date": extended,
                "spell_length_days": (extended - entry).days + 1
            })
        if idx % 5000 == 0:
            log(f"Processed {idx:,}/{n_members:,} members...")

    spells = pd.DataFrame(spells)
    log(f"Total spells detected: {len(spells):,}")
    if spells.empty:
        log("No spells detected. Exiting.")
        return

    # ---------- Enrollment censoring ----------
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
        mask = (g["event_date"] >= r.entry_date) & (g["event_date"] <= r.followup_end_date)
        sub = g[mask]
        if sub.empty:
            had_ae.append(False); first_ae_date.append(pd.NaT); ae_codes.append([])
        else:
            had_ae.append(True)
            first_ae_date.append(sub["event_date"].min())
            ae_codes.append(sub["CodeValue"].unique().tolist())
        if idx % 100000 == 0 and idx > 0:
            log(f"Labeled {idx:,} spells...")

    spells["had_ae"] = had_ae
    spells["first_ae_date"] = first_ae_date
    spells["ae_codes"] = ae_codes
    log("AE labeling completed.")

    # ---------- Save outputs ----------
    log("Saving output files...")
    spells.to_parquet(base / "spells_with_labels.parquet", index=False)
    spells.head(500).to_csv(base / "spells_debug_sample.csv", index=False)
    log(f"✅ Wrote {len(spells):,} spells to {base/'spells_with_labels.parquet'}")


if __name__ == "__main__":
    main()
