#!/usr/bin/env python3
"""
Split spells into consecutive segments by **ADD** drug events until the first AE
(or follow‑up end if no AE). Writes a Parquet file.

INPUT TABLES (Parquet)
- Spells: columns (at minimum)
    MemberUID, spell_id, entry_date, raw_exit_date, extended_exit_date,
    followup_end_date, spell_length_days, had_ae, first_ae_date,
    first_ae_code, ae_codes
- Drug changes: columns (at minimum)
    MemberUID, spell_id, date, change_type, ndc11code

OUTPUT TABLE (Parquet)
- Same columns as Spells, plus:
    split_seq (int, 1..N per original spell)
    segment_end_reason ("next_add" | "ae" | "followup_end")

Rules
- Only considers change_type == "add".
- Relies on (MemberUID, spell_id) to match changes → spells.
- Emits **one row per ADD** within the spell window up to and including the
  cutoff (first_ae_date if present, else followup_end_date; falls back to
  extended_exit_date if needed).
- Segment start = ADD date.
- Segment end = min(day before next ADD, cutoff); clamp to start if negative.
- `had_ae=True` only for the final segment that ends **on** the AE date; its
  `first_ae_date` is set; earlier segments have `had_ae=False` and `first_ae_date=NaT`.
- `spell_id` remains the original spell identifier.

USAGE
  python split_spells_simple.py
  (Edit the file paths below if needed.)
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

# === Hard-coded file paths ===
BASE_PATH = "/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50/"
SUFFIX = "_opioid_sample1M_grace15_minspell7"
SPELLS_PATH = BASE_PATH + "spells_with_labels" + SUFFIX + ".parquet"
CHANGES_PATH = BASE_PATH + "drug_changes" + SUFFIX + ".parquet"
OUT_PATH = BASE_PATH + "split_spells" + SUFFIX + ".parquet"

DATE_COLS_SPELLS = [
    "entry_date",
    "raw_exit_date",
    "extended_exit_date",
    "followup_end_date",
    "first_ae_date",
]
DATE_COLS_CHANGES = ["date"]


def _to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()

def _inclusive_days(start: pd.Timestamp, end: pd.Timestamp) -> Optional[int]:
    if pd.isna(start) or pd.isna(end):
        return None
    return int((end - start).days) + 1

def _prepare_inputs(spells_path: str, changes_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    spells = pd.read_parquet(spells_path)
    changes = pd.read_parquet(changes_path)

    # Normalize dates
    for c in DATE_COLS_SPELLS:
        spells[c] = _to_dt(spells[c])
    for c in DATE_COLS_CHANGES:
        changes[c] = _to_dt(changes[c])

    # Keep only ADD changes
    changes = changes[changes["change_type"].str.lower() == "add"].copy()

    # Sort for efficient grouping
    spells = spells.sort_values(["MemberUID", "spell_id", "entry_date"]).reset_index(drop=True)
    changes = changes.sort_values(["MemberUID", "spell_id", "date"]).reset_index(drop=True)

    return spells, changes


def _build_add_segments_for_spell(spell: pd.Series, add_dates: List[pd.Timestamp]) -> List[pd.Series]:
    """Create one row per ADD date, truncated to AE/cutoff and next ADD."""
    out: List[pd.Series] = []
    # Get a distinct list of ADD dates
    add_dates = sorted(set(add_dates))

    cutoff = spell["first_ae_date"] if pd.notna(spell["first_ae_date"]) else spell["followup_end_date"]
    if pd.isna(cutoff):
        cutoff = spell["extended_exit_date"]

    window_start = spell["entry_date"]
    window_end = min(spell["extended_exit_date"], cutoff) if pd.notna(cutoff) else spell["extended_exit_date"]

    # Consider only ADDs inside the spell window and on/before the cutoff
    add_dates = [d for d in add_dates if (pd.notna(d) and d >= window_start and d <= window_end)]
    if not add_dates:
        row = spell.copy()
        row["split_seq"] = 1
        row["segment_end_reason"] = 'no_adds'
        out.append(row)  # no ADDs; return original spell as-is
        return out

    for i, d in enumerate(add_dates, start=1):  # segments 1..N
        next_d = add_dates[i] if i < len(add_dates) else None

        seg_start = d
        if next_d is not None:
            candidate_end = next_d - pd.Timedelta(days=1)
            end_reason = "next_add"
        else:
            candidate_end = window_end
            end_reason = (
                "ae"
                if pd.notna(spell["first_ae_date"]) and window_end == spell["first_ae_date"]
                else "followup_end"
            )

        seg_end = min(candidate_end, window_end)
        if seg_end < seg_start:
            seg_end = seg_start  # guard for same-day multiple ADDs ordering

        row = spell.copy()
        row["entry_date"] = seg_start
        row["raw_exit_date"] = seg_end
        row["extended_exit_date"] = seg_end
        row["spell_length_days"] = _inclusive_days(seg_start, seg_end)

        is_ae_seg = pd.notna(spell["first_ae_date"]) and seg_end == spell["first_ae_date"]
        row["had_ae"] = bool(is_ae_seg)
        row["first_ae_date"] = spell["first_ae_date"] if is_ae_seg else pd.NaT

        row["split_seq"] = i
        row["segment_end_reason"] = end_reason

        out.append(row)

    return out


def split_spells(spells: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    # Group ADD dates by exact (MemberUID, spell_id)
    adds_by_key = {
        (uid, sid): grp["date"].tolist()
        for (uid, sid), grp in changes.groupby(["MemberUID", "spell_id"], sort=False)
    }

    out_rows: List[pd.Series] = []

    for _, spell in spells.iterrows():
        key = (spell["MemberUID"], spell["spell_id"])
        add_dates = adds_by_key.get(key, [])
        if not isinstance(add_dates, list):
            add_dates = list(add_dates)

        out_rows.extend(_build_add_segments_for_spell(spell, add_dates))

    out = pd.DataFrame(out_rows)

    # Order columns: original first (keeping spell_id), then metadata
    meta_cols = ["split_seq", "segment_end_reason"]
    ordered_cols = [c for c in spells.columns if c != "spell_id"] + ["spell_id"] + meta_cols
    out = out[ordered_cols]

    # Deterministic sort
    out = out.sort_values(["MemberUID", "spell_id", "split_seq", "entry_date"]).reset_index(drop=True)

    return out


def main() -> None:
    print("Loading input data...")
    spells, changes = _prepare_inputs(SPELLS_PATH, CHANGES_PATH)
    print(f"Loaded {len(spells):,} spells and {len(changes):,} drug changes.")
    print("Splitting spells by ADD events...")
    split = split_spells(spells, changes)
    print("Writing output Parquet file...")

    split.to_parquet(OUT_PATH, engine="pyarrow", index=False)
    print(
        f"Wrote {len(split):,} rows to {OUT_PATH}. "
        f"Unique spells: {spells['spell_id'].nunique():,}; members: {spells['MemberUID'].nunique():,}"
    )


if __name__ == "__main__":
    main()
