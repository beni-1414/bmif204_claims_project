from pathlib import Path

import numpy as np
import pandas as pd


def final_common_preprocessing(spells: pd.DataFrame, dem: pd.DataFrame, icd: pd.DataFrame, AE_WINDOW_DAYS: int = 30) -> pd.DataFrame:
    """
    Final preprocessing steps before modelling:
    - Ensure date columns are datetime
    - Filter spells to only those with drug_combo length >= 3
    - Create AE within 30 days label
    - Deduplicate MemberUID + drug_combo, preferring AE rows
    - Merge demographics + spells + icd codes
    """

    # ---------------------------------------------------------------------
    # 3. Basic cleaning / filters
    # ---------------------------------------------------------------------

    print(f"dem shape:    {dem.shape}")
    print(f"spells shape: {spells.shape}")
    print(f"icd shape:    {icd.shape}")

    # Ensure date columns are datetime
    date_cols = ["entry_date", "raw_exit_date", "extended_exit_date",
                "followup_end_date", "first_ae_date"]
    for c in date_cols:
        if c in spells.columns:
            spells[c] = pd.to_datetime(spells[c])

    # Filter spells: keep only drug_combo with len >= 3
    # (and make sure drug_combo is actually a list)
    spells = spells[spells["drug_combo"].map(len) >= 3]
    print(f"spells after drug_combo filter: {spells.shape}")

    ## NEW: filter out all spells with length shorter than 15 days that have no AE
    initial_len = len(spells)
    spells = spells[~((spells["had_ae"] == False) &
                    (spells["spell_length_days"] < 15))]
    print(f"spells after short no-AE spell filter: {spells.shape} (removed {round(100*(initial_len - len(spells))/initial_len, 2)}% rows)")


    # ---------------------------------------------------------------------
    # 4. Clean AE variable & define AE within 30 days
    # ---------------------------------------------------------------------
    # AE within 30 days of entry_date
    spells["ae_within_30d"] = (
        spells["had_ae"]
        & (spells["first_ae_date"] <= spells["entry_date"] + pd.Timedelta(days=AE_WINDOW_DAYS))
    )

    # How many AEs?
    print("Number of spells with AE", spells["had_ae"].sum())

    # Binary label
    spells["y"] = spells["ae_within_30d"].astype(int)
    print("Number of AE within 30 days:", spells["y"].sum())
    print("AE within 30 days rate:", spells["y"].mean())

    # ---------------------------------------------------------------------
    # 4b. Deduplicate MemberUID + drug_combo, preferring AE rows
    # ---------------------------------------------------------------------

    # combo_key: canonical representation of drug combo (sorted tuple)
    spells["combo_key"] = spells["drug_combo"].apply(lambda combo: tuple(sorted(combo)))

    # Use the already-defined outcome for the dedup logic
    spells["y"] = spells["ae_within_30d"].astype(int)

    def deduplicate_member_drug_combos_fast(df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate by (MemberUID, combo_key), preferring rows with y == 1
        (AE within 30 days). Among ties (multiple AE rows), keep first in
        original order.
        """
        df_sorted = df.sort_values(
            ["MemberUID", "combo_key", "y"],
            ascending=[True, True, False],   # y=1 first
            kind="mergesort"                 # stable: keeps earlier rows' order
        )

        deduped = (
            df_sorted
            .drop_duplicates(subset=["MemberUID", "combo_key"], keep="first")
            .reset_index(drop=True)
        )
        return deduped

    print("Spells before deduplication:", len(spells))
    spells = deduplicate_member_drug_combos_fast(spells)
    print("Spells after deduplication:", len(spells))

    # (y already exists and is consistent with ae_within_30d)
    print("Number of AE within 30 days after dedup:", spells["y"].sum())


    # ---------------------------------------------------------------------
    # 5. Merge demographics + spells + icd codes
    # ---------------------------------------------------------------------
    # icd has: ['MemberUID', 'spell_id', 'split_seq', 'icd10_codes']
    # spells has: ['MemberUID', 'spell_id', 'split_seq', ...]
    # demographics has: ['MemberUID', 'birthyear', 'gendercode',
    #                    'raceethnicitytypecode', 'zip3value', 'statecode']

    print("Merging tables...")
    dem = dem.drop_duplicates(subset=["MemberUID"], keep="first") # ensure unique MemberUIDs in dem
    df = spells.merge(dem, on="MemberUID", how="left")
    print("After merging demographics, df shape:", df.shape)
    df = df.merge(icd, on=["MemberUID", "spell_id", "split_seq"], how="left")

    print("Combined df shape:", df.shape)
    print("Number of AE within 30 days after merge:", df["y"].sum())
    # Print how many rows have NaN or None in birthyear
    nan_birthyear_count = df['birthyear'].isna().sum()
    gender_none_count = df['gendercode'].isna().sum()
    print(f"Number of rows with NaN in birthyear: {nan_birthyear_count}")
    print(f"Number of rows with None in gender: {gender_none_count}")

    # Filter out rows that have the first letter+two digits of first_ae_code in icd10_codes to avoid leakage
    def filter_ae_leakage_by_icd_prefix(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows where the ICD-10 prefix (letter + 2 digits) of first_ae_code
        appears in icd10_codes for that same row.

        This avoids label leakage from outcome-defining AE codes into features.
        """
        df = df.copy()

        # --- Ensure icd10_codes is list-like per row ---
        def to_list_or_empty(x):
            if isinstance(x, (list, tuple)):
                return list(x)
            if isinstance(x, np.ndarray):
                return x.tolist()
            if x is None:
                return []
            try:
                if pd.isna(x):
                    return []
            except TypeError:
                pass
            return [x]

        df["icd10_codes"] = df["icd10_codes"].apply(to_list_or_empty)

        # --- Extract AE prefix: letter + 2 digits, e.g. 'I21', 'E11' ---
        df["ae_prefix"] = (
            df["first_ae_code"]
            .astype(str)
            .str.extract(r"^([A-Za-z]\d{2})", expand=False)
        )

        # Only rows with a valid AE prefix can leak
        df["row_id"] = np.arange(len(df))
        candidates = df[(df["ae_prefix"].notna()) & (df["had_ae"])][["row_id", "ae_prefix", "icd10_codes"]]


        # --- Explode ICD list and compute ICD prefixes ---
        exploded = candidates.explode("icd10_codes")
        exploded["icd_prefix"] = (
            exploded["icd10_codes"]
            .astype(str)
            .str.extract(r"^([A-Za-z]\d{2})", expand=False)
        )

        # --- Find rows where AE prefix appears among ICD prefixes ---
        leaky_ids = exploded.loc[
            exploded["icd_prefix"].notna()
            & (exploded["icd_prefix"] == exploded["ae_prefix"]),
            "row_id",
        ].unique()

        print(
            f"Filtering out {len(leaky_ids)} rows "
            f"({len(leaky_ids) / len(df):.2%}) where AE ICD prefix "
            f"appears in icd10_codes (potential leakage)."
        )

        # --- Keep only non-leaky rows ---
        df_filtered = df[~df["row_id"].isin(leaky_ids)].drop(columns=["row_id", "ae_prefix"])
        df_filtered = df_filtered.reset_index(drop=True)
        return df_filtered


    # Filter out rows that have the first letter+two digits of first_ae_code in icd10_codes to avoid leakage
    df = filter_ae_leakage_by_icd_prefix(df)
    print("Number of AE within 30 days after leakage filter:", df["y"].sum())
    print("Rows remaining after leakage filter:", len(df))

    # ---------------------------------------------------------------------
    # 6. Simple feature engineering - SKIP FOR NOW
    # ---------------------------------------------------------------------
    # Age at spell entry
    df["age"] = df["entry_date"].dt.year - df["birthyear"]
    # Drop those rows where age is greater than 120
    before_age_filter_len = len(df)
    df = df[df["age"] <= 120]
    print(f"Dropped {before_age_filter_len - len(df)} rows with age > 120.")

    # You can add more later (e.g., spell_length_days, utilization, etc.)
    numeric_cols = ["age"]
    for col in numeric_cols:
        if col not in df.columns:
            print(f"Warning: numeric col {col} not in df, dropping from list.")
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    # Categorical demographics
    cat_cols = ["gendercode", "raceethnicitytypecode"]

    # Drop rows with missing essential info for now
    df = df.dropna(subset=["age", "gendercode"])
    print("Number of AE within 30 days after dropping missing essential info:", df["y"].sum())
    print("AE within 30 days rate:", df["y"].mean())
    print("Final df shape:", df.shape)

    return df, numeric_cols, cat_cols