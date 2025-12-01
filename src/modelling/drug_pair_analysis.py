#!/usr/bin/env python

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from collections import Counter
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

from src.modelling.final_preprocessing import final_common_preprocessing


def chi2_pvalue_vs_rest(row, overall_ae, overall_no_ae):
    a = row["n_ae"]
    b = row["n_no_ae"]
    c = overall_ae - a
    d = overall_no_ae - b
    chi2, p, *_ = chi2_contingency(np.array([[a, b], [c, d]]))
    return p


def main():
    # --- Load base data and build df ---
    base = Path("/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50/")

    input_suffix = "_opioid_sample5M"
    output_suffix = "_opioid_sample5M_grace15_minspell7_ae_censoring"

    split_spells = pd.read_parquet(base / f"split_spells{output_suffix}.parquet")
    demographics = pd.read_parquet(base / f"demographics{input_suffix}.parquet")
    comorbidities = pd.read_parquet(
        base / f"icd10_codes_from_spells{output_suffix}_clustered.parquet"
    )

    # df must have columns: drug_combo (iterable of ATC codes), had_ae (bool)
    df, _, _ = final_common_preprocessing(split_spells, demographics, comorbidities, 30)

    # ATC lookup
    ndc_to_atc = pd.read_csv(
        base / "NDC_to_ATC_levels1234_clean.csv"
    )[["atc_3_code", "atc_3_name"]].drop_duplicates()

    # --- Overall AE stats ---
    overall_n = len(df)
    overall_ae = df["had_ae"].sum()
    overall_no_ae = overall_n - overall_ae
    overall_rate = overall_ae / overall_n

    print(f"Overall AE proportion: {overall_rate:.4f}\n")

    # ======================================================================
    #                              PAIRS
    # ======================================================================
    pair_counts = Counter()
    pair_ae_counts = Counter()

    for combo, had_ae in zip(df["drug_combo"], df["had_ae"]):
        if len(combo) < 2:
            continue
        for pair in combinations(sorted(combo), 2):  # unordered
            pair_counts[pair] += 1
            if had_ae:
                pair_ae_counts[pair] += 1

    min_n = 200
    rows = [
        (d1, d2, n_total, pair_ae_counts.get((d1, d2), 0))
        for (d1, d2), n_total in pair_counts.items()
        if n_total >= min_n
    ]

    pair_summary = pd.DataFrame(
        rows, columns=["drug1", "drug2", "n_total", "n_ae"]
    )
    pair_summary["n_no_ae"] = pair_summary["n_total"] - pair_summary["n_ae"]
    pair_summary["ae_prop"] = pair_summary["n_ae"] / pair_summary["n_total"]

    # χ² vs all other spells
    pair_summary["p_value"] = pair_summary.apply(
        chi2_pvalue_vs_rest,
        axis=1,
        overall_ae=overall_ae,
        overall_no_ae=overall_no_ae,
    )

    # BH correction
    rej, p_adj, _, _ = multipletests(pair_summary["p_value"], method="fdr_bh")
    pair_summary["p_adj"] = p_adj
    pair_summary["significant"] = rej  # True if FDR < 0.05

    # --- Add drug names for pairs ---
    for col in ["drug1", "drug2"]:
        pair_summary = pair_summary.merge(
            ndc_to_atc.rename(
                columns={
                    "atc_3_code": col,
                    "atc_3_name": f"{col}_name",
                }
            ),
            on=col,
            how="left",
        )

    signal_pairs = pair_summary[
        (pair_summary["ae_prop"] > overall_rate) &
        (pair_summary["significant"])
    ].copy()

    # --- Print pair results ---
    print("Top AE-proportion drug pairs (n_total >= 200):")
    print(
        pair_summary.sort_values("ae_prop", ascending=False)
        .head(10)[
            [
                "drug1", "drug1_name",
                "drug2", "drug2_name",
                "n_total", "n_ae",
                "ae_prop", "p_value", "p_adj",
            ]
        ]
        .to_string(index=False)
    )

    print("\nTop AE-occurrence drug pairs with FDR < 0.05:")
    if not signal_pairs.empty:
        print(
            signal_pairs.sort_values("n_ae", ascending=False)
            .head(10)[
                [
                    "drug1", "drug1_name",
                    "drug2", "drug2_name",
                    "n_total", "n_ae",
                    "ae_prop", "p_value", "p_adj",
                ]
            ]
            .to_string(index=False)
        )
    else:
        print("No BH-significant pairs with elevated AE rate.\n")

    # Save pair CSVs, sorting in a similar manner as the prints. First remove rows that do not have N02A in any drugx column
    drug_columns = [col for col in pair_summary.columns if col.startswith('drug')]
    pair_summary = pair_summary[pair_summary[drug_columns].apply(lambda row: row.astype(str).str.contains('N02A').any(), axis=1)]
    signal_pairs = signal_pairs[signal_pairs[drug_columns].apply(lambda row: row.astype(str).str.contains('N02A').any(), axis=1)]

    pair_summary.sort_values("ae_prop", ascending=False).to_csv("drug_pairs_ae_min200_with_BH.csv", index=False)
    signal_pairs.sort_values("n_ae", ascending=False).to_csv("drug_pairs_ae_BH_significant.csv", index=False)

    # ======================================================================
    #                              TRIOS
    # ======================================================================
    trio_counts = Counter()
    trio_ae_counts = Counter()

    for combo, had_ae in zip(df["drug_combo"], df["had_ae"]):
        if len(combo) < 3:
            continue
        for trio in combinations(sorted(combo), 3):  # unordered
            trio_counts[trio] += 1
            if had_ae:
                trio_ae_counts[trio] += 1

    rows = [
        (d1, d2, d3, n_total, trio_ae_counts.get((d1, d2, d3), 0))
        for (d1, d2, d3), n_total in trio_counts.items()
        if n_total >= min_n
    ]

    trio_summary = pd.DataFrame(
        rows, columns=["drug1", "drug2", "drug3", "n_total", "n_ae"]
    )
    trio_summary["n_no_ae"] = trio_summary["n_total"] - trio_summary["n_ae"]
    trio_summary["ae_prop"] = trio_summary["n_ae"] / trio_summary["n_total"]

    trio_summary["p_value"] = trio_summary.apply(
        chi2_pvalue_vs_rest,
        axis=1,
        overall_ae=overall_ae,
        overall_no_ae=overall_no_ae,
    )

    rej, p_adj, _, _ = multipletests(trio_summary["p_value"], method="fdr_bh")
    trio_summary["p_adj"] = p_adj
    trio_summary["significant"] = rej

    # --- Add drug names for trios ---
    for col in ["drug1", "drug2", "drug3"]:
        trio_summary = trio_summary.merge(
            ndc_to_atc.rename(
                columns={
                    "atc_3_code": col,
                    "atc_3_name": f"{col}_name",
                }
            ),
            on=col,
            how="left",
        )

    signal_trios = trio_summary[
        (trio_summary["ae_prop"] > overall_rate) &
        (trio_summary["significant"])
    ].copy()

    # --- Print trio results ---
    print("\nTop AE-proportion drug trios (n_total >= 200):")
    if not trio_summary.empty:
        print(
            trio_summary.sort_values("ae_prop", ascending=False)
            .head(10)[
                [
                    "drug1", "drug1_name",
                    "drug2", "drug2_name",
                    "drug3", "drug3_name",
                    "n_total", "n_ae",
                    "ae_prop", "p_value", "p_adj",
                ]
            ]
            .to_string(index=False)
        )
    else:
        print("No trios with n_total >= 200.\n")

    print("\nTop AE-occurrence drug trios with FDR < 0.05:")
    if not signal_trios.empty:
        print(
            signal_trios.sort_values("n_ae", ascending=False)
            .head(10)[
                [
                    "drug1", "drug1_name",
                    "drug2", "drug2_name",
                    "drug3", "drug3_name",
                    "n_total", "n_ae",
                    "ae_prop", "p_value", "p_adj",
                ]
            ]
            .to_string(index=False)
        )
    else:
        print("No BH-significant trios with elevated AE rate.\n")

    # Save trio CSVs, sorting in a similar manner as the prints. First remove rows that do not have N02A in any drugx column
    drug_columns = [col for col in trio_summary.columns if col.startswith('drug')]
    trio_summary = trio_summary[trio_summary[drug_columns].apply(lambda row: row.astype(str).str.contains('N02A').any(), axis=1)]
    signal_trios = signal_trios[signal_trios[drug_columns].apply(lambda row: row.astype(str).str.contains('N02A').any(), axis=1)]
    
    trio_summary.sort_values("ae_prop", ascending=False).to_csv("drug_trios_ae_min200_with_BH.csv", index=False)
    signal_trios.sort_values("n_ae", ascending=False).to_csv("drug_trios_ae_BH_significant.csv", index=False)


if __name__ == "__main__":
    main()
