from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report
)

from scipy import sparse
from xgboost import XGBClassifier
import joblib
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Extract raw SQL data for opioid study")

    parser.add_argument("--title", type=str, default="LogReg AE30d model",
                        help="Description/title for this training run")
    parser.add_argument("--db", type=int, default=5, help="Database size (1 or 5) M")


    args = parser.parse_args()
    return args

args = parse_args()
TITLE = args.title
# ---------------------------------------------------------------------
# 1. Paths & constants
# ---------------------------------------------------------------------
SUFFIX = f"_opioid_sample{args.db}M_grace15_minspell7_ae_censoring"
BASE = Path("/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50")

demographics_path = BASE / f"demographics_opioid_sample{args.db}M.parquet"
split_spells_path = BASE / f"split_spells{SUFFIX}.parquet"
icd10_path = BASE / f"icd10_codes_from_spells{SUFFIX}_clustered.parquet"

AE_WINDOW_DAYS = 30

# ---------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------
print("Loading data...")
dem = pd.read_parquet(demographics_path)
spells = pd.read_parquet(split_spells_path)
icd = pd.read_parquet(icd10_path)

print(f"dem shape:    {dem.shape}")
print(f"spells shape: {spells.shape}")
print(f"icd shape:    {icd.shape}")

# ---------------------------------------------------------------------
# 3. Basic cleaning / filters
# ---------------------------------------------------------------------

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
print("AE within 30 days rate after dedup:", spells["y"].mean())


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

# ---------------------------------------------------------------------
# 6. Simple feature engineering - SKIP FOR NOW
# ---------------------------------------------------------------------
# Age at spell entry
df["age"] = df["entry_date"].dt.year - df["birthyear"]

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

# ---------------------------------------------------------------------
# 7. Multi-hot encode drugs & ICD10 groups
# ---------------------------------------------------------------------
# Ensure icd10_codes is list-like; replace NaNs with empty list
def to_list_or_empty(x):
    # already a proper list/tuple
    if isinstance(x, (list, tuple)):
        return list(x)
    
    # numpy array from parquet
    if isinstance(x, np.ndarray):
        return x.tolist()
    
    # explicit None
    if x is None:
        return []
    
    # pandas NA / NaN scalars
    try:
        if pd.isna(x):
            return []
    except TypeError:
        # pd.isna on weird types (like arrays) would error,
        # but we've already handled arrays above.
        pass
    
    # fallback: treat as single code
    return [x]

df["icd10_codes"] = df["icd10_codes"].apply(to_list_or_empty)

# drug_combo is already list-like from earlier filter
df["drug_combo"] = df["drug_combo"].apply(lambda x: list(x))

df["num_drugs_in_spell"] = df["drug_combo"].apply(len)
df["num_icd10_codes"] = df["icd10_codes"].apply(len)
numeric_cols.extend(["num_drugs_in_spell", "num_icd10_codes"])

print("Fitting MultiLabelBinarizer for drugs...")
mlb_drugs = MultiLabelBinarizer(sparse_output=True)
X_drugs = mlb_drugs.fit_transform(df["drug_combo"])

print("Number of drug classes:", len(mlb_drugs.classes_))

# ---------------------------------------------------------------------
# 7b. Multi-hot encode ICD10 groups, but keep only frequent ones
# ---------------------------------------------------------------------

print("Fitting MultiLabelBinarizer for ICD10 groups...")
mlb_icd = MultiLabelBinarizer(sparse_output=True)
X_icd_full = mlb_icd.fit_transform(df["icd10_codes"])
icd_classes = np.array(mlb_icd.classes_)

print("Total ICD10 group features:", len(icd_classes))

# Frequency filter: keep ICD groups that appear in at least min_icd_freq spells
min_icd_freq = 500   # you can tune this (e.g., 200, 1000, etc.)
icd_freq = np.asarray(X_icd_full.sum(axis=0)).ravel()
keep_mask = icd_freq >= min_icd_freq

print(f"ICD groups kept (freq >= {min_icd_freq}): {keep_mask.sum()} out of {len(icd_classes)}")

X_icd = X_icd_full[:, keep_mask]
icd_kept_classes = icd_classes[keep_mask]

# optional: keep for inspection / saving
df["num_icd10_codes"] = df["icd10_codes"].apply(len)
numeric_cols.extend(["num_drugs_in_spell", "num_icd10_codes"])


# ---------------------------------------------------------------------
# 8. One-hot encode demographics & combine all features
# ---------------------------------------------------------------------
print("One-hot encoding demographics...")
X_base = df[numeric_cols + cat_cols].copy()

# Fill numeric NaNs with median
for c in numeric_cols:
    X_base[c] = X_base[c].fillna(X_base[c].median())

X_base = pd.get_dummies(X_base, columns=cat_cols, dummy_na=True)

# Make sure everything is numeric and not object
X_base = X_base.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")

# Convert base to sparse and hstack with drugs & icd
X_base_sparse = sparse.csr_matrix(X_base.values)

print("Stacking feature matrices for logreg...")
X = sparse.hstack([X_base_sparse, X_drugs, X_icd]).tocsr()
y = df["y"].values

print("Final feature matrix shape:", X.shape)

print("Final feature matrix shape:", X.shape)

# ---------------------------------------------------------------------
# 9. Train/validation split (stratified)
# ---------------------------------------------------------------------
print("Splitting train/validation...")
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

pos = y_train.sum()
neg = len(y_train) - pos
scale_pos_weight = neg / pos
print(f"Train positives: {pos}, negatives: {neg}, scale_pos_weight: {scale_pos_weight:.2f}")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report

# ---------------------------------------------------------------------
# 10. Logistic regression (penalized, sparse, scalable)
# ---------------------------------------------------------------------

pos = y_train.sum()
neg = len(y_train) - pos
print(f"Train positives: {pos}, negatives: {neg}")

# You can try both L2 and elastic-net; here's L2 first (faster, stable)
logreg = LogisticRegression(
    penalty="l2",
    solver="saga",
    max_iter=200,          # adjust if you see convergence warnings
    n_jobs=8,
    class_weight="balanced",  # compensates for imbalance
    verbose=1,
)

print("Training logistic regression...")
logreg.fit(X_train, y_train)

print("Evaluating logistic regression...")
y_valid_proba = logreg.predict_proba(X_valid)[:, 1]

auc_pr = average_precision_score(y_valid, y_valid_proba)
auc_roc = roc_auc_score(y_valid, y_valid_proba)

print(f"LogReg AUC-PR:  {auc_pr:.4f}")
print(f"LogReg AUC-ROC: {auc_roc:.4f}")

threshold = 0.01
y_valid_pred = (y_valid_proba >= threshold).astype(int)

print(f"\nLogReg classification report at threshold = {threshold}:")
print(classification_report(y_valid, y_valid_pred, digits=3))


# Build feature name list in the same order as X
base_feature_names = list(X_base.columns)
drug_feature_names = [f"DRUG_{c}" for c in mlb_drugs.classes_]
icd_feature_names  = [f"ICD_{c}" for c in icd_kept_classes]

feature_names = base_feature_names + drug_feature_names + icd_feature_names

coef = logreg.coef_.ravel()
logreg_summary = pd.DataFrame({
    "feature": feature_names,
    "coef": coef,
    "OR": np.exp(coef)
}).sort_values("OR", ascending=False)

print("Top positive features (highest OR):")
print(logreg_summary.head(50))

print("Top negative features (protective):")
print(logreg_summary.tail(50))
