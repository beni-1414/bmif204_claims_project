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

    parser.add_argument("--title", type=str, default="XGBoost AE30d model",
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

print("Fitting MultiLabelBinarizer for ICD10 groups...")
mlb_icd = MultiLabelBinarizer(sparse_output=True)
X_icd = mlb_icd.fit_transform(df["icd10_codes"])

print("Number of ICD10 groups:", len(mlb_icd.classes_))

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

print("Stacking feature matrices...")
# X = sparse.hstack([X_drugs, X_icd]).tocsr()
X = sparse.hstack([X_base_sparse, X_drugs, X_icd]).tocsr()
y = df["y"].values

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

single_train = False  # Set to True to do single training run without hyperparam search
if single_train:

    # ---------------------------------------------------------------------
    # 10. Train XGBoost model (simple baseline)
    # ---------------------------------------------------------------------
    print("Training XGBoost...")
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,   # keep as neg/pos for now
        max_depth=4,                         # was 5
        min_child_weight=8,                  # was 5
        learning_rate=0.05,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=8,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=50,
        # early_stopping_rounds=50
    )


    # ---------------------------------------------------------------------
    # 11. Evaluation
    # ---------------------------------------------------------------------
    from datetime import datetime


    print("Evaluating...")
    y_valid_proba = model.predict_proba(X_valid)[:, 1]

    auc_pr = average_precision_score(y_valid, y_valid_proba)
    auc_roc = roc_auc_score(y_valid, y_valid_proba)

    print(f"AUC-PR:  {auc_pr:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    # Choose an initial probability threshold (you'll tune this later)
    threshold = 0.01
    y_valid_pred = (y_valid_proba >= threshold).astype(int)

    print(f"\nClassification report at threshold = {threshold}:")
    report_str = classification_report(y_valid, y_valid_pred, digits=3)
    print(report_str)

    log_path = f"training/training_log{SUFFIX}.txt"
    with open(log_path, "a") as f:
        f.write("\n" + "="*60 + "\n")
        f.write(f"Run description: {TITLE}\n")
        f.write(f"Run time: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"SUFFIX: {SUFFIX}\n")
        f.write(f"AUC-PR:  {auc_pr:.4f}\n")
        f.write(f"AUC-ROC: {auc_roc:.4f}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}\n")
        f.write("Classification report:\n")
        f.write(report_str + "\n")

    # ---------------------------------------------------------------------
    # 12. Save model & encoders
    # ---------------------------------------------------------------------
    print("Saving model and encoders...")
    joblib.dump(model, BASE / f"xgb_ae30d_model{SUFFIX}.joblib")
    joblib.dump(mlb_drugs, BASE / "mlb_drugs.joblib")
    joblib.dump(mlb_icd, BASE / "mlb_icd.joblib")
    # joblib.dump(X_base.columns.tolist(), BASE / "base_feature_columns.joblib")

    print("Done.")

else:
    # ---------------------------------------------------------------------
    # 10. Hyperparameter sweep for XGBoost
    # ---------------------------------------------------------------------
    from datetime import datetime

    print("Starting hyperparameter search...")

    # Make sure training log directory exists
    log_dir = Path("training")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"training_log{SUFFIX}.txt"

    # Small manual grid – you can tweak these values
    param_grid = [
        {"max_depth": 3, "min_child_weight": 5,  "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 3, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "min_child_weight": 5,  "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 3, "min_child_weight": 5,  "subsample": 0.9, "colsample_bytree": 0.9},
        {"max_depth": 3, "min_child_weight": 10, "subsample": 0.9, "colsample_bytree": 0.9},
        {"max_depth": 4, "min_child_weight": 5,  "subsample": 0.9, "colsample_bytree": 0.9},
        {"max_depth": 4, "min_child_weight": 10, "subsample": 0.9, "colsample_bytree": 0.9},
    ]

    best_auc_pr = -np.inf
    best_model = None
    best_params = None
    best_y_valid_proba = None

    with open(log_path, "a") as f_log:
        f_log.write("\n" + "="*60 + "\n")
        f_log.write(f"NEW HYPERPARAM SEARCH RUN\n")
        f_log.write(f"Run description: {TITLE}\n")
        f_log.write(f"Run time: {datetime.now().isoformat(timespec='seconds')}\n")
        f_log.write(f"SUFFIX: {SUFFIX}\n")
        f_log.write(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}\n")
        f_log.write(f"scale_pos_weight: {scale_pos_weight:.4f}\n")

    for i, p in enumerate(param_grid, start=1):
        print("\n" + "-"*70)
        print(f"Config {i}/{len(param_grid)}: {p}")

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            max_depth=p["max_depth"],
            min_child_weight=p["min_child_weight"],
            learning_rate=0.05,
            n_estimators=400,
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            tree_method="hist",
            n_jobs=8,
            random_state=42,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=50,
            # early_stopping_rounds=50
        )

        # Eval for this config
        y_valid_proba = model.predict_proba(X_valid)[:, 1]
        auc_pr = average_precision_score(y_valid, y_valid_proba)
        auc_roc = roc_auc_score(y_valid, y_valid_proba)

        print(f"Config {i} AUC-PR:  {auc_pr:.4f}")
        print(f"Config {i} AUC-ROC: {auc_roc:.4f}")

        # Log this config
        with open(log_path, "a") as f_log:
            f_log.write("\nConfig {}/{}:\n".format(i, len(param_grid)))
            f_log.write(f"  params: {p}\n")
            f_log.write(f"  AUC-PR:  {auc_pr:.4f}\n")
            f_log.write(f"  AUC-ROC: {auc_roc:.4f}\n")

        # Track best
        if auc_pr > best_auc_pr:
            best_auc_pr = auc_pr
            best_model = model
            best_params = p
            best_y_valid_proba = y_valid_proba

    # ---------------------------------------------------------------------
    # 11. Final evaluation for best model
    # ---------------------------------------------------------------------
    print("\n" + "="*70)
    print("Best hyperparameters:", best_params)
    print(f"Best AUC-PR:  {best_auc_pr:.4f}")

    best_auc_roc = roc_auc_score(y_valid, best_y_valid_proba)
    print(f"Best AUC-ROC: {best_auc_roc:.4f}")

    # Choose an arbitrary initial threshold (you'll tune later)
    threshold = 0.01
    y_valid_pred = (best_y_valid_proba >= threshold).astype(int)
    print(f"\nClassification report at threshold = {threshold}:")
    report_str = classification_report(y_valid, y_valid_pred, digits=3)
    print(report_str)

    with open(log_path, "a") as f_log:
        f_log.write("\n" + "-"*70 + "\n")
        f_log.write("BEST MODEL SUMMARY\n")
        f_log.write(f"  Best params: {best_params}\n")
        f_log.write(f"  Best AUC-PR:  {best_auc_pr:.4f}\n")
        f_log.write(f"  Best AUC-ROC: {best_auc_roc:.4f}\n")
        f_log.write(f"  Threshold used for report: {threshold}\n")
        f_log.write("  Classification report:\n")
        f_log.write(report_str + "\n")

    # ---------------------------------------------------------------------
    # 12. Save best model & encoders
    # ---------------------------------------------------------------------
    print("Saving best model and encoders...")

    model_path = BASE / f"xgb_ae30d_model_best{SUFFIX}.joblib"
    joblib.dump(best_model, model_path)
    joblib.dump(mlb_drugs, BASE / "mlb_drugs.joblib")
    joblib.dump(mlb_icd, BASE / "mlb_icd.joblib")
    # joblib.dump(X_base.columns.tolist(), BASE / "base_feature_columns.joblib")

    print(f"Best model saved to: {model_path}")
    print("Done.")