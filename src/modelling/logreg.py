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

from src.modelling.final_preprocessing import final_common_preprocessing

def parse_args():
    parser = argparse.ArgumentParser(description="Extract raw SQL data for opioid study")

    parser.add_argument("--title", type=str, default="LogReg AE30d model",
                        help="Description/title for this training run")
    parser.add_argument("--db", type=int, default=5, help="Database size (1 or 5) M")
    parser.add_argument("--bootstrap", action="store_true", help="Whether to perform bootstrap CIs for ORs")


    args = parser.parse_args()
    return args

args = parse_args()
TITLE = args.title
BOOTSTRAP = args.bootstrap
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


df, numeric_cols, cat_cols = final_common_preprocessing(spells, dem, icd, AE_WINDOW_DAYS)

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
    max_iter=3000,          # adjust if you see convergence warnings
    n_jobs=8,
    class_weight="balanced",  # compensates for imbalance
    verbose=1,
)

print("Training logistic regression...")
logreg.fit(X_train, y_train)

model_path = BASE / f"logreg_model{SUFFIX}.joblib"
joblib.dump(logreg, model_path)

print(f"Saved logistic regression model to: {model_path}")


print("Evaluating logistic regression...")
y_valid_proba = logreg.predict_proba(X_valid)[:, 1]

auc_pr = average_precision_score(y_valid, y_valid_proba)
auc_roc = roc_auc_score(y_valid, y_valid_proba)

print(f"LogReg AUC-PR:  {auc_pr:.4f}")
print(f"LogReg AUC-ROC: {auc_roc:.4f}")

if BOOTSTRAP:
    # ---------------------------------------------------------------------
    # 11. Build feature names (must match X column order)
    # ---------------------------------------------------------------------
    base_feature_names = list(X_base.columns)
    drug_feature_names = [f"DRUG_{c}" for c in mlb_drugs.classes_]
    icd_feature_names  = [f"ICD_{c}" for c in icd_kept_classes]

    feature_names = base_feature_names + drug_feature_names + icd_feature_names
    assert len(feature_names) == X.shape[1], "Feature name length mismatch!"

    # ---------------------------------------------------------------------
    # 12. Bootstrap CIs for coefficients / odds ratios
    # ---------------------------------------------------------------------
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression

    B = 200
    n_train, n_features = X_train.shape
    rng_master = np.random.default_rng(42)

    print(f"Bootstrapping logistic regression coefficients with B={B}...")

    def fit_bootstrap(seed):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n_train, size=n_train)

        Xb = X_train[idx]
        yb = y_train[idx]

        # NOTE: n_jobs=1 here, we parallelize over bootstraps instead
        model = LogisticRegression(
            penalty="l2",
            solver="saga",
            max_iter=1000,
            n_jobs=1,
            class_weight="balanced",
            verbose=0,
        )
        model.fit(Xb, yb)
        return model.coef_.ravel()

    seeds = rng_master.integers(0, 1_000_000_000, size=B)

    # Use n_jobs equal to number of cores on the node, e.g. 8
    coefs_list = Parallel(n_jobs=8, verbose=10)(
        delayed(fit_bootstrap)(int(s)) for s in seeds
    )

    coefs_boot = np.vstack(coefs_list).astype(np.float32)

    print("Finished bootstrapping.")

    # ---------------------------------------------------------------------
    # 13. Summarize ORs and 95% CIs
    # ---------------------------------------------------------------------
    coef_orig = logreg.coef_.ravel()  # from your original model fit
    coef_mean = coefs_boot.mean(axis=0)
    coef_lower = np.percentile(coefs_boot, 2.5, axis=0)
    coef_upper = np.percentile(coefs_boot, 97.5, axis=0)

    or_df = pd.DataFrame({
        "feature": feature_names,
        "coef_orig": coef_orig,
        "coef_boot_mean": coef_mean,
        "OR": np.exp(coef_orig),
        "OR_boot_mean": np.exp(coef_mean),
        "CI_lower": np.exp(coef_lower),
        "CI_upper": np.exp(coef_upper),
    })

    # Sort by original OR, descending
    or_df = or_df.sort_values("OR", ascending=False)

    ci_path = BASE / f"logreg_or_bootstrap_ci.parquet"
    or_df.to_parquet(ci_path, index=False)
    print(f"Saved bootstrap OR CI table to: {ci_path}")