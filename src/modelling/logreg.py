from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import joblib

from src.modelling.final_preprocessing import final_common_preprocessing


# ---------------------------------------------------------------------
# 0. CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train AE model (LogReg)")

    parser.add_argument(
        "--title",
        type=str,
        default="LogReg AE30d model",
        help="Description/title for this training run",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_FINAL",
        help="Suffix for input files (e.g. '_FINAL')",
    )
    parser.add_argument(
        "--db",
        type=int,
        default=5,
        help="Database size (1 or 5) M, used in file names",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Whether to perform bootstrap CIs for ORs",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50"),
        help="Base path for input/output files",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def to_list_or_empty(x):
    """Normalize various representations of codes into a Python list."""
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


# ---------------------------------------------------------------------
# 7–7b–8–9: feature construction + split
# ---------------------------------------------------------------------
def prepare_features_and_split(
    df: pd.DataFrame,
    numeric_cols: list,
    cat_cols: list,
    min_icd_freq: int = 500,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Steps 7–7b–8–9:
      - Multi-hot encode drugs and ICD10 groups (with frequency filter).
      - One-hot encode demographics and other categorical features.
      - Combine into a sparse design matrix X and label vector y.
      - Do a stratified train/validation split.

    Returns:
      X_train, X_valid, y_train, y_valid,
      X_base (dense DataFrame used as base),
      mlb_drugs (fitted MultiLabelBinarizer),
      icd_kept_classes (array of ICD codes kept after freq filter)
    """
    # 7. Multi-hot encode drugs & ICD10 groups
    # ---------------------------------------
    df = df.copy()

    # Ensure icd10_codes is list-like; replace NaNs with empty list
    df["icd10_codes"] = df["icd10_codes"].apply(to_list_or_empty)

    # drug_combo is already list-like from earlier filter (ensure list)
    df["drug_combo"] = df["drug_combo"].apply(lambda x: list(x))

    df["num_drugs_in_spell"] = df["drug_combo"].apply(len)
    df["num_icd10_codes"] = df["icd10_codes"].apply(len)
    numeric_cols = list(numeric_cols) + ["num_drugs_in_spell", "num_icd10_codes"]

    print("Fitting MultiLabelBinarizer for drugs...")
    mlb_drugs = MultiLabelBinarizer(sparse_output=True)
    X_drugs = mlb_drugs.fit_transform(df["drug_combo"])
    print("Number of drug classes:", len(mlb_drugs.classes_))

    # 7b. Multi-hot encode ICD10 groups, with frequency filter
    # --------------------------------------------------------
    print("Fitting MultiLabelBinarizer for ICD10 groups...")
    mlb_icd = MultiLabelBinarizer(sparse_output=True)
    X_icd_full = mlb_icd.fit_transform(df["icd10_codes"])
    icd_classes = np.array(mlb_icd.classes_)

    print("Total ICD10 group features:", len(icd_classes))

    icd_freq = np.asarray(X_icd_full.sum(axis=0)).ravel()
    keep_mask = icd_freq >= min_icd_freq

    print(
        f"ICD groups kept (freq >= {min_icd_freq}): "
        f"{keep_mask.sum()} out of {len(icd_classes)}"
    )

    X_icd = X_icd_full[:, keep_mask]
    icd_kept_classes = icd_classes[keep_mask]

    # 8. One-hot encode demographics & combine all features
    # -----------------------------------------------------
    print("One-hot encoding demographics & base features...")
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
    X = sparse.hstack([X_base_sparse, X_drugs, X_icd]).tocsr()
    y = df["y"].values

    print("Final feature matrix shape:", X.shape)

    # 9. Train/validation split (stratified)
    # --------------------------------------
    print("Splitting train/validation...")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos
    print(
        f"Train positives: {pos}, negatives: {neg}, "
        f"scale_pos_weight: {scale_pos_weight:.2f}"
    )

    return (
        X_train,
        X_valid,
        y_train,
        y_valid,
        X_base,
        mlb_drugs,
        icd_kept_classes,
    )


# ---------------------------------------------------------------------
# Bootstrap helper
# ---------------------------------------------------------------------
def bootstrap_logreg_ORs(
    logreg_model: LogisticRegression,
    X_train,
    y_train,
    X_base: pd.DataFrame,
    mlb_drugs: MultiLabelBinarizer,
    icd_kept_classes: np.ndarray,
    n_bootstrap: int = 200,
    n_jobs: int = 8,
    random_state: int = 42,
):
    """
    Bootstrap logistic regression coefficients to obtain OR CIs.
    Returns a DataFrame with ORs and 95% CIs.
    """
    from joblib import Parallel, delayed

    n_train, n_features = X_train.shape
    rng_master = np.random.default_rng(random_state)

    base_feature_names = list(X_base.columns)
    drug_feature_names = [f"DRUG_{c}" for c in mlb_drugs.classes_]
    icd_feature_names = [f"ICD_{c}" for c in icd_kept_classes]
    feature_names = base_feature_names + drug_feature_names + icd_feature_names
    assert len(feature_names) == n_features, "Feature name length mismatch!"

    print(f"Bootstrapping logistic regression coefficients with B={n_bootstrap}...")

    def fit_bootstrap(seed):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n_train, size=n_train)

        Xb = X_train[idx]
        yb = y_train[idx]

        model = LogisticRegression(
            penalty="l2",
            solver="saga",
            max_iter=1000,
            n_jobs=1,  # parallelization is over bootstraps, not inside solver
            class_weight="balanced",
            verbose=0,
        )
        model.fit(Xb, yb)
        return model.coef_.ravel()

    seeds = rng_master.integers(0, 1_000_000_000, size=n_bootstrap)

    coefs_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(fit_bootstrap)(int(s)) for s in seeds
    )
    coefs_boot = np.vstack(coefs_list).astype(np.float32)

    print("Finished bootstrapping.")

    coef_orig = logreg_model.coef_.ravel()
    coef_mean = coefs_boot.mean(axis=0)
    coef_lower = np.percentile(coefs_boot, 2.5, axis=0)
    coef_upper = np.percentile(coefs_boot, 97.5, axis=0)

    or_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coef_orig": coef_orig,
            "coef_boot_mean": coef_mean,
            "OR": np.exp(coef_orig),
            "OR_boot_mean": np.exp(coef_mean),
            "CI_lower": np.exp(coef_lower),
            "CI_upper": np.exp(coef_upper),
        }
    ).sort_values("OR", ascending=False)

    return or_df


# ---------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    TITLE = args.title
    BOOTSTRAP = args.bootstrap

    # 1. Paths & constants
    SUFFIX = args.suffix
    BASE = args.path

    demographics_path = BASE / f"demographics_opioid_sample{args.db}M.parquet"
    split_spells_path = BASE / f"split_spells{SUFFIX}.parquet"
    icd10_path = BASE / f"icd10_codes_from_spells{SUFFIX}_clustered.parquet"

    AE_WINDOW_DAYS = 30

    # 2. Load data
    print("Loading data...")
    dem = pd.read_parquet(demographics_path)
    spells = pd.read_parquet(split_spells_path)
    icd = pd.read_parquet(icd10_path)

    df, numeric_cols, cat_cols = final_common_preprocessing(
        spells, dem, icd, AE_WINDOW_DAYS
    )

    # 7–9. Features + split
    (
        X_train,
        X_valid,
        y_train,
        y_valid,
        X_base,
        mlb_drugs,
        icd_kept_classes,
    ) = prepare_features_and_split(df, numeric_cols, cat_cols)

    # 10. Logistic regression
    pos = y_train.sum()
    neg = len(y_train) - pos
    print(f"Train positives: {pos}, negatives: {neg}")

    logreg = LogisticRegression(
        penalty="l2",
        solver="saga",
        max_iter=3000,  # adjust if convergence warnings arise
        n_jobs=8,
        class_weight="balanced",  # compensates for imbalance
        verbose=1,
    )

    print("Training logistic regression...")
    logreg.fit(X_train, y_train)

    model_path = BASE / f"logreg_model{SUFFIX}.joblib"
    joblib.dump(logreg, model_path)
    print(f"Saved logistic regression model to: {model_path}")

    # Evaluation
    print("Evaluating logistic regression...")
    y_valid_proba = logreg.predict_proba(X_valid)[:, 1]

    auc_pr = average_precision_score(y_valid, y_valid_proba)
    auc_roc = roc_auc_score(y_valid, y_valid_proba)

    print(f"Run title: {TITLE}")
    print(f"LogReg AUC-PR:  {auc_pr:.4f}")
    print(f"LogReg AUC-ROC: {auc_roc:.4f}")

    # Optional bootstrap
    if BOOTSTRAP:
        or_df = bootstrap_logreg_ORs(
            logreg_model=logreg,
            X_train=X_train,
            y_train=y_train,
            X_base=X_base,
            mlb_drugs=mlb_drugs,
            icd_kept_classes=icd_kept_classes,
            n_bootstrap=200,
            n_jobs=8,
            random_state=42,
        )
        ci_path = BASE / f"logreg_or_bootstrap_ci.parquet"
        or_df.to_parquet(ci_path, index=False)
        print(f"Saved bootstrap OR CI table to: {ci_path}")


if __name__ == "__main__":
    main()
