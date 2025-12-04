from pathlib import Path
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
)

from xgboost import XGBClassifier
import joblib

from src.modelling.final_preprocessing import final_common_preprocessing


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost AE30d model")

    parser.add_argument(
        "--title",
        type=str,
        default="XGBoost AE30d model",
        help="Description/title for this training run",
    )
    parser.add_argument(
        "--db",
        type=int,
        default=5,
        help="Database size (1 or 5) M",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_FINAL",
        help="Suffix for input files",
    )
    parser.add_argument(
        "--single_train",
        action="store_true",
        help="If set, do a single training run instead of hyperparameter search",
    )
    parser.add_argument(
        "--pos-neg-optimize",
        action="store_true",
        help="If set, optimize only the pos-neg weight ratio",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50"),
        help="Base path for input/output files",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------
def to_list_or_empty(x):
    """Normalize various representations of codes into a Python list."""
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
        # pd.isna on weird types (like arrays) would error; arrays handled above
        pass

    return [x]


def prepare_features_and_split(
    df: pd.DataFrame,
    numeric_cols: list,
    cat_cols: list,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Steps 7–8–9:
      - Multi-hot encode drugs & ICD10 groups.
      - One-hot encode demographics & categorical features.
      - Combine into sparse X and y.
      - Stratified train/validation split.

    Returns:
      X_train, X_valid, y_train, y_valid,
      X_base (dense base feature frame),
      mlb_drugs (fitted MultiLabelBinarizer),
      mlb_icd (fitted MultiLabelBinarizer)
    """
    df = df.copy()

    # 7. Multi-hot encode drugs & ICD10 groups
    # ---------------------------------------
    df["icd10_codes"] = df["icd10_codes"].apply(to_list_or_empty)
    df["drug_combo"] = df["drug_combo"].apply(lambda x: list(x))

    df["num_drugs_in_spell"] = df["drug_combo"].apply(len)
    df["num_icd10_codes"] = df["icd10_codes"].apply(len)
    numeric_cols = list(numeric_cols) + ["num_drugs_in_spell", "num_icd10_codes"]

    print("Fitting MultiLabelBinarizer for drugs...")
    mlb_drugs = MultiLabelBinarizer(sparse_output=True)
    X_drugs = mlb_drugs.fit_transform(df["drug_combo"])
    print("Number of drug classes:", len(mlb_drugs.classes_))

    print("Fitting MultiLabelBinarizer for ICD10 groups...")
    mlb_icd = MultiLabelBinarizer(sparse_output=True)
    X_icd = mlb_icd.fit_transform(df["icd10_codes"])
    print("Number of ICD10 groups:", len(mlb_icd.classes_))

    # 8. One-hot encode demographics & combine all features
    # -----------------------------------------------------
    print("One-hot encoding demographics & base features...")
    X_base = df[numeric_cols + cat_cols].copy()

    for c in numeric_cols:
        X_base[c] = X_base[c].fillna(X_base[c].median())

    X_base = pd.get_dummies(X_base, columns=cat_cols, dummy_na=True)
    X_base = X_base.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")

    X_base_sparse = sparse.csr_matrix(X_base.values)

    print("Stacking feature matrices...")
    X = sparse.hstack([X_base_sparse, X_drugs, X_icd]).tocsr()
    y = df["y"].values

    print("Final feature matrix shape:", X.shape)

    # 9. Train/validation split
    # -------------------------
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

    return X_train, X_valid, y_train, y_valid, X_base, mlb_drugs, mlb_icd


# ---------------------------------------------------------------------
# Training modes
# ---------------------------------------------------------------------
def train_xgb_single_run(
    X_train,
    X_valid,
    y_train,
    y_valid,
    scale_pos_weight: float,
) -> tuple[XGBClassifier, np.ndarray, float, float]:
    """Single XGBoost training run with fixed hyperparameters."""
    print("Training XGBoost (single run)...")
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=0.2 * scale_pos_weight,
        max_depth=4,
        min_child_weight=8,
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
        # early_stopping_rounds=50,
    )

    print("Evaluating single-run model...")
    y_valid_proba = model.predict_proba(X_valid)[:, 1]
    auc_pr = average_precision_score(y_valid, y_valid_proba)
    auc_roc = roc_auc_score(y_valid, y_valid_proba)

    print(f"AUC-PR:  {auc_pr:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    return model, y_valid_proba, auc_pr, auc_roc


def train_xgb_posneg_optimize(
    X_train,
    X_valid,
    y_train,
    y_valid,
    base_scale_pos_weight: float,
) -> tuple[XGBClassifier, np.ndarray, dict, float, float]:
    """Hyperparameter sweep over scale_pos_weight only."""
    print("Starting pos-neg weight hyperparameter search...")

    param_grid = [{"scale_pos_weight": scale} for scale in [0.05, 0.1, 0.15, 0.2]]

    best_auc_pr = -np.inf
    best_model = None
    best_params = None
    best_y_valid_proba = None

    for i, p in enumerate(param_grid, start=1):
        print("\n" + "-" * 70)
        print(f"Config {i}/{len(param_grid)}: {p}")
        eff_scale = base_scale_pos_weight * p["scale_pos_weight"]
        print(f"Using scale_pos_weight: {eff_scale:.4f}")

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=eff_scale,
            max_depth=4,
            min_child_weight=8,
            learning_rate=0.05,
            n_estimators=400,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=8,
            random_state=42,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=50,
            # early_stopping_rounds=50,
        )

        y_valid_proba = model.predict_proba(X_valid)[:, 1]
        auc_pr = average_precision_score(y_valid, y_valid_proba)
        auc_roc = roc_auc_score(y_valid, y_valid_proba)

        print(f"Config {i} AUC-PR:  {auc_pr:.4f}")
        print(f"Config {i} AUC-ROC: {auc_roc:.4f}")

        if auc_pr > best_auc_pr:
            best_auc_pr = auc_pr
            best_model = model
            best_params = p
            best_y_valid_proba = y_valid_proba

    print("\n" + "=" * 70)
    print("Best hyperparameters (scale only):", best_params)
    print(f"Best AUC-PR:  {best_auc_pr:.4f}")
    best_auc_roc = roc_auc_score(y_valid, best_y_valid_proba)
    print(f"Best AUC-ROC: {best_auc_roc:.4f}")

    threshold = 0.01
    y_valid_pred = (best_y_valid_proba >= threshold).astype(int)
    print(f"\nClassification report at threshold = {threshold}:")
    report_str = classification_report(y_valid, y_valid_pred, digits=3)
    print(report_str)

    return best_model, best_y_valid_proba, best_params, best_auc_pr, best_auc_roc


def train_xgb_full_search(
    X_train,
    X_valid,
    y_train,
    y_valid,
    scale_pos_weight: float,
    title: str,
    suffix: str,
    log_dir: Path | None = None,
) -> tuple[XGBClassifier, np.ndarray, dict, float, float]:
    """Full hyperparameter search over depth, min_child_weight, subsample, colsample."""
    print("Starting full hyperparameter search...")

    if log_dir is None:
        log_dir = Path("training")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"training_log{suffix}.txt"

    param_grid = [
        {"max_depth": 3, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 3, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 3, "min_child_weight": 5, "subsample": 0.9, "colsample_bytree": 0.9},
        {"max_depth": 3, "min_child_weight": 10, "subsample": 0.9, "colsample_bytree": 0.9},
        {"max_depth": 4, "min_child_weight": 5, "subsample": 0.9, "colsample_bytree": 0.9},
        {"max_depth": 4, "min_child_weight": 10, "subsample": 0.9, "colsample_bytree": 0.9},
    ]

    best_auc_pr = -np.inf
    best_model = None
    best_params = None
    best_y_valid_proba = None

    with open(log_path, "a") as f_log:
        f_log.write("\n" + "=" * 60 + "\n")
        f_log.write("NEW HYPERPARAM SEARCH RUN\n")
        f_log.write(f"Run description: {title}\n")
        f_log.write(f"Run time: {datetime.now().isoformat(timespec='seconds')}\n")
        f_log.write(f"SUFFIX: {suffix}\n")
        f_log.write(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}\n")
        f_log.write(f"scale_pos_weight: {scale_pos_weight:.4f}\n")

    for i, p in enumerate(param_grid, start=1):
        print("\n" + "-" * 70)
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
            # early_stopping_rounds=50,
        )

        y_valid_proba = model.predict_proba(X_valid)[:, 1]
        auc_pr = average_precision_score(y_valid, y_valid_proba)
        auc_roc = roc_auc_score(y_valid, y_valid_proba)

        print(f"Config {i} AUC-PR:  {auc_pr:.4f}")
        print(f"Config {i} AUC-ROC: {auc_roc:.4f}")

        with open(log_path, "a") as f_log:
            f_log.write("\nConfig {}/{}:\n".format(i, len(param_grid)))
            f_log.write(f"  params: {p}\n")
            f_log.write(f"  AUC-PR:  {auc_pr:.4f}\n")
            f_log.write(f"  AUC-ROC: {auc_roc:.4f}\n")

        if auc_pr > best_auc_pr:
            best_auc_pr = auc_pr
            best_model = model
            best_params = p
            best_y_valid_proba = y_valid_proba

    print("\n" + "=" * 70)
    print("Best hyperparameters:", best_params)
    print(f"Best AUC-PR:  {best_auc_pr:.4f}")
    best_auc_roc = roc_auc_score(y_valid, best_y_valid_proba)
    print(f"Best AUC-ROC: {best_auc_roc:.4f}")

    threshold = 0.01
    y_valid_pred = (best_y_valid_proba >= threshold).astype(int)
    print(f"\nClassification report at threshold = {threshold}:")
    report_str = classification_report(y_valid, y_valid_pred, digits=3)
    print(report_str)

    return best_model, best_y_valid_proba, best_params, best_auc_pr, best_auc_roc


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    TITLE = args.title
    SINGLE_TRAIN = args.single_train
    POS_NEG_OPTIMIZE = args.pos_neg_optimize

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
        X_base,      # noqa: F841
        mlb_drugs,   # noqa: F841 (kept for possible future use)
        mlb_icd,     # noqa: F841
    ) = prepare_features_and_split(df, numeric_cols, cat_cols)

    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos
    print(f"Train positives: {pos}, negatives: {neg}")

    # Choose training mode
    if SINGLE_TRAIN:
        best_model, best_y_valid_proba, best_auc_pr, best_auc_roc = train_xgb_single_run(
            X_train, X_valid, y_train, y_valid, scale_pos_weight
        )
    elif POS_NEG_OPTIMIZE:
        (
            best_model,
            best_y_valid_proba,
            best_params,
            best_auc_pr,
            best_auc_roc,
        ) = train_xgb_posneg_optimize(
            X_train, X_valid, y_train, y_valid, scale_pos_weight
        )
    else:
        (
            best_model,
            best_y_valid_proba,
            best_params,
            best_auc_pr,
            best_auc_roc,
        ) = train_xgb_full_search(
            X_train,
            X_valid,
            y_train,
            y_valid,
            scale_pos_weight=scale_pos_weight,
            title=TITLE,
            suffix=SUFFIX,
        )

    # 12. Save model
    print("Saving model...")
    model_path = BASE / f"xgb_ae30d_model{SUFFIX}.joblib"
    joblib.dump(best_model, model_path)
    print(f"Saved best model to: {model_path}")
    print("Done.")


if __name__ == "__main__":
    main()
