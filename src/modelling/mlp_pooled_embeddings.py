# train_mlp_medtok_pooled.py

import argparse
from pathlib import Path
import json
import math
import copy
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    f1_score,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.modelling.final_preprocessing import final_common_preprocessing

# ---------------------------------------------------------------------
# 0. HARD-CODED PARAMETERS
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP AE30d model with MedTok embeddings")

    parser.add_argument("--title", type=str, default="MLP AE30d model with MedTok embeddings (pooled)",
                        help="Description/title for this training run")
    parser.add_argument("--suffix", type=str, default="_FINAL", help="Suffix for input files")
    parser.add_argument("--db", type=int, default=5, help="Database size (1 or 5) M")

    args = parser.parse_args()
    return args


args = parse_args()
TITLE = args.title
db_size = args.db

SUFFIX = args.suffix
BASE = Path("/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50")

demographics_path = BASE / f"demographics_opioid_sample{db_size}M.parquet"
split_spells_path = BASE / f"split_spells{SUFFIX}.parquet"
icd10_path = BASE / f"icd10_codes_from_spells{SUFFIX}_clustered.parquet"
code2emb_path = BASE / "code2embeddings.json"

AE_WINDOW_DAYS = 30

# model / training hyperparameters
MAX_DIAG_LEN = 50
MAX_DRUG_LEN = 15
BATCH_SIZE = 512
EPOCHS = 12
LR = 1e-3
HIDDEN_DIMS = (256, 128)
DROPOUT = 0.4
EARLY_STOPPING_PATIENCE = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# ---------------------------------------------------------------------
# 1. SMALL HELPERS
# ---------------------------------------------------------------------

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


def bucket_age(age):
    """ETHOS-style coarse age buckets; tweak bins as needed."""
    if age < 25:
        return "18-24"
    elif age < 35:
        return "25-34"
    elif age < 45:
        return "35-44"
    elif age < 55:
        return "45-54"
    elif age < 65:
        return "55-64"
    elif age < 75:
        return "65-74"
    else:
        return "75+"


def build_code_vocab(series_of_lists):
    """Builds a vocab with PAD=0, UNK=1, then sorted codes from data."""
    code_set = set()
    for lst in series_of_lists:
        for code in lst:
            if code is None:
                continue
            code_set.add(str(code))
    code_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    idx = 2
    for code in sorted(code_set):
        code_to_idx[code] = idx
        idx += 1
    return code_to_idx


# ---------------------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------------------

print("Loading data...")
dem = pd.read_parquet(demographics_path)
spells = pd.read_parquet(split_spells_path)
icd = pd.read_parquet(icd10_path)

df, _, _ = final_common_preprocessing(spells, dem, icd, AE_WINDOW_DAYS)

# Ensure the sequence columns are list-like
df["icd10_codes"] = df["icd10_codes"].apply(to_list_or_empty)
df["drug_combo"] = df["drug_combo"].apply(to_list_or_empty)

# ---------------------------------------------------------------------
# 3. LOAD MEDTOK EMBEDDINGS
# ---------------------------------------------------------------------

print(f"Loading MedTok embeddings from {code2emb_path} ...")
with open(code2emb_path, "r") as f:
    code2emb_raw = json.load(f)

code2emb = {str(k): np.array(v, dtype=np.float32) for k, v in code2emb_raw.items()}
embed_dim = len(next(iter(code2emb.values())))
print(f"Loaded {len(code2emb)} MedTok codes, embedding dim = {embed_dim}")


# ---------------------------------------------------------------------
# 4. VOCABS FOR ICD, DRUGS, AGE, RACE, GENDER
# ---------------------------------------------------------------------

print("Building vocabularies...")

icd_code_to_idx = build_code_vocab(df["icd10_codes"])
drug_code_to_idx = build_code_vocab(df["drug_combo"])

print(f"Number of ICD tokens (incl PAD/UNK): {len(icd_code_to_idx)}")
print(f"Number of drug tokens (incl PAD/UNK): {len(drug_code_to_idx)}")

# age bins
df["age_bin"] = df["age"].apply(bucket_age)
age_bins = sorted(df["age_bin"].unique())
agebin_to_idx = {ab: i for i, ab in enumerate(age_bins)}
print("Age bins:", age_bins)

# race / gender
df["raceethnicitytypecode"] = df["raceethnicitytypecode"].fillna("UNK").astype(str)
df["gendercode"] = df["gendercode"].fillna("UNK").astype(str)

race_values = sorted(df["raceethnicitytypecode"].unique())
gender_values = sorted(df["gendercode"].unique())

race_to_idx = {r: i for i, r in enumerate(race_values)}
gender_to_idx = {g: i for i, g in enumerate(gender_values)}

print("Race categories:", race_values)
print("Gender categories:", gender_values)

# id columns
df["age_bin_id"] = df["age_bin"].map(agebin_to_idx)
df["race_id"] = df["raceethnicitytypecode"].map(race_to_idx)
df["gender_id"] = df["gendercode"].map(gender_to_idx)


# ---------------------------------------------------------------------
# 5. TRAIN / VALID SPLIT
# ---------------------------------------------------------------------

y = df["y"].values
X_train_df, X_valid_df, y_train, y_valid = train_test_split(
    df,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

pos = y_train.sum()
neg = len(y_train) - pos

raw_pos_weight = neg / pos
pos_weight_value = min(raw_pos_weight, 50.0)  # full ratio but capped
print(
    f"Train positives: {pos}, negatives: {neg}, "
    f"raw_pos_weight: {raw_pos_weight:.2f}, used_pos_weight: {pos_weight_value:.2f}"
)


# ---------------------------------------------------------------------
# 6. BUILD EMBEDDING MATRICES FOR ICD & DRUGS
# ---------------------------------------------------------------------

def build_embedding_matrix(vocab, embed_dim, code2emb, pad_token=PAD_TOKEN, unk_token=UNK_TOKEN):
    """Returns a (V, D) embedding matrix for this vocab using code2emb when available."""
    V = len(vocab)
    emb_matrix = np.zeros((V, embed_dim), dtype=np.float32)

    # UNK: random normal
    unk_idx = vocab[unk_token]
    emb_matrix[unk_idx] = np.random.normal(scale=0.01, size=(embed_dim,))

    for code, idx in vocab.items():
        if code in (pad_token, unk_token):
            continue
        vec = code2emb.get(code)
        if vec is not None and vec.shape[0] == embed_dim:
            emb_matrix[idx] = vec
        else:
            # random init if not found or wrong dimension
            emb_matrix[idx] = np.random.normal(scale=0.01, size=(embed_dim,))
    return emb_matrix


print("Building ICD and drug embedding matrices from MedTok...")
icd_emb_matrix = build_embedding_matrix(icd_code_to_idx, embed_dim, code2emb)
drug_emb_matrix = build_embedding_matrix(drug_code_to_idx, embed_dim, code2emb)


# ---------------------------------------------------------------------
# 7. DATASET AND DATALOADERS
# ---------------------------------------------------------------------

class ClaimsSequenceDataset(Dataset):
    def __init__(self, df, icd_code_to_idx, drug_code_to_idx,
                 max_diag_len, max_drug_len):
        self.df = df.reset_index(drop=True)
        self.icd_code_to_idx = icd_code_to_idx
        self.drug_code_to_idx = drug_code_to_idx
        self.max_diag_len = max_diag_len
        self.max_drug_len = max_drug_len

    def _encode_seq(self, codes, vocab, max_len):
        ids = []
        for c in codes[:max_len]:
            c_str = str(c)
            ids.append(vocab.get(c_str, vocab[UNK_TOKEN]))
        if len(ids) < max_len:
            ids.extend([vocab[PAD_TOKEN]] * (max_len - len(ids)))
        return np.array(ids, dtype=np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        diag_ids = self._encode_seq(row["icd10_codes"],
                                    self.icd_code_to_idx,
                                    self.max_diag_len)
        drug_ids = self._encode_seq(row["drug_combo"],
                                    self.drug_code_to_idx,
                                    self.max_drug_len)

        age_bin_id = int(row["age_bin_id"])
        race_id = int(row["race_id"])
        gender_id = int(row["gender_id"])
        label = int(row["y"])

        return (
            torch.from_numpy(diag_ids),
            torch.from_numpy(drug_ids),
            torch.tensor(age_bin_id, dtype=torch.long),
            torch.tensor(race_id, dtype=torch.long),
            torch.tensor(gender_id, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


train_dataset = ClaimsSequenceDataset(
    X_train_df,
    icd_code_to_idx,
    drug_code_to_idx,
    max_diag_len=MAX_DIAG_LEN,
    max_drug_len=MAX_DRUG_LEN,
)

valid_dataset = ClaimsSequenceDataset(
    X_valid_df,
    icd_code_to_idx,
    drug_code_to_idx,
    max_diag_len=MAX_DIAG_LEN,
    max_drug_len=MAX_DRUG_LEN,
)

# WeightedRandomSampler to help with imbalance
train_labels = y_train.astype(int)
class_sample_count = np.bincount(train_labels)  # [num_neg, num_pos]
class_weights = 1.0 / class_sample_count
sample_weights = class_weights[train_labels]

sample_weights = torch.from_numpy(sample_weights).double()
train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)


# ---------------------------------------------------------------------
# 8. MODEL WITH POOLED SEQUENCES
# ---------------------------------------------------------------------

class AE_MLP(nn.Module):
    """
    Embedding-based MLP with pooling:
      - ICD & drug tokens -> MedTok embeddings
      - Age, race, gender -> embeddings
      - Masked mean pool ICD and drug sequences (ignore PAD)
      - Concatenate [age, race, gender, pooled_icd, pooled_drug]
      - Feed to MLP
    """
    def __init__(
        self,
        icd_emb_matrix,
        drug_emb_matrix,
        num_age_bins,
        num_race,
        num_gender,
        max_diag_len,
        max_drug_len,
        hidden_dims=(512, 256),
        dropout=0.3,
        pad_idx_icd=0,
        pad_idx_drug=0,
    ):
        super().__init__()

        self.embed_dim = icd_emb_matrix.shape[1]
        self.max_diag_len = max_diag_len
        self.max_drug_len = max_drug_len
        self.pad_idx_icd = pad_idx_icd
        self.pad_idx_drug = pad_idx_drug

        # ICD & drug embeddings from MedTok
        self.icd_embedding = nn.Embedding.from_pretrained(
            torch.tensor(icd_emb_matrix, dtype=torch.float32),
            padding_idx=pad_idx_icd,
            freeze=False,  # fine-tune embeddings
        )
        self.drug_embedding = nn.Embedding.from_pretrained(
            torch.tensor(drug_emb_matrix, dtype=torch.float32),
            padding_idx=pad_idx_drug,
            freeze=False,
        )

        # demographic embeddings (same dim so they act like tokens)
        self.age_embedding = nn.Embedding(num_age_bins, self.embed_dim)
        self.race_embedding = nn.Embedding(num_race, self.embed_dim)
        self.gender_embedding = nn.Embedding(num_gender, self.embed_dim)

        # age + race + gender (3D) + pooled_icd (D) + pooled_drug (D) => 5D
        input_dim = 5 * self.embed_dim

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def masked_mean(self, emb, ids, pad_idx):
        """
        emb: (B, L, D), ids: (B, L)
        compute mean across non-pad tokens
        """
        mask = (ids != pad_idx).unsqueeze(-1)  # (B, L, 1)
        masked_emb = emb * mask
        lengths = mask.sum(dim=1).clamp(min=1)  # (B, 1)
        return masked_emb.sum(dim=1) / lengths  # (B, D)

    def forward(self, diag_ids, drug_ids, age_ids, race_ids, gender_ids):
        diag_emb = self.icd_embedding(diag_ids)    # (B, Ld, D)
        drug_emb = self.drug_embedding(drug_ids)   # (B, Lm, D)

        diag_pooled = self.masked_mean(diag_emb, diag_ids, self.pad_idx_icd)   # (B, D)
        drug_pooled = self.masked_mean(drug_emb, drug_ids, self.pad_idx_drug)  # (B, D)

        age_emb = self.age_embedding(age_ids)        # (B, D)
        race_emb = self.race_embedding(race_ids)     # (B, D)
        gender_emb = self.gender_embedding(gender_ids)  # (B, D)

        x = torch.cat(
            [age_emb, race_emb, gender_emb, diag_pooled, drug_pooled],
            dim=1,
        )  # (B, 5D)

        logits = self.mlp(x).squeeze(-1)
        return logits


model = AE_MLP(
    icd_emb_matrix=icd_emb_matrix,
    drug_emb_matrix=drug_emb_matrix,
    num_age_bins=len(agebin_to_idx),
    num_race=len(race_to_idx),
    num_gender=len(gender_to_idx),
    max_diag_len=MAX_DIAG_LEN,
    max_drug_len=MAX_DRUG_LEN,
    hidden_dims=HIDDEN_DIMS,
    dropout=DROPOUT,
    pad_idx_icd=icd_code_to_idx[PAD_TOKEN],
    pad_idx_drug=drug_code_to_idx[PAD_TOKEN],
).to(DEVICE)

pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
)

# Optional LR scheduler on AUC-PR
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1,
    threshold=1e-4,
)

# AMP setup
USE_AMP = (DEVICE.type == "cuda")
if USE_AMP:
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
else:
    autocast = None
    scaler = None


# ---------------------------------------------------------------------
# 9. TRAINING LOOP
# ---------------------------------------------------------------------

best_auc_pr = -1.0
best_state_dict = None
best_epoch = -1
epochs_without_improvement = 0

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    model.train()
    running_loss = 0.0

    for batch in tqdm(train_loader, desc="Training"):
        diag_ids, drug_ids, age_ids, race_ids, gender_ids, labels = batch
        diag_ids = diag_ids.to(DEVICE, non_blocking=True)
        drug_ids = drug_ids.to(DEVICE, non_blocking=True)
        age_ids = age_ids.to(DEVICE, non_blocking=True)
        race_ids = race_ids.to(DEVICE, non_blocking=True)
        gender_ids = gender_ids.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        if USE_AMP:
            with autocast():
                logits = model(diag_ids, drug_ids, age_ids, race_ids, gender_ids)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()

            # unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(diag_ids, drug_ids, age_ids, race_ids, gender_ids)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        running_loss += loss.item() * labels.size(0)

    avg_train_loss = running_loss / len(train_dataset)
    print(f"Train loss: {avg_train_loss:.4f}")

    # validation
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validation"):
            diag_ids, drug_ids, age_ids, race_ids, gender_ids, labels = batch
            diag_ids = diag_ids.to(DEVICE, non_blocking=True)
            drug_ids = drug_ids.to(DEVICE, non_blocking=True)
            age_ids = age_ids.to(DEVICE, non_blocking=True)
            race_ids = race_ids.to(DEVICE, non_blocking=True)
            gender_ids = gender_ids.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if USE_AMP:
                with autocast():
                    logits = model(diag_ids, drug_ids, age_ids, race_ids, gender_ids)
                    probs = torch.sigmoid(logits)
            else:
                logits = model(diag_ids, drug_ids, age_ids, race_ids, gender_ids)
                probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    auc_pr = average_precision_score(all_labels, all_probs)
    auc_roc = roc_auc_score(all_labels, all_probs)
    print(f"Valid AUC-PR:  {auc_pr:.4f}")
    print(f"Valid AUC-ROC: {auc_roc:.4f}")

    # Step LR scheduler based on AUC-PR
    scheduler.step(auc_pr)

    if auc_pr > best_auc_pr + 1e-4:  # small tolerance
        best_auc_pr = auc_pr
        best_state_dict = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        epochs_without_improvement = 0
        print("New best model found!")
    else:
        epochs_without_improvement += 1
        print(f"No improvement in AUC-PR for {epochs_without_improvement} epoch(s).")

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(
            f"Early stopping after {epoch} epochs. Best epoch was {best_epoch} "
            f"with AUC-PR = {best_auc_pr:.4f}."
        )
        break


# ---------------------------------------------------------------------
# 10. FINAL EVAL & SAVE
# ---------------------------------------------------------------------

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
    print(f"\nLoaded best model from epoch {best_epoch} with AUC-PR = {best_auc_pr:.4f}")
else:
    print("\nWarning: no best_state_dict was set; using last epoch model.")

model.eval()
all_probs = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(valid_loader, desc="Final evaluation"):
        diag_ids, drug_ids, age_ids, race_ids, gender_ids, labels = batch
        diag_ids = diag_ids.to(DEVICE, non_blocking=True)
        drug_ids = drug_ids.to(DEVICE, non_blocking=True)
        age_ids = age_ids.to(DEVICE, non_blocking=True)
        race_ids = race_ids.to(DEVICE, non_blocking=True)
        gender_ids = gender_ids.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if USE_AMP:
            with autocast():
                logits = model(diag_ids, drug_ids, age_ids, race_ids, gender_ids)
                probs = torch.sigmoid(logits)
        else:
            logits = model(diag_ids, drug_ids, age_ids, race_ids, gender_ids)
            probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

all_probs = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels)

final_auc_pr = average_precision_score(all_labels, all_probs)
final_auc_roc = roc_auc_score(all_labels, all_probs)
print("\nBest model on validation:")
print(f"AUC-PR:  {final_auc_pr:.4f}")
print(f"AUC-ROC: {final_auc_roc:.4f}")

# Threshold sweep for best F1
best_thr = 0.5
best_f1 = -1.0

for thr in np.linspace(0.0, 1.0, 101):
    y_pred_tmp = (all_probs >= thr).astype(int)
    f1 = f1_score(all_labels, y_pred_tmp)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

print(f"\nBest F1 on validation: {best_f1:.4f} at threshold = {best_thr:.3f}")

y_pred = (all_probs >= best_thr).astype(int)
print(f"\nClassification report at threshold = {best_thr:.3f}:")
print(classification_report(all_labels, y_pred, digits=3))

save_dir = BASE / "mlp_models"
save_dir.mkdir(parents=True, exist_ok=True)

# Save a log with all detail about the model and training in a text file
log_path = save_dir / f"mlp_ae30d_medtok_model{SUFFIX}.txt"
with open(log_path, "w") as f:
    f.write(f"Title: {TITLE}\n")
    f.write(f"ICD vocab size: {len(icd_code_to_idx)}\n")
    f.write(f"Drug vocab size: {len(drug_code_to_idx)}\n")
    f.write(f"Age bins: {agebin_to_idx}\n")
    f.write(f"Race categories: {race_to_idx}\n")
    f.write(f"Gender categories: {gender_to_idx}\n")
    f.write(f"Embedding dimension: {embed_dim}\n")
    f.write(f"Max diag length: {MAX_DIAG_LEN}\n")
    f.write(f"Max drug length: {MAX_DRUG_LEN}\n")
    f.write(f"Hidden dims: {HIDDEN_DIMS}\n")
    f.write(f"Dropout: {DROPOUT}\n")
    f.write(f"Best AUC-PR: {final_auc_pr:.4f}\n")
    f.write(f"Best AUC-ROC: {final_auc_roc:.4f}\n")
    f.write(f"Best F1: {best_f1:.4f}\n")
    f.write(f"Best F1 threshold: {best_thr:.3f}\n")
    f.write(f"Title: {TITLE}\n")

model_path = save_dir / f"mlp_ae30d_medtok_model{SUFFIX}.pt"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "icd_code_to_idx": icd_code_to_idx,
        "drug_code_to_idx": drug_code_to_idx,
        "agebin_to_idx": agebin_to_idx,
        "race_to_idx": race_to_idx,
        "gender_to_idx": gender_to_idx,
        "embed_dim": embed_dim,
        "max_diag_len": MAX_DIAG_LEN,
        "max_drug_len": MAX_DRUG_LEN,
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
        "best_auc_pr": final_auc_pr,
        "best_auc_roc": final_auc_roc,
        "best_threshold": float(best_thr),
        "title": TITLE,
    },
    model_path,
)

print(f"\nModel saved to: {model_path}")
print(f"Log saved to: {log_path}")
print("Done.")
