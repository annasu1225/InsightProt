import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score

# dataset
def load_embedding(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return torch.as_tensor(obj, dtype=torch.float32).flatten()

class EmbeddingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mu=None, sigma=None, keep_mask=None):
        self.df = df.reset_index(drop=True)
        X = torch.stack([load_embedding(p) for p in df.embedding_path])

        if keep_mask is not None:
            X = X[:, keep_mask]

        if mu is not None:
            X = (X - mu) / (sigma + 1e-6)

        self.X = X

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.df.protein_id.iloc[idx]

def split_df(df: pd.DataFrame, seed: int = 0, train_frac: float = 0.75, val_frac: float = 0.15):

    df = df.copy()
    df["protein_id"] = df["protein_id"].astype(str)

    proteins = df["protein_id"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(proteins)

    n = len(proteins)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_p = set(proteins[:n_train])
    val_p   = set(proteins[n_train:n_train + n_val])
    test_p  = set(proteins[n_train + n_val:])

    train_df = df[df["protein_id"].isin(train_p)].reset_index(drop=True)
    val_df   = df[df["protein_id"].isin(val_p)].reset_index(drop=True)
    test_df  = df[df["protein_id"].isin(test_p)].reset_index(drop=True)

    split = {
        "train_proteins": sorted(train_p),
        "val_proteins": sorted(val_p),
        "test_proteins": sorted(test_p),
        "seed": seed,
        "fractions": {"train": train_frac, "val": val_frac, "test": 1 - train_frac - val_frac},
    }

    return train_df, val_df, test_df, split

# normalization
def compute_norm_from_train_df(train_df: pd.DataFrame, std_thresh: float = 1e-3):
    Xraw = torch.stack([load_embedding(p) for p in train_df["embedding_path"].tolist()], dim=0)  # (N, D)
    std = Xraw.std(dim=0)
    keep_mask = std >= std_thresh
    mu = Xraw.mean(dim=0)[keep_mask]
    sigma = std[keep_mask].clamp_min(1e-6)  # tiny safeguard after masking
    return {"mu": mu, "sigma": sigma, "keep_mask": keep_mask}

def save_norm(norm: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(norm, out_path)

def load_norm(norm_path: str):
    norm = torch.load(norm_path, map_location="cpu", weights_only=False)
    return norm

def load_and_normalize_df(df: pd.DataFrame, norm: dict):
    X = torch.stack([load_embedding(p) for p in df["embedding_path"].tolist()], dim=0)
    keep_mask = norm.get("keep_mask", None)
    if keep_mask is not None:
        X = X[:, keep_mask]
    X = (X - norm["mu"]) / norm["sigma"]
    return X

# evaluation metric
def set_auc(scores_same, scores_mixed):
    y = np.concatenate([np.ones(len(scores_same)), np.zeros(len(scores_mixed))])
    s = np.concatenate([scores_same, scores_mixed])
    return roc_auc_score(y, s)