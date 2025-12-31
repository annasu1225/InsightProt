import os
import argparse
import numpy as np
import pandas as pd
import torch

from vae import VAE
from utils import load_norm, load_and_normalize_df
from sklearn.metrics import roc_auc_score

@torch.no_grad()
def encode_mu(model: VAE, x: torch.Tensor) -> torch.Tensor:
    mu, _ = model.encode(x)
    return mu

@torch.no_grad()
def set_score(model: VAE, X: torch.Tensor, idxs: np.ndarray, device: str) -> float:
    x = X[idxs].to(device)
    mu = encode_mu(model, x)  # (K, latent_dim)
    # average pairwise squared distance (coherence = negative distance)
    diffs = mu.unsqueeze(1) - mu.unsqueeze(0)          # (K, K, d)
    d2 = (diffs ** 2).sum(dim=-1)                      # (K, K)
    return float((-d2.mean()).cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="outputs/vae_out")      
    ap.add_argument("--k", type=int, default=5)             # conformers per set
    ap.add_argument("--neg_sets", type=int, default=500)    # how many mixed negatives
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()


    df = pd.read_csv(args.manifest)
    df["protein_id"] = df["protein_id"].astype(str)

    split = torch.load(os.path.join(args.out, "split.pt"), map_location="cpu", weights_only=False)
    test_proteins = set(map(str, split["test_proteins"]))
    test_df = df[df["protein_id"].isin(test_proteins)].reset_index(drop=True)

    norm = load_norm(os.path.join(args.out, "norm.pt"))
    X_test = load_and_normalize_df(test_df, norm)

    input_dim = int(norm["mu"].numel())
    model = VAE(input_dim=input_dim, latent_dim=16, hidden_dim=128)
    model.load_state_dict(torch.load(os.path.join(args.out, "vae_best.pt"), map_location="cpu", weights_only=False))
    model.eval().to(args.device)

    test_groups = {}
    for i, pid in enumerate(test_df["protein_id"].tolist()):
        test_groups.setdefault(pid, []).append(i)

    rng = np.random.default_rng(0)

    # positive sets
    pos_scores = []
    for pid, idxs in test_groups.items():
        if len(idxs) < args.k:
            continue
        for _ in range(50):  # multiple samples per protein
            sel = rng.choice(idxs, size=args.k, replace=False)
            pos_scores.append(set_score(model, X_test, sel, args.device))

    # negative sets
    neg_scores = []
    test_pids = [p for p in test_groups.keys() if len(test_groups[p]) >= 1]

    for _ in range(args.neg_sets):
        chosen_pids = rng.choice(test_pids, size=args.k, replace=False)
        sel = np.array([rng.choice(test_groups[p], size=1, replace=False)[0] for p in chosen_pids], dtype=int)
        neg_scores.append(set_score(model, X_test, sel, args.device))

    # AUC
    y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    s = np.concatenate([np.array(pos_scores), np.array(neg_scores)])
    auc = roc_auc_score(y, s)

    print(f"Test proteins: {len(test_proteins)} | Pos sets: {len(pos_scores)} | Neg sets: {len(neg_scores)} | K={args.k}")
    print(f"Set-level ROC-AUC on TEST: {auc:.4f}")
    print(f"Score means: pos={float(np.mean(pos_scores)):.3f} neg={float(np.mean(neg_scores)):.3f}")

if __name__ == "__main__":
    main()
