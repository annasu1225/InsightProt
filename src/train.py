import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from vae import VAE
from utils import (
    split_df,
    compute_norm_from_train_df,
    save_norm,
    EmbeddingDataset
)

@torch.no_grad()
def eval_epoch(model, dl, device):
    model.eval()
    losses, recons, kls = [], [], []
    for x, _ in dl:
        x = x.to(device)
        loss, recon, kl = model.elbo_loss(x)
        losses.append(loss.item())
        recons.append(recon.item())
        kls.append(kl.item())
    return float(np.mean(losses)), float(np.mean(recons)), float(np.mean(kls))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="outputs/vae_out")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.manifest)

    # split
    train_df, val_df, test_df, split = split_df(df)
    torch.save(split, os.path.join(args.out, "split.pt"))

    # norm
    norm = compute_norm_from_train_df(train_df, std_thresh=1e-3)
    save_norm(norm, os.path.join(args.out, "norm.pt"))

    # datasets
    train_ds = EmbeddingDataset(train_df, mu=norm["mu"], sigma=norm["sigma"], keep_mask=norm["keep_mask"])
    val_ds   = EmbeddingDataset(val_df,   mu=norm["mu"], sigma=norm["sigma"], keep_mask=norm["keep_mask"])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = VAE(train_ds.X.shape[1], args.latent, args.hidden).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses, tr_recons, tr_kls = [], [], []

        for x, _ in train_dl:
            x = x.to(args.device)
            loss, recon, kl = model.elbo_loss(x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_losses.append(loss.item())
            tr_recons.append(recon.item())
            tr_kls.append(kl.item())

        tr_loss = float(np.mean(tr_losses))
        tr_recon = float(np.mean(tr_recons))
        tr_kl = float(np.mean(tr_kls))

        va_loss, va_recon, va_kl = eval_epoch(model, val_dl, args.device)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_recon": tr_recon,
            "train_kl": tr_kl,
            "val_loss": va_loss,
            "val_recon": va_recon,
            "val_kl": va_kl,
            "recon_logvar": float(model.recon_logvar.detach().cpu()),
        })

        print(
            f"Epoch {epoch:04d}/{args.epochs} | "
            f"train loss {tr_loss:8.3f} (recon {tr_recon:8.3f}, kl {tr_kl:6.3f}) | "
            f"val loss {va_loss:8.3f} (recon {va_recon:8.3f}, kl {va_kl:6.3f}) | "
            f"recon_logvar {history[-1]['recon_logvar']:.3f}"
        )

        # save best by validation loss
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), os.path.join(args.out, "vae_best.pt"))
    
    # save last
    torch.save(model.state_dict(), os.path.join(args.out, "vae_last.pt"))

    # write history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(args.out, "history.csv"), index=False)

    # plot curves
    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train")
    plt.plot(hist_df["epoch"], hist_df["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss (Recon + KL)")
    plt.title(f"VAE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "loss_curve.png"), dpi=200)
    plt.close()

    # print stats
    print("Best val loss:", best_val)
    print("Final recon_logvar:", float(model.recon_logvar.detach().cpu()))
    print("Final sigma:", float(torch.exp(0.5 * model.recon_logvar.detach().cpu())))

if __name__ == "__main__":
    main()