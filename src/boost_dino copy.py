"""XGBoost DINOv3 pipeline — Stage 2.

Loads precomputed patch embeddings (from scripts/extract_embeddings.py),
trains AmoHead (ratio) and GenderHead jointly on frozen embeddings, then
uses their predictions as 2-D features for an XGBoost regressor.

Usage:
    python src/xgboost_dino.py --config dino_xgb.yaml
"""
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from src.config import DEVICE, DATA, NUM_WORKERS, SUBMISSION_DIR, CHECKPOINT_DIR
from src.data.data_utils import compute_gender_weights, N_BINS_GENDER
from src.dino.run_lp import build_loss
from src.dino.utils import load_config, compute_laplacian_iw
from src.models.loss import PWScore

_PIN = DEVICE.type == "cuda"
_PW  = NUM_WORKERS > 0


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class AmoHead(nn.Module):
    """Per-patch shared MLP → 3-class softmax → occluded / (visible + occluded) ratio.
    Classes: 0=background, 1=visible, 2=occluded.
    Input: (B, n_patch, D)  Output: (B, 1).
    Copied from src/dino/dino_full.py."""
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, patch_tokens):           # (B, n_patch, D)
        probs = torch.softmax(self.net(patch_tokens), dim=-1)
        p_vis = probs[:, :, 1]
        p_occ = probs[:, :, 2]
        ratio = p_occ.sum(dim=1) / (p_vis.sum(dim=1) + p_occ.sum(dim=1) + 1e-8)
        return ratio.unsqueeze(1)              # (B, 1)


class GenderHead(nn.Module):
    """Per-patch shared MLP → mean-pool → P(female).
    Input: (B, n_patch, D)  Output: (B, 1)."""
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, patch_tokens):           # (B, n_patch, D)
        return self.net(patch_tokens).mean(dim=1)  # (B, 1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class XGBPatchDataset(Dataset):
    """Loads precomputed patch embeddings from a memory-mapped binary file.

    Mirrors the pattern of PatchDataset in src/dino/utils.py.  The meta CSV
    (copied from train_split / val_split by the extraction script) already
    carries a 'noisy' column, so filtering is a simple mask — no need to
    cross-reference validation_noisy.csv.
    """
    def __init__(self, split: str, cfg: dict):
        emb_dir = DATA / cfg["embedding_dir"]
        meta = pd.read_csv(emb_dir / f"{split}_meta.csv")

        with open(emb_dir / "shapes.json") as f:
            shapes = json.load(f)

        use_noisy_key = "val_use_noisy" if split == "val" else "train_use_noisy"
        if split != "test" and not cfg.get(use_noisy_key, True):
            mask = meta["noisy"] == 0
            keep_idx = meta[mask].index.tolist()
            meta = meta[mask].reset_index(drop=True)
        else:
            keep_idx = list(range(len(meta)))

        self.meta     = meta
        self.keep_idx = keep_idx

        sh = shapes[split]
        n_orig, n_patches, embed_dim = sh["N"], sh["n_patches"], sh["embed_dim"]
        self._patches_raw = np.memmap(emb_dir / f"{split}_patches.bin",
                                      dtype=np.float16, mode="r",
                                      shape=(n_orig, n_patches, embed_dim))

        n = len(self.meta)
        if split != "test":
            self.labels  = torch.tensor(self.meta["FaceOcclusion"].values, dtype=torch.float32)
            self.genders = torch.tensor(self.meta["gender"].values,        dtype=torch.float32)
            self.iws     = torch.tensor(
                compute_laplacian_iw(self.meta, cfg["smooth_alpha"]), dtype=torch.float32)
            self.pis     = 1 / 30 + self.labels
            W_F, W_M     = compute_gender_weights(self.labels, self.genders)
            bins_gender  = torch.linspace(0, 1, N_BINS_GENDER + 1)
            bin_idx      = (torch.bucketize(self.labels, bins_gender, right=False) - 1
                            ).clamp(0, N_BINS_GENDER - 1)
            self.gws     = torch.where(self.genders == 0.0, W_F[bin_idx], W_M[bin_idx])
        else:
            self.labels  = torch.zeros(n)
            self.genders = torch.full((n,), -1.0)
            self.iws     = torch.ones(n)
            self.pis     = torch.ones(n)
            self.gws     = torch.ones(n)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        orig = self.keep_idx[idx]
        patches = torch.from_numpy(self._patches_raw[orig].copy()).float()  # (N, D)
        return (patches, self.labels[idx], self.genders[idx],
                self.iws[idx], self.pis[idx], self.gws[idx])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _eval_heads(amo_head, val_loader):
    score_fn = PWScore()
    amo_head.eval()
    preds, ys, iws, pis, genders = [], [], [], [], []
    with torch.no_grad():
        for patches, y, gender, iw, pi, _ in val_loader:
            pred = amo_head(patches.to(DEVICE)).squeeze(-1).cpu()
            preds.append(pred); ys.append(y); iws.append(iw)
            pis.append(pi); genders.append(gender)
    p, y, iw, pi, g = (torch.cat(x) for x in [preds, ys, iws, pis, genders])
    val_score, err_f, err_m = score_fn(p, y, iw, pi, g)
    return val_score.item(), err_f.item(), err_m.item()


def train_heads(amo_head, gender_head, train_loader, val_loader, loss_fn, cfg, save_dir):
    optimizer = torch.optim.AdamW(
        list(amo_head.parameters()) + list(gender_head.parameters()),
        lr=cfg["learning_rate_head"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.get("lp_epochs", 30))
    gender_lambda = cfg.get("gender_lambda", 0.5)
    patience      = cfg.get("lp_patience", 5)
    n_epoch       = cfg.get("lp_epochs", 30)
    best_score, patience_ctr = float("inf"), 0

    for epoch in range(n_epoch):
        amo_head.train(); gender_head.train()
        running_loss = 0.0

        for patches, y, gender, iw, pi, gw in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epoch}"):
            patches = patches.to(DEVICE)
            y, gender = y.to(DEVICE), gender.to(DEVICE)
            iw, pi, gw = iw.to(DEVICE), pi.to(DEVICE), gw.to(DEVICE)

            optimizer.zero_grad()
            ratio_pred  = amo_head(patches).squeeze(-1)     # (B,)
            gender_pred = gender_head(patches).squeeze(-1)  # (B,)

            ratio_loss  = loss_fn(ratio_pred, y, iw, pi, gw, gender)
            gender_loss = F.binary_cross_entropy(gender_pred, (gender == 0).float())
            total_loss  = ratio_loss + gender_lambda * gender_loss

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        scheduler.step()

        val_score, err_f, err_m = _eval_heads(amo_head, val_loader)
        print(f"[{epoch+1}/{n_epoch}] loss={running_loss/len(train_loader):.4f}"
              f"  val_score={val_score:.4f}  err_f={err_f:.4f}  err_m={err_m:.4f}")

        if val_score < best_score:
            best_score, patience_ctr = val_score, 0
            torch.save(amo_head.state_dict(),    save_dir / "amo_head_best.pt")
            torch.save(gender_head.state_dict(), save_dir / "gender_head_best.pt")
            print(f"  checkpoint saved (val_score={val_score:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stop at epoch {epoch+1}")
                break

    amo_head.load_state_dict(   torch.load(save_dir / "amo_head_best.pt",    map_location="cpu"))
    gender_head.load_state_dict(torch.load(save_dir / "gender_head_best.pt", map_location="cpu"))
    return amo_head, gender_head, best_score


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_features(amo_head, gender_head, loader):
    """Returns (X [N, D+2], y [N], gender [N], iw [N], pi [N]) numpy arrays.

    Features: mean-pooled patch embedding (D) + ratio pred (1) + gender pred (1).
    """
    amo_head.eval(); gender_head.eval()
    feats, labels, genders, iws, pis = [], [], [], [], []
    with torch.no_grad():
        for patches, y, gender, iw, pi, _ in loader:
            patches = patches.to(DEVICE)
            pooled  = patches.mean(dim=1).cpu()                # (B, D)
            r = amo_head(patches).squeeze(-1).cpu()            # (B,)
            g = gender_head(patches).squeeze(-1).cpu()         # (B,)
            feats.append(torch.cat([
                pooled,          # (B, D) — raw DINO features
                r.unsqueeze(1),  # (B, 1) — ratio prediction
                g.unsqueeze(1),  # (B, 1) — gender prediction
            ], dim=1))
            labels.append(y); genders.append(gender)
            iws.append(iw);   pis.append(pi)
    X = torch.cat(feats).numpy()
    y = torch.cat(labels).numpy()
    g = torch.cat(genders).numpy()
    iw = torch.cat(iws).numpy()
    pi = torch.cat(pis).numpy()
    return X, y, g, iw, pi


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_xgb(cfg):
    bs = cfg["lp_batch_size"]
    kw = dict(num_workers=NUM_WORKERS, pin_memory=_PIN, persistent_workers=_PW)

    train_ds = XGBPatchDataset("train", cfg)
    val_ds   = XGBPatchDataset("val",   cfg)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, **kw)

    embed_dim   = int(train_ds._patches_raw.shape[2])
    amo_head    = AmoHead(in_dim=embed_dim,
                          hidden=cfg.get("amo_hidden", 128),
                          dropout=cfg.get("lp_dropout", 0.1)).to(DEVICE)
    gender_head = GenderHead(in_dim=embed_dim,
                             hidden=cfg.get("gender_hidden", 64),
                             dropout=cfg.get("lp_dropout", 0.1)).to(DEVICE)
    loss_fn = build_loss(cfg)

    if ckpt_dir := cfg.get("head_checkpoint_dir"):
        ckpt_dir = Path(ckpt_dir)
        amo_head.load_state_dict(   torch.load(ckpt_dir / "amo_head_best.pt",    map_location="cpu"))
        gender_head.load_state_dict(torch.load(ckpt_dir / "gender_head_best.pt", map_location="cpu"))
        print(f"Heads loaded from {ckpt_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir  = CHECKPOINT_DIR / f"{timestamp}_xgb"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Training heads ===")
    amo_head, gender_head, best_head_score = train_heads(
        amo_head, gender_head, train_loader, val_loader, loss_fn, cfg, save_dir)
    print(f"Best head val_score: {best_head_score:.4f}")

    print("\n=== Extracting features ===")
    X_train, y_train, *_             = _extract_features(amo_head, gender_head, train_loader)
    X_val,   y_val,   g_val, iw_val, pi_val = _extract_features(amo_head, gender_head, val_loader)
    print(f"Train: {X_train.shape}  Val: {X_val.shape}")

    print("\n=== Training XGBoost ===")
    xgb_model = xgb.XGBRegressor(
        n_estimators=cfg.get("xgb_n_estimators", 1000),
        max_depth=cfg.get("xgb_max_depth", 4),
        learning_rate=cfg.get("xgb_lr", 0.05),
        subsample=0.8,
        colsample_bytree=0.3,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=100)

    preds_val = xgb_model.predict(X_val).clip(0, 1)
    score_fn  = PWScore()
    p_t  = torch.from_numpy(preds_val).float()
    y_t  = torch.from_numpy(y_val).float()
    iw_t = torch.from_numpy(iw_val).float()
    pi_t = torch.from_numpy(pi_val).float()
    g_t  = torch.from_numpy(g_val).float()
    val_score, err_f, err_m = score_fn(p_t, y_t, iw_t, pi_t, g_t)
    print(f"\nVal competition score (XGBoost): {val_score.item():.4f}"
          f"  err_f={err_f.item():.4f}  err_m={err_m.item():.4f}")

    emb_dir        = DATA / cfg["embedding_dir"]
    test_meta_path = emb_dir / "test_meta.csv"
    if test_meta_path.exists():
        test_ds     = XGBPatchDataset("test", cfg)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, **kw)
        X_test, *_ = _extract_features(amo_head, gender_head, test_loader)
        preds_test   = xgb_model.predict(X_test).clip(0, 1)

        out_dir = SUBMISSION_DIR / f"{timestamp}_xgb"
        out_dir.mkdir(parents=True, exist_ok=True)
        df_test = pd.read_csv(test_meta_path)
        df_test["FaceOcclusion"] = preds_test
        df_test["gender"] = "x"
        df_test[["filename", "FaceOcclusion", "gender"]].to_csv(out_dir / "test.csv", index=False)
        print(f"Submission saved → {out_dir}/test.csv  ({len(df_test)} rows)")
    else:
        print("No test_meta.csv found — skipping submission.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dino_xgb.yaml")
    args = parser.parse_args()
    run_xgb(load_config(args.config))


if __name__ == "__main__":
    main()
