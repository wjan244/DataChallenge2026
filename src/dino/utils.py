import yaml
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from src.data.data_utils import compute_gender_weights, N_BINS_GENDER
from src.data.data_stats import get_test_distribution_from_screenshot, N_BINS
from src.config import SCREENSHOT_PATH, DATA, DEVICE, SUBMISSION_DIR, CONFIG
from src.models.loss import PWScore


def load_config(config_name="dino.yaml"):
    return yaml.safe_load(open(CONFIG / "models" / config_name))

def compute_laplacian_iw(df,  n_bins=N_BINS, alpha=50):
    bins = np.linspace(0, 1, n_bins+1)
    # alpha = 1/n_bins if alpha == -1 else alpha
    n_ech = len(df)
    train_hist, _ = np.histogram(df["FaceOcclusion"], bins=bins)
    train_hist = (train_hist) / train_hist.sum()
    test_dist = get_test_distribution_from_screenshot(SCREENSHOT_PATH,n_bins=n_bins)
    
    ratio_dist = (test_dist+alpha/n_ech)/(train_hist+alpha/n_ech)
    
    y = df["FaceOcclusion"].values
    bin_idx = np.digitize(y, bins) -1 # for zero index
    bin_idx = np.clip(bin_idx,0, n_bins - 1)
    iw = ratio_dist[bin_idx] # no need to clip with smoothing .clip(0.05, 10) # to avoid extreme values

    return iw


class EmbeddingDataset(Dataset):
    def __init__(self, split, cfg):
        emb_dir = DATA / cfg["embedding_dir"]
        self.meta = pd.read_csv(emb_dir / f"{split}_meta.csv")
        
        # load embeddings based on lp_embedding config
        mode = cfg["lp_embedding"]  # "cls" | "patch_mean" | "concat"
        if mode in ("cls", "concat"):
            cls = torch.load(emb_dir / f"{split}_cls.pt", weights_only=False).float()
        if mode in ("patch_mean", "concat"):
            patch = torch.load(emb_dir / f"{split}_patch_mean.pt", weights_only=False).float()

        if mode == "cls":
            self.embeddings = cls # [N x D]
        elif mode == "patch_mean":
            self.embeddings = patch # [N x D]
        else:  # concat
            self.embeddings = torch.cat([cls, patch], dim=1) # [N x 2D]

        n = len(self.meta)
        if split != "test":
            self.labels  = torch.tensor(self.meta["FaceOcclusion"].values, dtype=torch.float32)
            self.genders = torch.tensor(self.meta["gender"].values,        dtype=torch.float32)
            self.iws     = torch.tensor(compute_laplacian_iw(self.meta, cfg["n_bins"], cfg["smooth_alpha"]), dtype=torch.float32)
            self.pis     = 1/30 + self.labels
            W_F, W_M    = compute_gender_weights(self.labels, self.genders)
            bins_gender = torch.linspace(0, 1, N_BINS_GENDER + 1)
            bin_idx     = (torch.bucketize(self.labels, bins_gender, right=False) - 1).clamp(0, N_BINS_GENDER - 1)
            self.gws    = torch.where(self.genders == 0.0, W_F[bin_idx], W_M[bin_idx])
        else:
            self.labels   = torch.zeros(n)
            self.genders  = torch.full((n,), -1.0)
            self.iws      = torch.ones(n)
            self.pis      = torch.ones(n)
            self.gws      = torch.ones(n)

            
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        return self.embeddings[idx, :], self.labels[idx], self.genders[idx], self.iws[idx], self.pis[idx], self.gws[idx]


def eval_epoch(model, loader, loss_fn):
    score_fn = PWScore()
    preds, ys, iws, pis, gws, genders = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for emb, y, gender, iw, pi, gw in loader:
            pred = model(emb.to(DEVICE)).squeeze(-1).cpu()
            preds.append(pred); ys.append(y); iws.append(iw)
            pis.append(pi); gws.append(gw); genders.append(gender)

    p, y, iw, pi, gw, g = (torch.cat(x) for x in [preds, ys, iws, pis, gws, genders])
    
    val_score, err_f, err_m = score_fn(p, y, iw, pi, g)
    
    val_loss = loss_fn(p, y, iw, pi, gw, g).item()
    
    return val_score.item(), err_f.item(), err_m.item(), val_loss


def save_submission(model, cfg, loader, timestamp, split="test"):
    emb_dir = DATA / cfg["embedding_dir"]
    model.eval()
    preds = []
    with torch.no_grad():
        for emb, *_ in loader:
            pred = model(emb.to(DEVICE)).squeeze(-1).cpu()
            preds.append(pred)
    
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    name = cfg.get("name", cfg["embedding_dir"])
    out = SUBMISSION_DIR / f"{timestamp}_{name}" / f"{split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)    
    
    df = pd.read_csv(emb_dir / "test_meta.csv", delimiter=',')
    df["FaceOcclusion"] = torch.cat(preds).numpy()
    df["gender"] = "x"
    df[["filename", "FaceOcclusion", "gender"]].to_csv(out, index=False)
    print(f"Submission saved → {out}  ({len(df)} rows)")

