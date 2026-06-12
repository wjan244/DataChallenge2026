import yaml
import numpy as np
import pandas as pd
import mlflow

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from src.data.data_utils import compute_gender_weights, N_BINS_GENDER
from src.data.data_stats import get_test_distribution_from_screenshot, N_BINS
from src.data.transforms import get_augmentation_pretrained_transforms
from src.config import SCREENSHOT_PATH, DATA, IMG_DIR, DEVICE, SUBMISSION_DIR, CONFIG
from src.models.loss import PWScore, PScore


def load_config(config_name="dino.yaml"):
    return yaml.safe_load(open(CONFIG / "models" / config_name))

def compute_laplacian_iw(df, alpha=50):
    bins = np.linspace(0, 1, N_BINS+1)
    # alpha = 1/n_bins if alpha == -1 else alpha
    n_ech = len(df)
    train_hist, _ = np.histogram(df["FaceOcclusion"], bins=bins)
    train_hist = (train_hist) / train_hist.sum()
    test_dist = get_test_distribution_from_screenshot(SCREENSHOT_PATH,n_bins=N_BINS)
    
    ratio_dist = (test_dist+alpha/n_ech)/(1e-6+train_hist+alpha/n_ech)
    
    y = df["FaceOcclusion"].values
    bin_idx = np.digitize(y, bins) -1 # for zero index
    bin_idx = np.clip(bin_idx,0, N_BINS - 1)
    iw = ratio_dist[bin_idx] # no need to clip with smoothing .clip(0.05, 10) # to avoid extreme values

    return iw


class EmbeddingDataset(Dataset):
    def __init__(self, split, cfg):
        emb_dir = DATA / cfg["embedding_dir"]
        self.meta = pd.read_csv(emb_dir / f"{split}_meta.csv")
        
        # Remove the noisy samples
        noisy = pd.read_csv(DATA / "occlusion_datasets" / "validation_noisy.csv")
        noisy_files = set(noisy["filename"])
        mask = self.meta["filename"].isin(noisy_files)   # True = noisy

        if split == "val" and not cfg.get("val_use_noisy", True):
            keep_idx = self.meta[~mask].index.tolist()        # original indices to keep
            self.meta = self.meta[~mask].reset_index(drop=True)
            
        elif split == "train" and not cfg.get("train_use_noisy", True):
            keep_idx = self.meta[~mask].index.tolist()        # original indices to keep
            self.meta = self.meta[~mask].reset_index(drop=True)
        else:
            keep_idx = list(range(len(self.meta)))
            
        # load embeddings based on lp_embedding config
        mode = cfg["lp_embedding"]  # "cls" | "patch_mean" | "concat"
        if mode in ("cls", "concat"):
            cls = torch.load(emb_dir / f"{split}_cls.pt", weights_only=False).float()
        if mode in ("patch_mean", "concat"):
            patch = torch.load(emb_dir / f"{split}_patch_mean.pt", weights_only=False).float()

        if mode == "cls":
            self.embeddings = cls[keep_idx] # [N x D]
        elif mode == "patch_mean":
            self.embeddings = patch[keep_idx] # [N x D]
        else:  # concat
            self.embeddings = torch.cat([cls, patch], dim=1)[keep_idx] # [N x 2D]

        n = len(self.meta)
        if split != "test":
            self.labels  = torch.tensor(self.meta["FaceOcclusion"].values, dtype=torch.float32)
            self.genders = torch.tensor(self.meta["gender"].values,        dtype=torch.float32)
            self.iws     = torch.tensor(compute_laplacian_iw(self.meta, cfg["smooth_alpha"]), dtype=torch.float32)
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



def eval_epoch_cnn(model, loader, loss_fn):
    score_fn = PWScore()
    preds, ys, iws, pis, gws, genders = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for emb, cls, y, gender, iw, pi, gw in loader:
            pred = model(emb.to(DEVICE), cls.to(DEVICE)).squeeze(-1).cpu()
            preds.append(pred); ys.append(y); iws.append(iw)
            pis.append(pi); gws.append(gw); genders.append(gender)

    p, y, iw, pi, gw, g = (torch.cat(x) for x in [preds, ys, iws, pis, gws, genders])
    
    val_score, err_f, err_m = score_fn(p, y, iw, pi, g)
    
    val_loss = loss_fn(p, y, iw, pi, gw, g).item()
    
    return val_score.item(), err_f.item(), err_m.item(), val_loss


def eval_final_cnn(model, loader):
    score_fn = PScore()
    preds, ys, iws, pis, gws, genders = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for emb, cls, y, gender, iw, pi, gw in loader:
            pred = model(emb.to(DEVICE), cls.to(DEVICE)).squeeze(-1).cpu()
            preds.append(pred); ys.append(y); iws.append(iw)
            pis.append(pi); gws.append(gw); genders.append(gender)

    p, y, iw, pi, gw, g = (torch.cat(x) for x in [preds, ys, iws, pis, gws, genders])
    
    val_score = score_fn(p, y, iw, pi, g)

    return val_score.item()

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

def save_submission_cnn(model, cfg, loader, timestamp, split="test"):
    emb_dir = DATA / cfg["embedding_dir"]
    model.eval()
    preds = []
    with torch.no_grad():
        for emb, cls, *_ in loader:
            pred = model(emb.to(DEVICE), cls.to(DEVICE)).squeeze(-1).cpu()
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
    
    
class PatchDataset(Dataset):
    def __init__(self, split, cfg):
        emb_dir = DATA / cfg["embedding_dir"]
        self.meta = pd.read_csv(emb_dir / f"{split}_meta.csv")
        N_orig = len(self.meta)
          
        if (split == "val" and not cfg.get("val_use_noisy", True)) or \
        (split == "train" and not cfg.get("train_use_noisy", True)):
            noisy = pd.read_csv(DATA / "occlusion_datasets" / "validation_noisy.csv",
                                        header=None, names=["filename", "FaceOcclusion", "gender"])
            noisy_files = set(noisy["filename"])
            self.meta = self.meta[~self.meta["filename"].isin(noisy_files)]
            keep_idx = self.meta.index.tolist()   # original indices before reset
            self.meta = self.meta.reset_index(drop=True)
        else:
            keep_idx = list(range(len(self.meta)))
        
        embed_dim = cfg.get("embed_dim", 1280)
        self.embed_dim = embed_dim
        patches_raw = np.memmap(emb_dir / f"{split}_patches.bin",
                                dtype=np.float16, mode='r',
                                shape=(N_orig, 196, embed_dim))
        self.keep_idx = keep_idx  # store for __getitem__
        self._patches_raw = patches_raw  # keep reference alive
   
        # load embeddings based on lp_embedding config
        if cfg.get("patch_use_cls", False):
            self.cls = torch.load(emb_dir / f"{split}_cls.pt",
                                mmap=True, weights_only=False)[keep_idx]
        else:
            self.cls = None
            
        n = len(self.meta)
        if split != "test":
            self.labels  = torch.tensor(self.meta["FaceOcclusion"].values, dtype=torch.float32)
            self.genders = torch.tensor(self.meta["gender"].values,        dtype=torch.float32)
            self.iws     = torch.tensor(compute_laplacian_iw(self.meta, cfg["smooth_alpha"]), dtype=torch.float32)
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
        orig_idx = self.keep_idx[idx]
        x = torch.from_numpy(self._patches_raw[orig_idx].copy()).float().reshape(self.embed_dim, 14, 14)
        cls = self.cls[idx].float() if self.cls is not None else torch.zeros(self.embed_dim)

        return x, cls, self.labels[idx], self.genders[idx], self.iws[idx], self.pis[idx], self.gws[idx]


class ImageDataset(Dataset):
    """Loads raw images from IMG_DIR using train_split.csv / val_split.csv (which include a
    `noisy` column = 1 for noisy-label samples).  The base_transform (normalization) is passed
    in from outside; augmentation is composed on top.
    """
    def __init__(self, split: str, cfg: dict, transform, augment: bool = False):
        if split == "test":
            self.meta = pd.read_csv(DATA / "occlusion_datasets" / "test_students.csv")
        else:
            self.meta = pd.read_csv(DATA / "occlusion_datasets" / f"{split}_split.csv")
            use_noisy_key = "val_use_noisy" if split == "val" else "train_use_noisy"
            if not cfg.get(use_noisy_key, True):
                self.meta = self.meta[self.meta["noisy"] == 0].reset_index(drop=True)

        self.filenames = self.meta["filename"].tolist()
        n = len(self.meta)

        if split != "test":
            self.labels  = torch.tensor(self.meta["FaceOcclusion"].values, dtype=torch.float32)
            self.genders = torch.tensor(self.meta["gender"].values,        dtype=torch.float32)
            self.iws     = torch.tensor(compute_laplacian_iw(self.meta, cfg["smooth_alpha"]), dtype=torch.float32)
            self.pis     = 1/30 + self.labels
            W_F, W_M     = compute_gender_weights(self.labels, self.genders)
            bins_gender  = torch.linspace(0, 1, N_BINS_GENDER + 1)
            bin_idx      = (torch.bucketize(self.labels, bins_gender, right=False) - 1).clamp(0, N_BINS_GENDER - 1)
            self.gws     = torch.where(self.genders == 0.0, W_F[bin_idx], W_M[bin_idx])
        else:
            self.labels  = torch.zeros(n)
            self.genders = torch.full((n,), -1.0)
            self.iws = self.pis = self.gws = torch.ones(n)

        aug = get_augmentation_pretrained_transforms() if (augment and split == "train") else None
        self.transform = v2.Compose([transform, aug]) if aug else transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(IMG_DIR / self.filenames[idx]).convert("RGB")
        return (self.transform(img), self.labels[idx], self.genders[idx],
                self.iws[idx], self.pis[idx], self.gws[idx])


def eval_epoch_image(model, loader, loss_fn):
    """Same as eval_epoch but for image loaders: calls model(img) instead of model(emb)."""
    score_fn = PWScore()
    preds, ys, iws, pis, gws, genders = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for img, y, gender, iw, pi, gw in loader:
            pred = model(img.to(DEVICE)).squeeze(-1).cpu()
            preds.append(pred); ys.append(y); iws.append(iw)
            pis.append(pi); gws.append(gw); genders.append(gender)

    p, y, iw, pi, gw, g = (torch.cat(x) for x in [preds, ys, iws, pis, gws, genders])
    val_score, err_f, err_m = score_fn(p, y, iw, pi, g)
    val_loss = loss_fn(p, y, iw, pi, gw, g).item()
    return val_score.item(), err_f.item(), err_m.item(), val_loss
