import inspect
import math
import time
import yaml

import mlflow
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from src.config import SCREENSHOT_PATH, DATA, IMG_DIR, DEVICE, SUBMISSION_DIR, CONFIG, CHECKPOINT_DIR, NUM_WORKERS
from src.data.data_utils import compute_gender_weights, N_BINS_GENDER
from src.data.data_stats import get_test_distribution_from_screenshot, N_BINS
from src.models.loss import PWScore, LOSS_MAPPING, UniversalLossWrapper

_PIN = DEVICE.type == "cuda"
_PW  = NUM_WORKERS > 0


# ---------------------------------------------------------------------------
# Inlined helpers
# ---------------------------------------------------------------------------

def load_config(config_name="dino_convnext.yaml"):
    with open(CONFIG  / "models" / config_name) as f:
        return yaml.safe_load(f)


def build_loss(cfg):
    loss_cls = LOSS_MAPPING[cfg["lp_loss"]]
    all_kwargs = {"alpha": cfg.get("lp_loss_alpha", 1.0), "beta": cfg.get("lp_loss_beta", 0.1),
                  "gamma": cfg.get("lp_loss_gamma", 1.0), "kappa": cfg.get("lp_loss_kappa", 1.0),
                  "tau": cfg.get("lp_loss_tau", 1.0)}
    sig = inspect.signature(loss_cls.__init__).parameters
    loss_kwargs = {k: v for k, v in all_kwargs.items() if k in sig}
    loss_fn = UniversalLossWrapper(loss_cls(**loss_kwargs))
    print(f"Loss: {loss_cls.__name__}  kwargs={loss_kwargs}")
    return loss_fn


def _augmentation():
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.RandomRotation(degrees=15)], p=0.3),
        v2.RandomApply([v2. ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)], p=0.5),
        v2.RandomGrayscale(p=0.15),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.1),
    ])


def _compute_laplacian_iw(df, alpha=50):
    bins = np.linspace(0, 1, N_BINS + 1)
    n_ech = len(df)
    train_hist, _ = np.histogram(df["FaceOcclusion"], bins=bins)
    train_hist = train_hist / train_hist.sum()
    test_dist = get_test_distribution_from_screenshot(SCREENSHOT_PATH, n_bins=N_BINS)
    ratio = (test_dist + alpha / n_ech) / (1e-6 + train_hist + alpha / n_ech)
    bin_idx = np.clip(np.digitize(df["FaceOcclusion"].values, bins) - 1, 0, N_BINS - 1)
    return ratio[bin_idx]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChallengeImageDataset(Dataset):
    """Loads raw images from IMG_DIR using a pre-loaded DataFrame.

    Returns (img, label, gender, iw, pi, gw) — 6 elements, indices 0-5.
    Pass is_train=False for the test split (labels/weights are all-ones dummies).
    """
    def __init__(self, df, transform, is_train: bool, augment: bool, smooth_alpha: float = 0):
        self.df = df.reset_index(drop=True)
        self.filenames = self.df["filename"].tolist()
        n = len(self.df)

        if is_train:
            self.labels  = torch.tensor(self.df["FaceOcclusion"].values, dtype=torch.float32)
            self.genders = torch.tensor(self.df["gender"].values, dtype=torch.float32)
            self.iws     = torch.tensor(_compute_laplacian_iw(self.df, smooth_alpha), dtype=torch.float32)
            self.pis     = 1 / 30 + self.labels
            W_F, W_M     = compute_gender_weights(self.labels, self.genders)
            bins_g       = torch.linspace(0, 1, N_BINS_GENDER + 1)
            bin_idx      = (torch.bucketize(self.labels, bins_g, right=False) - 1).clamp(0, N_BINS_GENDER - 1)
            self.gws     = torch.where(self.genders == 0.0, W_F[bin_idx], W_M[bin_idx])
        else:
            self.labels  = torch.zeros(n)
            self.genders = torch.full((n,), -1.0)
            self.iws = self.pis = self.gws = torch.ones(n)

        aug = _augmentation() if (augment and is_train) else None
        self.transform = v2.Compose([transform, aug]) if aug else transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(IMG_DIR / self.filenames[idx]).convert("RGB")
        return (self.transform(img), self.labels[idx], self.genders[idx],
                self.iws[idx], self.pis[idx], self.gws[idx])


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _build_loaders(cfg):
    bs              = cfg["lp_batch_size"]
    aug             = cfg.get("augmentation", True)
    alpha           = cfg.get("smooth_alpha", 0)
    mn              = cfg["model_name"]
    no_val          = cfg.get("no_val", False)
    use_noisy_train = cfg.get("train_use_noisy", True)
    use_noisy_val   = cfg.get("val_use_noisy", True)

    data_cfg = timm.data.resolve_model_data_config(timm.create_model(mn, pretrained=False))
    train_tf = timm.data.create_transform(**data_cfg, is_training=True)
    val_tf   = timm.data.create_transform(**data_cfg, is_training=False)

    df_train   = pd.read_csv(DATA / "occlusion_datasets" / "train_split.csv")
    df_val     = pd.read_csv(DATA / "occlusion_datasets" / "val_split.csv")
    df_val_all = df_val.copy()   # always includes noisy — used for final scoring
    df_test    = pd.read_csv(DATA / "occlusion_datasets" / "test_students.csv").dropna().reset_index(drop=True)

    if not use_noisy_train:
        df_train = df_train[df_train["noisy"] == 0]
    if not use_noisy_val:
        df_val = df_val[df_val["noisy"] == 0]

    kw = dict(num_workers=NUM_WORKERS, pin_memory=_PIN, persistent_workers=_PW)

    train_loader = DataLoader(
        ChallengeImageDataset(df_train, train_tf, is_train=True, augment=aug, smooth_alpha=alpha),
        batch_size=bs, shuffle=True, **kw)
    val_loader = None if no_val else DataLoader(
        ChallengeImageDataset(df_val, val_tf, is_train=True, augment=False, smooth_alpha=alpha),
        batch_size=bs, shuffle=False, **kw)
    val_loader_noisy = None if no_val else DataLoader(
        ChallengeImageDataset(df_val_all, val_tf, is_train=True, augment=False, smooth_alpha=alpha),
        batch_size=bs, shuffle=False, **kw)
    test_loader = DataLoader(
        ChallengeImageDataset(df_test, val_tf, is_train=False, augment=False),
        batch_size=bs, shuffle=False, **kw)

    return train_loader, val_loader, val_loader_noisy, df_test, test_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PatchHead(nn.Module):
    """
    Tête partagée appliquée à chaque position spatiale.
    Entrée : (B, C, H, W)  — carte de features ConvNeXt
    Sortie : (B, 1) — ratio occludé / visage ∈ (0,1)
    """
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),        # fond / visible / occludé
        )

    def forward(self, fmap):
        # (B, C, H, W) -> (B, H*W, C) : chaque position devient un "token"
        tokens = fmap.flatten(2).transpose(1, 2)
        probs  = torch.softmax(self.net(tokens), dim=-1)   # (B, HW, 3)

        p_vis, p_occ = probs[..., 1], probs[..., 2]
        ratio = p_occ.sum(1) / (p_vis.sum(1) + p_occ.sum(1) + 1e-8)
        return ratio


class PatchOcclusionModel(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone                      # nom stable conservé
        in_dim = backbone.feature_info.channels()[-1]   # 384 pour tiny stage 3
        self.head = PatchHead(in_dim=in_dim, hidden=cfg["lp_hidden"])
        self.pretrained_state = {k: v.clone() for k, v in backbone.state_dict().items()}
        self.n_backbone_trainable = 0

    def forward(self, x):
        fmap = self.backbone(x)[-1]                   # features_only -> liste; (B,384,14,14)
        return self.head(fmap)


def build_model(cfg):
    backbone = timm.create_model(
        cfg["model_name"],
        pretrained=True,
        features_only=True,
        out_indices=(2,),                 # stage 3 uniquement, stride 16
    )
    model = PatchOcclusionModel(backbone, cfg)
    _set_n_stages_unfrozen(backbone, 0)   # freeze backbone for phase-0 head warmup
    print("feature channels:", backbone.feature_info.channels())
    return model




# --------

def _get_stages(backbone) -> list:
    """Return ordered list of ConvNeXt stages from FeatureListNet (exposed as stages_0, stages_1, ...)."""
    stages = [(name, mod) for name, mod in backbone.named_children() if name.startswith("stages_")]
    stages.sort(key=lambda x: x[0])
    return [mod for _, mod in stages]


def _set_n_stages_unfrozen(backbone, n: int):
    """Freeze entire backbone then unfreeze top n stages (ConvNeXt — no cls head in features_only)."""
    for p in backbone.parameters():
        p.requires_grad = False
    if n == 0:
        return
    for stage in _get_stages(backbone)[-n:]:
        for p in stage.parameters():
            p.requires_grad = True


def _backbone_wd(cfg: dict) -> float:
    """Backbone weight decay: only applied when training from scratch (pretrained=false) and L2-SP is off."""
    if float(cfg.get("l2_sp_lambda", 0.0)) > 0:
        return 0.0
    if cfg.get("pretrained", True):
        return 0.0
    return cfg["weight_decay"]


def _build_optimizer_head_only(model: PatchOcclusionModel, cfg: dict):
    return torch.optim.AdamW([
        {"params": list(model.head.parameters()),
         "lr": cfg["learning_rate_head"],
         "weight_decay": cfg["weight_decay"]},
    ])


def _add_stages_to_optimizer(optimizer, model: PatchOcclusionModel, n_total: int, cfg: dict):
    """Unfreeze top n_total stages and add newly unfrozen params as a new optimizer group."""
    _set_n_stages_unfrozen(model.backbone, n_total)
    already = {id(p) for g in optimizer.param_groups for p in g["params"]}
    new_params = [p for p in model.backbone.parameters() if p.requires_grad and id(p) not in already]
    optimizer.add_param_group({
        "params": new_params,
        "lr": cfg["learning_rate_backbone"],
        "weight_decay": _backbone_wd(cfg),
    })
    model.n_backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)


def _make_scheduler_phase(optimizer, total_epochs: int, warmup_epochs: int = 0):
    def schedule(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs          # linear ramp 0 → 1
        t = epoch - warmup_epochs
        n = max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + 0.9 * math.cos(math.pi * t / n))
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[schedule] * len(optimizer.param_groups)
    )


def l2_sp_penalty(model: PatchOcclusionModel, lambda_sp: float) -> torch.Tensor:
    """L2 penalty relative to pretrained weights, averaged over trainable backbone param count."""
    if model.n_backbone_trainable == 0:
        return torch.tensor(0.0, device=DEVICE)
    penalty = torch.tensor(0.0, device=DEVICE)
    for name, param in model.backbone.named_parameters():
        if param.requires_grad:
            ref = model.pretrained_state[name]   # already on DEVICE — moved once at run start
            penalty = penalty + (param - ref).pow(2).sum()
    return lambda_sp * penalty / model.n_backbone_trainable


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _eval(model, val_loader, loss_fn):
    score_fn = PWScore()
    preds, ys, iws, pis, gws, genders = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            X = batch[0].to(DEVICE)
            preds.append(model(X).cpu())
            ys.append(batch[1].float().view(-1))
            genders.append(batch[2].float().view(-1))
            iws.append(batch[3].float().view(-1))
            pis.append(batch[4].float().view(-1))
            gws.append(batch[5].float().view(-1))

    p  = torch.cat(preds)
    y  = torch.cat(ys)
    g  = torch.cat(genders)
    iw = torch.cat(iws)
    pi = torch.cat(pis)
    gw = torch.cat(gws)

    val_score, err_f, err_m = score_fn(
        p.to(DEVICE), y.to(DEVICE), iw.to(DEVICE), pi.to(DEVICE), g.to(DEVICE)
    )
    val_loss = loss_fn(p, y, iw, pi, gw, g).item()
    return val_score.item(), err_f.item(), err_m.item(), val_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_full(model, train_loader, val_loader, loss_fn, save_path, cfg):
    n_epoch         = cfg.get("lp_epochs", 50)
    n_head_epochs   = cfg.get("n_head_epochs", 0)
    n_warmup_stages = cfg.get("n_warmup_blocks", 2)
    n_warmup_epochs = cfg.get("n_warmup_block_epochs", 5)
    no_val          = cfg.get("no_val", False)
    patience        = cfg.get("lp_patience", 10)
    n_total_stages  = cfg["n_blocks"]
    lambda_sp       = float(cfg.get("l2_sp_lambda", 0.0))
    warmup_epochs   = cfg.get("warmup_epochs", 0)
    phase1_path     = save_path.parent / (save_path.stem + "_phase1.pt")

    optimizer = _build_optimizer_head_only(model, cfg)
    scheduler = _make_scheduler_phase(optimizer, max(n_head_epochs, 1), warmup_epochs)
    best_score, patience_ctr = float("inf"), 0

    for epoch in range(n_epoch):
        # Phase 1: partial unfreeze — only when a warmup window actually exists
        if n_warmup_epochs > 0 and epoch == n_head_epochs:
            _add_stages_to_optimizer(optimizer, model, n_warmup_stages, cfg)
            scheduler = _make_scheduler_phase(optimizer, n_warmup_epochs, warmup_epochs)
            patience_ctr = 0
            torch.cuda.empty_cache()
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            mlflow.log_param("phase1_start_epoch", epoch)
            mlflow.log_param("trainable_params_phase1", trainable)
            print(f"Phase 1: {n_warmup_stages} stages unfrozen, trainable={trainable:,}")

        # Phase 2: full unfreeze — fires even when both head and warmup epochs are 0
        if epoch == n_head_epochs + n_warmup_epochs:
            _add_stages_to_optimizer(optimizer, model, n_total_stages, cfg)
            scheduler = _make_scheduler_phase(optimizer, max(n_epoch - epoch, 1), warmup_epochs)
            patience_ctr = 0
            if n_warmup_epochs > 0:
                torch.save(model.state_dict(), phase1_path)
            torch.cuda.empty_cache()
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            mlflow.log_param("phase2_start_epoch", epoch)
            mlflow.log_param("trainable_params_phase2", trainable)
            print(f"Phase 2: {n_total_stages} stages unfrozen, trainable={trainable:,}")

        t0 = time.time()
        model.train()
        running_loss, bb_norm, hd_norm, running_sp = 0.0, 0.0, 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epoch}")
        for batch in pbar:
            X      = batch[0].to(DEVICE)
            y      = batch[1].float().view(-1).to(DEVICE)
            gender = batch[2].float().view(-1).to(DEVICE)
            iw     = batch[3].float().view(-1).to(DEVICE)
            pi     = batch[4].float().view(-1).to(DEVICE)
            gw     = batch[5].float().view(-1).to(DEVICE)

            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y, iw, pi, gw, gender)
            if lambda_sp > 0:
                sp = l2_sp_penalty(model, lambda_sp)
                running_sp += sp.item()
                loss = loss + sp
            loss.backward()
            bb_norm += clip_grad_norm_(
                [p for p in model.backbone.parameters() if p.requires_grad], 1.0
            ).item()
            hd_norm += clip_grad_norm_(model.head.parameters(), 1.0).item()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        n_b = len(train_loader)

        metrics = {
            "train_loss":         running_loss / n_b,
            "l2_sp_penalty":      running_sp / n_b,
            "grad_norm_backbone": bb_norm / n_b,
            "grad_norm_head":     hd_norm / n_b,
            "lr_head":            optimizer.param_groups[0]["lr"],
            "epoch_time_s":       time.time() - t0,
        }
        if len(optimizer.param_groups) > 1:
            metrics["lr_backbone"] = optimizer.param_groups[1]["lr"]

        if not no_val:
            val_score, err_f, err_m, val_loss = _eval(model, val_loader, loss_fn)
            metrics.update({"val_loss": val_loss, "val_score": val_score,
                            "val_err_female": err_f, "val_err_male": err_m})
            mlflow.log_metrics({k: v for k, v in metrics.items() if math.isfinite(v)}, step=epoch)
            print(f"[{epoch+1}/{n_epoch}] val_score={val_score:.4f}  err_f={err_f:.4f}  err_m={err_m:.4f}")

            if val_score < best_score:
                best_score, patience_ctr = val_score, 0
                torch.save(model.state_dict(), save_path)
                print(f"  checkpoint saved (val_score={val_score:.4f})")
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"Early stop at epoch {epoch+1}")
                    break
        else:
            mlflow.log_metrics({k: v for k, v in metrics.items() if math.isfinite(v)}, step=epoch)
            torch.save(model.state_dict(), save_path)

    if not no_val:
        model.load_state_dict(torch.load(save_path, map_location="cpu"))
    return model, best_score if not no_val else None


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def save_submission(model, cfg, test_loader, df_test, timestamp, loss_fn, val_loader=None):
    name  = cfg.get("name", cfg["model_name"].replace("/", "-"))
    tta_n = cfg.get("tta_n", 0)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch[0].to(DEVICE)
            if tta_n >= 2:
                pred = (model(imgs) + model(imgs.flip(-1))) / 2
            else:
                pred = model(imgs)
            preds.append(pred.cpu())

    out = SUBMISSION_DIR / f"{timestamp}_{name}_dino_conv" / "test.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = df_test[["filename"]].copy()
    df["FaceOcclusion"] = torch.cat(preds).numpy()
    df["gender"] = "x"
    df[["filename", "FaceOcclusion", "gender"]].to_csv(out, index=False)
    print(f"Test submission saved → {out}  ({len(df)} rows)")

    if val_loader is not None:
        val_score, err_f, err_m, _ = _eval(model, val_loader, loss_fn)
        mlflow.log_metrics({
            "final_val_score":      val_score,
            "final_val_err_female": err_f,
            "final_val_err_male":   err_m,
        })
        print(f"Final val: score={val_score:.4f}  err_f={err_f:.4f}  err_m={err_m:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_dino_conv(file_name, timestamp, experiment_id):
    cfg = load_config(file_name)

    seed = cfg.get("seed", 42)
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    no_val = cfg.get("no_val", False)

    train_loader, val_loader, val_loader_noisy, df_test, test_loader = _build_loaders(cfg)

    loss_fn   = build_loss(cfg)
    model     = build_model(cfg).to(DEVICE)
    # move pretrained reference weights to device once so l2_sp_penalty avoids per-batch .to() calls
    model.pretrained_state = {k: v.to(DEVICE) for k, v in model.pretrained_state.items()}  # captured in __init__

    resume = cfg.get("resume_checkpoint")
    if resume:
        ckpt = torch.load(resume, map_location=DEVICE)
        model.load_state_dict(ckpt)
        print(f"Resumed from checkpoint: {resume}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = CHECKPOINT_DIR / f"{timestamp}_full.pt"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Phase 0 (head only) — trainable: {trainable:,} / {total:,}")

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"full_{timestamp}"):
        mlflow.log_params({k: v for k, v in cfg.items() if not isinstance(v, (dict, list))})
        mlflow.log_params({"trainable_params_phase0": trainable, "no_val": no_val})

        model, best_score = train_full(
            model, train_loader, val_loader, loss_fn, save_path, cfg
        )
        save_submission(model, cfg, test_loader, df_test, timestamp, loss_fn, val_loader_noisy)
        if best_score is not None:
            mlflow.log_metric("best_val_score", best_score)
        print("Done.")
