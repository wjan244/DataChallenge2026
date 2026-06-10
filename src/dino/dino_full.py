import math
import time
import mlflow
import torch
import torch.nn as nn
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import v2
from transformers import AutoModel, AutoImageProcessor

from src.config import DEVICE, CHECKPOINT_DIR, NUM_WORKERS, SUBMISSION_DIR, DATA
from src.dino.run_lp import build_loss
from src.dino.run_cnn import PatchCNN
from src.dino.utils import load_config, ImageDataset, eval_epoch_image

_PIN = DEVICE.type == "cuda"
_PW  = NUM_WORKERS > 0


# --- copied from run_unfreeze.py ---

class DinoFinetuneModel(nn.Module):
    def __init__(self, backbone, head: PatchCNN):
        super().__init__()
        self.backbone    = backbone
        self.head        = head
        n_reg            = getattr(backbone.config, "num_register_tokens", 0)
        self.patch_start = 1 + n_reg

    def forward(self, pixel_values):
        h       = self.backbone(pixel_values=pixel_values).last_hidden_state
        cls     = h[:, 0]
        patches = h[:, self.patch_start:]
        B, N, D = patches.shape
        grid    = patches.reshape(B, 14, 14, D).permute(0, 3, 1, 2)
        return self.head(grid, cls)


def _get_layers_and_norm(backbone):
    if hasattr(backbone, 'model') and hasattr(backbone.model, 'layer'):
        return backbone.model.layer, backbone.norm
    if hasattr(backbone, 'layer'):
        return backbone.layer, getattr(backbone, 'norm', None)
    if hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'layer'):
        norm = getattr(backbone, 'layernorm', getattr(backbone, 'norm', None))
        return backbone.encoder.layer, norm
    children = [n for n, _ in backbone.named_children()]
    raise AttributeError(f"Cannot find layers in {type(backbone).__name__}. Children: {children}")


def _build_image_transform(model_name: str):
    processor = AutoImageProcessor.from_pretrained(model_name)
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])


def _make_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    def head_lambda(epoch):
        return 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, total_epochs)))

    def backbone_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        e = epoch - warmup_epochs
        n = max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * e / n))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[head_lambda, backbone_lambda])

# --- end copied from run_unfreeze.py ---


def _make_scheduler_phase2(optimizer, total_epochs: int):
    """Cosine decay for all param groups — used after phase-2 adds a 3rd group."""
    def cosine(epoch):
        return 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, total_epochs)))
    n_groups = len(optimizer.param_groups)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[cosine] * n_groups)


def _set_blocks_unfrozen(backbone, n: int):
    """Freeze entire backbone then unfreeze top n blocks (attn + mlp + layer_scale only)."""
    for p in backbone.parameters():
        p.requires_grad = False
    if n == 0:
        return
    layers, _ = _get_layers_and_norm(backbone)
    for layer in layers[-n:]:
        for p in layer.attention.parameters():    p.requires_grad = True
        for p in layer.mlp.parameters():          p.requires_grad = True
        for p in layer.layer_scale1.parameters(): p.requires_grad = True
        for p in layer.layer_scale2.parameters(): p.requires_grad = True
        # norm1, norm2, final norm stay frozen


def build_dino_full_model(cfg) -> DinoFinetuneModel:
    backbone = AutoModel.from_pretrained(cfg["model_name"])
    for p in backbone.parameters():
        p.requires_grad = False

    D    = backbone.config.hidden_size
    head = PatchCNN(patch_dim=D, dropout=cfg["lp_dropout"], use_cls=True)
    if ckpt := cfg.get("head_checkpoint"):
        head.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"Head loaded from {ckpt}")
    else:
        print("Head initialised from scratch")

    # backbone starts fully frozen — unfreezing happens inside train_full
    model = DinoFinetuneModel(backbone, head)
    # save pretrained weights for L2-SP penalty (penalise deviation from init, not from zero)
    model.pretrained_state     = {k: v.clone().detach() for k, v in backbone.state_dict().items()}
    model.n_backbone_trainable = 0
    return model


def l2_sp_penalty(model: DinoFinetuneModel, lambda_sp: float) -> torch.Tensor:
    """L2 penalty relative to pretrained weights, averaged over trainable backbone param count (L2-SP)."""
    if model.n_backbone_trainable == 0:
        return torch.tensor(0.0, device=DEVICE)
    penalty = torch.tensor(0.0, device=DEVICE)
    for name, param in model.backbone.named_parameters():
        if param.requires_grad:
            ref = model.pretrained_state[name].to(DEVICE)
            penalty = penalty + (param - ref).pow(2).sum()
    return lambda_sp * penalty / model.n_backbone_trainable


def _backbone_wd(cfg: dict) -> float:
    """Zero backbone weight decay when L2-SP handles regularisation."""
    return 0.0 if float(cfg.get("l2_sp_lambda", 0.0)) > 0 else cfg["weight_decay"]


def _build_optimizer_head_only(model: DinoFinetuneModel, cfg: dict):
    return torch.optim.AdamW([
        {"params": list(model.head.parameters()), "lr": cfg["learning_rate_head"], "weight_decay": cfg["weight_decay"]},
    ])


def _add_blocks_to_optimizer(optimizer, model: DinoFinetuneModel, n_total: int, cfg: dict):
    """Unfreeze top n_total backbone blocks and add newly unfrozen params as a new optimizer group.
    Uses id-set guard so it's safe regardless of n_total/n_prev values (no slice edge cases).
    Also updates model.n_backbone_trainable."""
    _set_blocks_unfrozen(model.backbone, n_total)
    already = {id(p) for g in optimizer.param_groups for p in g["params"]}
    new_params = [p for p in model.backbone.parameters() if p.requires_grad and id(p) not in already]
    optimizer.add_param_group({"params": new_params, "lr": cfg["learning_rate_backbone"], "weight_decay": _backbone_wd(cfg)})
    model.n_backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)


def train_full(model, train_loader, train_no_aug_loader, val_loader, loss_fn, save_path, cfg):
    n_epoch         = cfg.get("lp_epochs", 30)
    n_head_epochs   = cfg.get("n_head_epochs", 0)
    n_warmup_blocks = cfg.get("n_warmup_blocks", 3)
    n_warmup_epochs = cfg.get("n_warmup_block_epochs", 5)
    no_aug_last     = cfg.get("no_aug_last_n_epochs", 2)
    no_val          = cfg.get("no_val", False)
    patience        = cfg.get("lp_patience", 5)
    n_total_blocks  = cfg["n_blocks"]
    lambda_sp       = float(cfg.get("l2_sp_lambda", 0.0))
    phase1_path     = save_path.parent / (save_path.stem + "_phase1.pt")

    optimizer = _build_optimizer_head_only(model, cfg)
    scheduler = _make_scheduler_phase2(optimizer, max(n_head_epochs, 1))
    best_score, patience_ctr = float("inf"), 0

    for epoch in range(n_epoch):
        # Phase 1 transition: unfreeze top n_warmup_blocks
        if epoch == n_head_epochs:
            _add_blocks_to_optimizer(optimizer, model, n_warmup_blocks, cfg)
            scheduler = _make_scheduler_phase2(optimizer, max(n_warmup_epochs, 1))
            patience_ctr = 0
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            mlflow.log_param("phase1_start_epoch", epoch)
            mlflow.log_param("trainable_params_phase1", trainable)
            print(f"Phase 1: {n_warmup_blocks} blocks unfrozen, trainable={trainable:,}, LR reset")

        # Phase 2 transition: unfreeze all n_total_blocks
        if epoch == n_head_epochs + n_warmup_epochs:
            _add_blocks_to_optimizer(optimizer, model, n_total_blocks, cfg)
            scheduler = _make_scheduler_phase2(optimizer, max(n_epoch - epoch, 1))
            patience_ctr = 0
            torch.save(model.state_dict(), phase1_path)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            mlflow.log_param("phase2_start_epoch", epoch)
            mlflow.log_param("trainable_params_phase2", trainable)
            print(f"Phase 2: {n_total_blocks} blocks unfrozen, trainable={trainable:,}, LR reset")

        loader = train_no_aug_loader if (n_epoch - epoch <= no_aug_last) else train_loader
        no_aug_active = loader is train_no_aug_loader

        t0 = time.time()
        model.train()
        running_loss, bb_norm, hd_norm, running_sp = 0.0, 0.0, 0.0, 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{n_epoch}{' [no-aug]' if no_aug_active else ''}")
        for img, y, gender, iw, pi, gw in pbar:
            img, y, gender = img.to(DEVICE), y.to(DEVICE), gender.to(DEVICE)
            iw, pi, gw     = iw.to(DEVICE),  pi.to(DEVICE),  gw.to(DEVICE)
            optimizer.zero_grad()
            pred = model(img).squeeze(-1)
            loss = loss_fn(pred, y, iw, pi, gw, gender)
            if lambda_sp > 0:
                sp = l2_sp_penalty(model, lambda_sp)
                running_sp += sp.item()
                loss = loss + sp
            loss.backward()
            bb_norm += clip_grad_norm_([p for p in model.backbone.parameters() if p.requires_grad], 1.0).item()
            hd_norm += clip_grad_norm_(model.head.parameters(), 1.0).item()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        n_b = len(loader)

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
            val_score, err_f, err_m, val_loss = eval_epoch_image(model, val_loader, loss_fn)
            metrics.update({"val_loss": val_loss, "val_score": val_score,
                            "val_err_female": err_f, "val_err_male": err_m})
            mlflow.log_metrics(metrics, step=epoch)
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
            mlflow.log_metrics(metrics, step=epoch)
            torch.save(model.state_dict(), save_path)

    if not no_val:
        model.load_state_dict(torch.load(save_path, map_location="cpu"))
    return model, best_score if not no_val else None


def _tta_predict(model, imgs, tta_n: int):
    """Average predictions over original + horizontal flip (+ more if tta_n > 2)."""
    preds = [model(imgs).squeeze(-1)]
    if tta_n >= 2:
        preds.append(model(imgs.flip(-1)).squeeze(-1))
    for _ in range(tta_n - 2):
        preds.append(model(imgs.flip(-2)).squeeze(-1))  # vertical flip for extra views
    return torch.stack(preds, dim=0).mean(0)


def save_submission_full(model, cfg, test_loader, val_noisy_loader, timestamp, loss_fn):
    emb_dir = DATA / cfg["embedding_dir"]
    name    = cfg.get("name", cfg["embedding_dir"])
    tta_n   = cfg.get("tta_n", 0)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch[0].to(DEVICE)
            pred = _tta_predict(model, imgs, tta_n).cpu() if tta_n > 1 else model(imgs).squeeze(-1).cpu()
            preds.append(pred)

    out = SUBMISSION_DIR / f"{timestamp}_{name}_full" / "test.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(emb_dir / "test_meta.csv")
    df["FaceOcclusion"] = torch.cat(preds).numpy()
    df["gender"] = "x"
    df[["filename", "FaceOcclusion", "gender"]].to_csv(out, index=False)
    print(f"Test submission saved → {out}  ({len(df)} rows)")

    if val_noisy_loader is not None:
        val_score, err_f, err_m, _ = eval_epoch_image(model, val_noisy_loader, loss_fn)
        mlflow.log_metrics({
            "final_val_score_noisy":      val_score,
            "final_val_err_female_noisy": err_f,
            "final_val_err_male_noisy":   err_m,
        })
        print(f"Val (with noisy): score={val_score:.4f}  err_f={err_f:.4f}  err_m={err_m:.4f}")


def run_full(file_name, timestamp, experiment_id):
    cfg    = load_config(file_name)
    bs     = cfg["lp_batch_size"]
    no_val = cfg.get("no_val", False)
    transform = _build_image_transform(cfg["model_name"])

    kw = dict(num_workers=NUM_WORKERS, pin_memory=_PIN, persistent_workers=_PW)

    if no_val:
        # val split used as additional training data; val portion not augmented (split="val" bypasses augment flag)
        train_base_ds = ConcatDataset([
            ImageDataset("train", cfg,                            transform=transform, augment=True),
            ImageDataset("val",   {**cfg, "val_use_noisy": True}, transform=transform, augment=False),
        ])
        train_no_aug_ds = ConcatDataset([
            ImageDataset("train", cfg,                            transform=transform, augment=False),
            ImageDataset("val",   {**cfg, "val_use_noisy": True}, transform=transform, augment=False),
        ])
        val_loader = None
    else:
        train_base_ds   = ImageDataset("train", cfg,                             transform=transform, augment=True)
        train_no_aug_ds = ImageDataset("train", cfg,                             transform=transform, augment=False)
        val_ds          = ImageDataset("val",   {**cfg, "val_use_noisy": False},  transform=transform, augment=False)
        val_loader      = DataLoader(val_ds, batch_size=bs, shuffle=False, **kw)

    val_noisy_ds = ImageDataset("val",  {**cfg, "val_use_noisy": True}, transform=transform, augment=False)
    test_ds      = ImageDataset("test", cfg,                            transform=transform, augment=False)

    train_loader        = DataLoader(train_base_ds,   batch_size=bs, shuffle=True,  **kw)
    train_no_aug_loader = DataLoader(train_no_aug_ds, batch_size=bs, shuffle=True,  **kw)
    val_noisy_loader    = DataLoader(val_noisy_ds,    batch_size=bs, shuffle=False, **kw)
    test_loader         = DataLoader(test_ds,         batch_size=bs, shuffle=False, **kw)

    loss_fn   = build_loss(cfg)
    model     = build_dino_full_model(cfg).to(DEVICE)
    save_path = CHECKPOINT_DIR / f"{timestamp}_dino_full.pt"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Phase 0 (head only) — trainable: {trainable:,} / {total:,}")

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"dino_full_{timestamp}"):
        mlflow.log_params({k: v for k, v in cfg.items() if not isinstance(v, (dict, list))})
        mlflow.log_params({"trainable_params_phase0": trainable, "no_val": no_val})

        model, best_score = train_full(
            model, train_loader, train_no_aug_loader, val_loader, loss_fn, save_path, cfg
        )
        save_submission_full(model, cfg, test_loader, val_noisy_loader, timestamp, loss_fn)
        if best_score is not None:
            mlflow.log_metric("best_val_score", best_score)
        print("Done.")
