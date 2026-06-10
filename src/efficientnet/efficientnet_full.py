import math
import time
import mlflow
import timm
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

from src.config import DEVICE, CHECKPOINT_DIR, NUM_WORKERS, SUBMISSION_DIR
from src.dino.run_lp import build_loss
from src.dino.utils import load_config
from src.models.loss import PWScore
from src.data.data_loader import (
    get_challenge_train_loader,
    get_challenge_val_loader,
    get_challenge_test_loader,
)
from src.data.data_utils import get_challenge_split

_PIN = DEVICE.type == "cuda"
_PW  = NUM_WORKERS > 0


class EfficientNetHead(nn.Module):
    def __init__(self, in_features: int, hidden_size: int = 512, mid_size: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, mid_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [B]


class EfficientNetFinetuneModel(nn.Module):
    def __init__(self, backbone, head: EfficientNetHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))  # [B]


def _get_stages(backbone) -> list:
    """Return flat list of EfficientNet stages (timm backbone.blocks children)."""
    return list(backbone.blocks.children())


def _set_n_stages_unfrozen(backbone, n: int):
    """Freeze entire backbone then unfreeze top n stages + conv_head + bn2."""
    for p in backbone.parameters():
        p.requires_grad = False
    if n == 0:
        return
    for stage in _get_stages(backbone)[-n:]:
        for p in stage.parameters():
            p.requires_grad = True
    for p in backbone.conv_head.parameters():
        p.requires_grad = True
    for p in backbone.bn2.parameters():
        p.requires_grad = True


def _backbone_wd(cfg: dict) -> float:
    """Zero backbone weight decay when L2-SP handles regularisation."""
    return 0.0 if float(cfg.get("l2_sp_lambda", 0.0)) > 0 else cfg["weight_decay"]


def _build_optimizer_head_only(model: EfficientNetFinetuneModel, cfg: dict):
    return torch.optim.AdamW([
        {"params": list(model.head.parameters()),
         "lr": cfg["learning_rate_head"],
         "weight_decay": cfg["weight_decay"]},
    ])


def _add_stages_to_optimizer(optimizer, model: EfficientNetFinetuneModel, n_total: int, cfg: dict):
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


def _make_scheduler_phase(optimizer, total_epochs: int):
    def cosine(epoch):
        return 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, total_epochs)))
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[cosine] * len(optimizer.param_groups)
    )


def l2_sp_penalty(model: EfficientNetFinetuneModel, lambda_sp: float) -> torch.Tensor:
    """L2 penalty relative to pretrained weights, averaged over trainable backbone param count."""
    if model.n_backbone_trainable == 0:
        return torch.tensor(0.0, device=DEVICE)
    penalty = torch.tensor(0.0, device=DEVICE)
    for name, param in model.backbone.named_parameters():
        if param.requires_grad:
            ref = model.pretrained_state[name]   # already on DEVICE — moved once at run start
            penalty = penalty + (param - ref).pow(2).sum()
    return lambda_sp * penalty / model.n_backbone_trainable


def _eval_efficientnet(model, val_loader, loss_fn):
    score_fn = PWScore()
    preds, ys, iws, pis, gws, genders = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            X = batch[0].to(DEVICE)
            preds.append(model(X).cpu())
            ys.append(batch[1].float().view(-1))
            genders.append(batch[2].float().view(-1))
            iws.append(batch[4].float().view(-1))
            pis.append(batch[5].float().view(-1))
            gws.append(batch[6].float().view(-1))

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


def build_efficientnet_ft_model(cfg) -> EfficientNetFinetuneModel:
    backbone = timm.create_model(cfg["model_name"], pretrained=True, num_classes=0)
    for p in backbone.parameters():
        p.requires_grad = False

    head = EfficientNetHead(
        in_features=backbone.num_features,
        hidden_size=cfg.get("hidden_size", 512),
        mid_size=cfg.get("mid_size", 64),
        dropout=cfg.get("lp_dropout", 0.2),
    )

    model = EfficientNetFinetuneModel(backbone, head)
    model.pretrained_state    = {k: v.clone().detach() for k, v in backbone.state_dict().items()}
    model.n_backbone_trainable = 0

    if ckpt := cfg.get("resume_checkpoint"):
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"Resumed from checkpoint: {ckpt}")

    return model


def train_efficientnet_full(model, train_loader, train_no_aug_loader, val_loader, loss_fn, save_path, cfg):
    n_epoch         = cfg.get("lp_epochs", 50)
    n_head_epochs   = cfg.get("n_head_epochs", 0)
    n_warmup_stages = cfg.get("n_warmup_blocks", 2)
    n_warmup_epochs = cfg.get("n_warmup_block_epochs", 5)
    no_aug_last     = cfg.get("no_aug_last_n_epochs", 2)
    no_val          = cfg.get("no_val", False)
    patience        = cfg.get("lp_patience", 10)
    n_total_stages  = cfg["n_blocks"]
    lambda_sp       = float(cfg.get("l2_sp_lambda", 0.0))
    phase1_path     = save_path.parent / (save_path.stem + "_phase1.pt")

    optimizer = _build_optimizer_head_only(model, cfg)
    scheduler = _make_scheduler_phase(optimizer, max(n_head_epochs, 1))
    best_score, patience_ctr = float("inf"), 0

    for epoch in range(n_epoch):
        # Phase 1: unfreeze top n_warmup_stages
        if epoch == n_head_epochs:
            _add_stages_to_optimizer(optimizer, model, n_warmup_stages, cfg)
            scheduler = _make_scheduler_phase(optimizer, max(n_warmup_epochs, 1))
            patience_ctr = 0
            torch.cuda.empty_cache()
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            mlflow.log_param("phase1_start_epoch", epoch)
            mlflow.log_param("trainable_params_phase1", trainable)
            print(f"Phase 1: {n_warmup_stages} stages unfrozen, trainable={trainable:,}")

        # Phase 2: unfreeze all n_total_stages
        if epoch == n_head_epochs + n_warmup_epochs:
            _add_stages_to_optimizer(optimizer, model, n_total_stages, cfg)
            scheduler = _make_scheduler_phase(optimizer, max(n_epoch - epoch, 1))
            patience_ctr = 0
            torch.save(model.state_dict(), phase1_path)
            torch.cuda.empty_cache()
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            mlflow.log_param("phase2_start_epoch", epoch)
            mlflow.log_param("trainable_params_phase2", trainable)
            print(f"Phase 2: {n_total_stages} stages unfrozen, trainable={trainable:,}")

        loader = train_no_aug_loader if (n_epoch - epoch <= no_aug_last) else train_loader
        no_aug_active = loader is train_no_aug_loader

        t0 = time.time()
        model.train()
        running_loss, bb_norm, hd_norm, running_sp = 0.0, 0.0, 0.0, 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{n_epoch}{' [no-aug]' if no_aug_active else ''}")
        for batch in pbar:
            X      = batch[0].to(DEVICE)
            y      = batch[1].float().view(-1).to(DEVICE)
            gender = batch[2].float().view(-1).to(DEVICE)
            iw     = batch[4].float().view(-1).to(DEVICE)
            pi     = batch[5].float().view(-1).to(DEVICE)
            gw     = batch[6].float().view(-1).to(DEVICE)

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
            val_score, err_f, err_m, val_loss = _eval_efficientnet(model, val_loader, loss_fn)
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


def save_submission_efficientnet(model, cfg, test_loader, df_test, timestamp, loss_fn, val_loader=None):
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

    out = SUBMISSION_DIR / f"{timestamp}_{name}_eff_full" / "test.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = df_test[["filename"]].copy()
    df["FaceOcclusion"] = torch.cat(preds).numpy()
    df["gender"] = "x"
    df[["filename", "FaceOcclusion", "gender"]].to_csv(out, index=False)
    print(f"Test submission saved → {out}  ({len(df)} rows)")

    if val_loader is not None:
        val_score, err_f, err_m, _ = _eval_efficientnet(model, val_loader, loss_fn)
        mlflow.log_metrics({
            "final_val_score":      val_score,
            "final_val_err_female": err_f,
            "final_val_err_male":   err_m,
        })
        print(f"Final val: score={val_score:.4f}  err_f={err_f:.4f}  err_m={err_m:.4f}")


def run_efficientnet_full(file_name, timestamp, experiment_id):
    cfg      = load_config(file_name)
    bs       = cfg["lp_batch_size"]
    aug      = cfg.get("augmentation", True)
    no_val   = cfg.get("no_val", False)
    model_nm = cfg["model_name"]

    train_loader        = get_challenge_train_loader(batch_size=bs, model_name=model_nm, augmentation=aug)
    train_no_aug_loader = get_challenge_train_loader(batch_size=bs, model_name=model_nm, augmentation=False)
    val_loader          = None if no_val else get_challenge_val_loader(
        split="val_samp", batch_size=bs, model_name=model_nm
    )
    _, _, _, df_test = get_challenge_split()
    test_loader      = get_challenge_test_loader(df_test, bs, model_name=model_nm)

    loss_fn   = build_loss(cfg)
    model     = build_efficientnet_ft_model(cfg).to(DEVICE)
    # move pretrained reference weights to device once so l2_sp_penalty avoids per-batch .to() calls
    model.pretrained_state = {k: v.to(DEVICE) for k, v in model.pretrained_state.items()}
    save_path = CHECKPOINT_DIR / f"{timestamp}_efficientnet_full.pt"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Phase 0 (head only) — trainable: {trainable:,} / {total:,}")

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"efficientnet_full_{timestamp}"):
        mlflow.log_params({k: v for k, v in cfg.items() if not isinstance(v, (dict, list))})
        mlflow.log_params({"trainable_params_phase0": trainable, "no_val": no_val})

        model, best_score = train_efficientnet_full(
            model, train_loader, train_no_aug_loader, val_loader, loss_fn, save_path, cfg
        )
        save_submission_efficientnet(model, cfg, test_loader, df_test, timestamp, loss_fn, val_loader)
        if best_score is not None:
            mlflow.log_metric("best_val_score", best_score)
        print("Done.")
