import torch
import mlflow
import time
import optuna
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.config import DEVICE, CHECKPOINT_DIR, NUM_WORKERS
from src.dino.run_lp import build_loss
from src.dino.utils import load_config, PatchDataset, save_submission_cnn, eval_epoch_cnn, eval_final_cnn

_PIN = DEVICE.type == "cuda"
_PW  = NUM_WORKERS > 0



class PatchCNN(nn.Module):
    def __init__(self, patch_dim=1280, dropout=0.3, use_cls=True):
        super().__init__()
        self.use_cls = use_cls

        self.conv = nn.Sequential(
            # Stage 1: channel reduction, no spatial mixing
            nn.Conv2d(patch_dim, 256, 1, bias=False),
            nn.GroupNorm(1, 256),
            nn.GELU(),

            # # Stage 2: spatial mixing, keep 16×16
            # nn.Conv2d(512, 256, 3, padding=1, bias=False),
            # nn.GroupNorm(1, 256),
            # nn.GELU(),

            # Stage 3: spatial mixing, keep 16×16
            nn.Conv2d(256, 64, 3, stride=1, padding=1, bias=False, padding_mode='reflect'),
            nn.GroupNorm(1, 64),
            nn.GELU(),

            # Stage 4: downsample 14→7
            nn.Conv2d(64, 8, 3, stride=2, padding=1, bias=False, padding_mode='reflect'),
            nn.GroupNorm(1, 8),
            nn.GELU(),

        )
        # Flatten 32×7×7 = 1568 — stride-2 on 14×14 gives 7×7
        conv_out_dim = 8 * 7 * 7   # 392

        cls_out_dim = 128 if use_cls else 0
        self.cls_proj = nn.Linear(patch_dim, cls_out_dim) if use_cls else None

        head_in = conv_out_dim + cls_out_dim  # 520 = 520

        self.head = nn.Sequential(
            nn.Linear(head_in, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self._init_weights()
                
    def forward(self, x, cls=None):
        feat = self.conv(x)       # [B, 32, 7, 7]
        feat = feat.flatten(1)    # [B, 1568]

        if self.use_cls and cls is not None:
            cls_feat = F.gelu(self.cls_proj(cls))       # [B, 512]
            feat = torch.cat([feat, cls_feat], dim=1)   # [B, 2560]

        return self.head(feat)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 'leaky_relu' with a=0 ≈ GELU scaling
                # or just use the exact GELU-aware formula
                nn.init.kaiming_normal_(m.weight, 
                                        mode='fan_out',
                                        nonlinearity='linear')  # conservative
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                # Transformer-style init for linear layers
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    

def train_cnn(model, train_loader, val_loader, optimizer, scheduler, loss_fn, save_path, cfg, trial=None):
    best_score = float("inf")
    n_epoch = cfg.get("lp_epochs", 10)
    patience_counter = 0
    patience = cfg.get("lp_patience", 10)

    
    for epoch in range(n_epoch):
        epoch_start = time.time()
        
        running_loss = 0.0
        model.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch+1}/{n_epoch}")
        for emb, cls, y, gender, iw, pi, gw in pbar:
            emb, cls, y, gender = emb.to(DEVICE), cls.to(DEVICE), y.to(DEVICE), gender.to(DEVICE)
            iw, pi, gw = iw.to(DEVICE), pi.to(DEVICE), gw.to(DEVICE)
            optimizer.zero_grad()
            pred = model(emb, cls).squeeze(-1)
            loss = loss_fn(pred, y, iw, pi, gw, gender)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        
        train_loss = running_loss / len(train_loader)
        val_score, val_score_f, val_score_m, val_loss = eval_epoch_cnn(model, val_loader, loss_fn)
        mlflow.log_metrics({
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss, "val_loss": val_loss,
                "val_score": val_score, "val_err_f": val_score_f, "val_err_m": val_score_m,
                "epoch_time_s": time.time() - epoch_start
                }, step=epoch)
        print(f"[{epoch+1}/{n_epoch}] train={train_loss:.4f} | val_score={val_score:.4f} | err_f={val_score_f:.4f} | err_m={val_score_m:.4f}")
        
        
        # --- early stopping ---
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            mlflow.log_artifact(str(save_path))
            print(f"  → checkpoint saved (val_score={val_score:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} because patience {patience} is over")
                break
    
        #optuna pruning
        if trial is not None:
            trial.report(val_score, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return model, best_score



def run_cnn(file_name, timestamp, experiment_id):
    # config
    cfg = load_config(file_name)
    lr = cfg.get("lp_lr", 1e-3)
    n_epoch = cfg.get("lp_epochs", 10)
    
    # datasets & loaders
    train_loader = DataLoader(PatchDataset("train", cfg), batch_size=cfg["lp_batch_size"], shuffle=True, num_workers=NUM_WORKERS, pin_memory=_PIN, persistent_workers=_PW)
    val_loader = DataLoader(PatchDataset("val", cfg), batch_size=cfg["lp_batch_size"], num_workers=NUM_WORKERS, pin_memory=_PIN, persistent_workers=_PW)
    test_loader = DataLoader(PatchDataset("test", cfg), batch_size=cfg["lp_batch_size"], num_workers=NUM_WORKERS, pin_memory=_PIN, persistent_workers=_PW)
    
    # model
    model = PatchCNN(patch_dim=cfg.get("embed_dim", 1280),
                     dropout=cfg.get("patch_cnn_dropout", 0.3),
                     use_cls=cfg.get("patch_use_cls", True)).to(DEVICE)
    
    # loss
    loss_fn = build_loss(cfg)

    # optimizer + scheduler
    optimizer = torch.optim.AdamW(lr = lr, params=model.parameters(), weight_decay=cfg.get("lp_weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=n_epoch, eta_min=lr*1e-4)
    
    # training param
    save_path = CHECKPOINT_DIR / f"dino_cnn_{timestamp}.pt"


    # mlflow run
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"dino_cnn_{timestamp}"):
        mlflow.log_params(cfg)
        # training loop with early stopping
        model, best_score = train_cnn(model, train_loader, val_loader, optimizer, scheduler, loss_fn, save_path, cfg)
        
        # load best model
        model.load_state_dict(torch.load(save_path, weights_only=True))

        # save score without wi
        pscore = eval_final_cnn(model, val_loader)
        mlflow.log_metrics({"val_Pscore": pscore})
                
        # save checkpoint + submission
        save_submission_cnn(model, cfg, test_loader, timestamp)
        mlflow.log_metric("best_val_score", best_score)  # add after the epoch loop in train_lp
        print(f"End of training best_score={best_score}")

