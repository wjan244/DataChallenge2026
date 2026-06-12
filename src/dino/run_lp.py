import torch
import mlflow
import inspect
import time
import optuna

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.config import DEVICE, CHECKPOINT_DIR, NUM_WORKERS
from src.models.loss import LOSS_MAPPING, UniversalLossWrapper
from src.models.scratch_cnn import _init_weights
from src.dino.utils import load_config, EmbeddingDataset, eval_epoch, save_submission

_PIN = DEVICE.type == "cuda"
_PW  = NUM_WORKERS > 0

class LinearProbe(torch.nn.Module):
    def __init__(self, input_dim, hidden, dropout=0.1):
        super().__init__()
        if hidden == 0:
            self.net = torch.nn.Sequential(torch.nn.Linear(input_dim, 1), torch.nn.Sigmoid())
        else:
            # self.net = torch.nn.Sequential(
            #     torch.nn.Linear(input_dim, hidden), torch.nn.GELU(),
            #     torch.nn.Dropout(dropout),
            #     torch.nn.Linear(hidden, 1), torch.nn.Sigmoid()
            # )
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden), torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden // 2, 1), torch.nn.Sigmoid()
            )   
            
        _init_weights(self)
        
    def forward(self, x):
        return self.net(x)

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


def train_lp(model, train_loader, val_loader, optimizer, scheduler, loss_fn, save_path, cfg, trial=None):
    best_score = float("inf")
    n_epoch = cfg.get("lp_epochs", 10)
    patience_counter = 0
    patience = cfg.get("lp_patience", 10)

    
    for epoch in range(n_epoch):
        epoch_start = time.time()
        
        running_loss = 0.0
        model.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch+1}/{n_epoch}")
        for emb, y, gender, iw, pi, gw in pbar:
            emb, y, gender = emb.to(DEVICE), y.to(DEVICE), gender.to(DEVICE)
            iw, pi, gw = iw.to(DEVICE), pi.to(DEVICE), gw.to(DEVICE)
            optimizer.zero_grad()
            pred = model(emb).squeeze(-1)
            loss = loss_fn(pred, y, iw, pi, gw, gender)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        
        train_loss = running_loss / len(train_loader)
        val_score, val_score_f, val_score_m, val_loss = eval_epoch(model, val_loader, loss_fn)
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



def run_lp(file_name, timestamp, experiment_id):
    # config
    cfg = load_config(file_name)
    lr = cfg.get("lp_lr", 1e-3)
    n_epoch = cfg.get("lp_epochs", 10)
    
    # datasets & loaders
    train_loader = DataLoader(EmbeddingDataset("train", cfg), batch_size=cfg["lp_batch_size"], shuffle=True, num_workers=NUM_WORKERS, pin_memory=_PIN, persistent_workers=_PW)
    val_loader = DataLoader(EmbeddingDataset("val", cfg), batch_size=cfg["lp_batch_size"], num_workers=NUM_WORKERS, pin_memory=_PIN, persistent_workers=_PW)
    test_loader = DataLoader(EmbeddingDataset("test", cfg), batch_size=cfg["lp_batch_size"], num_workers=NUM_WORKERS, pin_memory=_PIN, persistent_workers=_PW)
    
    # model
    input_dim = train_loader.dataset.embeddings.shape[1]
    model = LinearProbe(input_dim,cfg.get("lp_hidden", 0), cfg.get("lp_dropout", 0.1)).to(DEVICE)
    
    # loss
    loss_fn = build_loss(cfg)

    # optimizer + scheduler
    optimizer = torch.optim.AdamW(lr = lr, params=model.parameters(), weight_decay=cfg.get("lp_weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=n_epoch, eta_min=lr*1e-4)
    
    # training param
    save_path = CHECKPOINT_DIR / f"{timestamp}_dino_lp.pt"


    # mlflow run
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"dino_lp_{timestamp}"):
        mlflow.log_params(cfg)
        # training loop with early stopping
        model, best_score = train_lp(model, train_loader, val_loader, optimizer, scheduler, loss_fn, save_path, cfg)
        # save checkpoint + submission
        save_submission(model, cfg, test_loader, timestamp)
        mlflow.log_metric("best_val_score", best_score)  # add after the epoch loop in train_lp
        print(f"End of training best_score={best_score}")

