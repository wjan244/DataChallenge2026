import optuna
import mlflow
import torch
import pandas as pd

from torch.utils.data import DataLoader
from src.config import DEVICE, CHECKPOINT_DIR, NUM_WORKERS, CONFIG
from src.dino.utils import load_config, EmbeddingDataset, PatchDataset
from src.dino.run_lp import LinearProbe, build_loss, train_lp, _PIN
from src.dino.run_cnn import PatchCNN, train_cnn

def objective_lp(trial, base_cfg, experiment_id):
    cfg = {**base_cfg}
    cfg["lp_lr"]           = trial.suggest_float("lp_lr", 1e-4, 1e-2, log=True)
    cfg["lp_hidden"]       = trial.suggest_categorical("lp_hidden", [128, 256, 512, 1024])
    cfg["lp_dropout"]      = trial.suggest_float("lp_dropout", 0.0, 0.5)
    cfg["lp_weight_decay"] = trial.suggest_float("lp_weight_decay", 1e-4, 1e-1, log=True)
    cfg["lp_loss_alpha"]   = trial.suggest_float("lp_loss_alpha", 0.1, 5.0)
    cfg["lp_loss_beta"]   = trial.suggest_float("lp_loss_beta", 0.01, 0.5)
    cfg["smooth_alpha"]    = trial.suggest_int("smooth_alpha", 5, 100)
    cfg["lp_loss"]         = trial.suggest_categorical("lp_loss", ["PWLoss", "PWGLoss", "PWGLossRegularized", "HuberPWGLossRegularized"])
    cfg["lp_epochs"]       = base_cfg.get("optuna_epochs", 20)
    cfg["lp_patience"]     = base_cfg.get("optuna_patience", 5)

    bs = cfg["lp_batch_size"]
    
    train_ds = EmbeddingDataset("train", cfg)
    val_ds   = EmbeddingDataset("val",   cfg)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=_PIN, persistent_workers=False)
    val_loader   = DataLoader(val_ds,   batch_size=bs,
                              num_workers=NUM_WORKERS, pin_memory=_PIN, persistent_workers=False)

    input_dim = train_ds.embeddings.shape[1]
    model     = LinearProbe(input_dim, cfg["lp_hidden"], cfg["lp_dropout"]).to(DEVICE)
    loss_fn   = build_loss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lp_lr"],
                                  weight_decay=cfg["lp_weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg["lp_epochs"], eta_min=cfg["lp_lr"] * 1e-4)
    save_path = CHECKPOINT_DIR / f"optuna_trial_{trial.number}.pt"

    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params({k: cfg[k] for k in
                   ["lp_lr", "lp_hidden", "lp_dropout",
                    "lp_weight_decay", "lp_loss_alpha", "lp_loss", "smooth_alpha"]})
        _, best_score = train_lp(model, train_loader, val_loader,
                         optimizer, scheduler, loss_fn, save_path, cfg, trial=trial)
    return best_score

def objective_cnn(trial, base_cfg, experiment_id):
    cfg = {**base_cfg}
    cfg["lp_lr"]           = trial.suggest_float("lp_lr", 1e-4, 1e-2, log=True)
    cfg["lp_dropout"]      = trial.suggest_float("lp_dropout", 0.0, 0.5)
    cfg["lp_weight_decay"] = trial.suggest_float("lp_weight_decay", 1e-4, 1e-1, log=True)
    cfg["lp_loss_alpha"]   = trial.suggest_float("lp_loss_alpha", 0.0, 3.0)
    cfg["lp_loss_beta"]    = trial.suggest_float("lp_loss_beta", 0.01, 1.0)
    cfg["lp_loss_gamma"]   = trial.suggest_float("lp_loss_gamma", 0.0, 2.0)
    cfg["lp_loss_kappa"]   = trial.suggest_float("lp_loss_kappa", 0.0, 2.0)
    cfg["smooth_alpha"]    = trial.suggest_int("smooth_alpha", 5, 100)
    cfg["lp_epochs"]       = base_cfg.get("optuna_epochs", 20)
    cfg["lp_patience"]     = base_cfg.get("optuna_patience", 5)

    bs = cfg["lp_batch_size"]

    train_ds = PatchDataset("train", cfg)
    val_ds   = PatchDataset("val",   cfg)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=_PIN, persistent_workers=False)
    val_loader   = DataLoader(val_ds,   batch_size=bs,
                              num_workers=NUM_WORKERS, pin_memory=_PIN, persistent_workers=False)

    model = PatchCNN(dropout=cfg.get("patch_cnn_dropout", 0.3),
                     use_cls=cfg.get("patch_use_cls", True)).to(DEVICE)
    loss_fn   = build_loss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lp_lr"],
                                  weight_decay=cfg["lp_weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg["lp_epochs"], eta_min=cfg["lp_lr"] * 1e-4)
    save_path = CHECKPOINT_DIR / f"optuna_trial_{trial.number}.pt"

    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params({k: cfg[k] for k in
                   ["lp_lr", "lp_dropout", "lp_weight_decay",
                    "lp_loss_alpha", "lp_loss_beta", "lp_loss_gamma", "lp_loss_kappa", "smooth_alpha"]})
        _, best_score = train_cnn(model, train_loader, val_loader,
                         optimizer, scheduler, loss_fn, save_path, cfg, trial=trial)
    return best_score


def run_optuna(file_name, timestamp, experiment_id, mode='lp'):
    print("Starting Optuna Optimization")
    cfg = load_config(file_name)
    n_trials = cfg.get("optuna_n_trials", 20)

    sampler = optuna.samplers.TPESampler(seed=cfg.get("seed", 42))
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study   = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"optuna_{timestamp}") as parent:
        mlflow.log_params(cfg)

        if mode == 'lp':
            study.optimize(
                lambda trial: objective_lp(trial, cfg, experiment_id),
                n_trials=n_trials,
                show_progress_bar=True,
            )
        elif mode == 'cnn': 
            study.optimize(
                lambda trial: objective_cnn(trial, cfg, experiment_id),
                n_trials=n_trials,
                show_progress_bar=True,
            )
            
        rows = []
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                row = {}
                row.update(t.params)
                row["best_score"] = t.value
                rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("best_score")

        out = CONFIG  / "optuna" / f"{timestamp}_optuna_results.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        mlflow.log_artifact(str(out))
        print(df.to_string(index=False))
        
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_score", study.best_value)

    print(f"\nBest score : {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
