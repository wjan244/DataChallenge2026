import time
import logging
import numpy as np
import mlflow
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)

from src.pipeline.evaluation import run_evaluation
from src.config import *
from src.data.data_loader import *
from src.pipeline.test import run_test, save_split_predictions
from src.models.models import OcclusionModel
from src.models.finetuning import inject_linear_mlp_probing
from src.models.loss import WeightedMSELoss, WeightedLiteMSELoss, UniversalLossWrapper
from src.data.data_utils import get_challenge_split
import timm

LOSS_MAPPING = {"MSE": nn.MSELoss, "BCE": nn.BCELoss, "nMSE": WeightedMSELoss, "nLiteMSE": WeightedLiteMSELoss}


def _build_model(model_name, probing_type, hidden_size=512, num_classes=1):
    backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model = OcclusionModel(backbone)
    return inject_linear_mlp_probing(model, probing_type, hidden_size)


def _count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def _freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False
        
def _unfreeze_head(model):
    for name, param in model.named_parameters():
        if "head" in name or "classifier" in name:
            param.requires_grad = True
            
def _unfreeze_top_n_blocks(model, n):
    backbone = model.model if hasattr(model, "model") else model
    if hasattr(backbone, "blocks"):
        for block in list(backbone.blocks)[-n:]:
            for param in block.parameters():
                param.requires_grad = True
                
                
def _train_phase(model, train_loader, val_loader, loss_fn,
                 cfg_glob, cfg_method,
                 lr, num_epoch, phase_idx, save_path, best_loss, global_step):
    loss_name = cfg_method["loss_name"]
    patience = cfg_glob["PATIENCE"]
    l2_weight_decay = cfg_method.get("l2_weight_decay", 0)

    optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=l2_weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=0)
    patience_counter = 0

                    
    for epoch in range(num_epoch):
            epoch_start = time.time()
            print(f"Phase {phase_idx} | Epoch {epoch+1}/{num_epoch}")
            model.train()
                        
            running_loss = 0.0
            pbar = tqdm(train_loader, desc="Train", leave=False)

            for batch in pbar:
                X = batch[0].to(DEVICE)
                y = batch[1].to(DEVICE).float().view(-1, 1)
                iw = batch[4].to(DEVICE).unsqueeze(1).float() if loss_name in ("nMSE", "nLiteMSE") else None
                pi = batch[5].to(DEVICE).unsqueeze(1).float() if loss_name == "nMSE" else None

                y_pred = model(X)
                loss = loss_fn(y_pred, y, iw, pi)
                running_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss = running_loss / len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.inference_mode():
                for batch in val_loader:
                    X_val = batch[0].to(DEVICE)
                    y_val = batch[1].to(DEVICE).float().view(-1, 1)
                    iw_val = batch[4].to(DEVICE).unsqueeze(1).float() if loss_name in ("nMSE", "nLiteMSE") else None
                    pi_val = batch[5].to(DEVICE).unsqueeze(1).float() if loss_name == "nMSE" else None
                    val_loss += loss_fn(model(X_val), y_val, iw_val, pi_val).item()

            val_loss /= len(val_loader)

            mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=global_step)
            mlflow.log_metric("train_loss", train_loss, step=global_step)
            mlflow.log_metric("val_loss", val_loss, step=global_step)
            mlflow.log_metric("epoch_time_s", time.time() - epoch_start, step=global_step)
            global_step += 1

            scheduler.step()

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                state_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
                torch.save(state_dict, save_path)
                print(f"  → checkpoint saved (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  → early stop at epoch {epoch+1}")
                break

    return best_loss, global_step    


def run_cnn_ft(cfg, timestamp, experiment_id):
    cfg_mod = cfg["model"]
    cfg_glob = cfg["globaux"]
    cfg_method = cfg["cnn_ft_training"]

    if cfg_method["run_execution"] != True:
        return None, None
    
    method_FT = cfg_method["method_FT"]
    learning_rate = cfg_method["learning_rate"]
    num_epoch_head = cfg_method["num_epoch_head"]
    num_epoch_per_phase = cfg_method["num_epoch_per_phase"]
    n_phases = cfg_method["n_phases"]
    lr_decay_factor = cfg_method["lr_decay_factor"]
    mkwargs = cfg_method.get("method_kwargs") or {}
    
    logger.setLevel(cfg_glob.get("VERBOSE", "WARNING"))
    print(f"Start of training by {method_FT}")

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{timestamp}_{cfg_mod}_{method_FT}") as run:
        print(f"MLflow run: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}/runs/{run.info.run_id}")

        train_loader = get_challenge_train_loader(
            batch_size=cfg_glob["BATCH_SIZE"], num_workers=NUM_WORKERS,
            model_name=cfg_mod, augmentation=cfg_method["augmentation"]
        )
        val_loader = get_challenge_val_loader(
            split="val_samp", batch_size=cfg_glob["BATCH_SIZE"],
            num_workers=NUM_WORKERS, model_name=cfg_mod
        )

        model = _build_model(cfg_mod, mkwargs.get("probing_type"), mkwargs.get("hidden_size"), cfg_glob["NUM_CLASSES"])
        model = model.to(DEVICE)
        
        if cfg_glob.get("COMPILE",False) :
            print("Compiling model")
            # compile for faster run but first epoch is slower
            if DEVICE.type == 'mps':
                model = torch.compile(model, backend="aot_eager")
            else:   
                model = torch.compile(model)
            

        _, total_params = _count_trainable_params(model)
        mlflow.log_params({
            **cfg_glob,
            **{k: v for k, v in cfg_method.items() if k != "method_kwargs"},
            **mkwargs,
            "model": cfg_mod,
            "model_tag": f"{cfg_mod}_{method_FT}",
            "timestamp": timestamp,
            "total_params": total_params,
        })

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = CHECKPOINT_DIR / f"{timestamp}_{cfg_mod}_{method_FT}.pt"
        loss_fn = UniversalLossWrapper(LOSS_MAPPING[cfg_method["loss_name"]]())
        best_loss = float("inf")
        global_step = 0

        # Phase 0: head warmup
        _freeze_all(model)
        _unfreeze_head(model)
        trainable, _ = _count_trainable_params(model)
        logger.info(f"Phase 0 — head warmup | trainable: {trainable:,} / {total_params:,} | lr={learning_rate}")
        mlflow.log_metric("trainable_params", trainable, step=0)
        print(f"\n=== Phase 0: head warmup (lr={learning_rate}) ===")
        best_loss, global_step = _train_phase(
            model, train_loader, val_loader, loss_fn, cfg_glob, cfg_method,
            lr=learning_rate, num_epoch=num_epoch_head,
            phase_idx=0, save_path=save_path, best_loss=best_loss, global_step=global_step
        )
        run_evaluation(
            timestamp=timestamp, val_loader=val_loader, loss_name=cfg_method["loss_name"],
            method_FT=method_FT, cfg_glob=cfg_glob, cfg_mod=cfg_mod,
            prefix=method_FT, method_kwargs=mkwargs, index="phase_0", save_val_csv=False
        )

        # Phases 1..n_phases: progressive unfreezing
        blocks_per_phase = int(np.ceil(len(model.model.blocks)/n_phases))
        for phase in range(1, n_phases + 1):
            lr = learning_rate * (lr_decay_factor ** phase)
            n_unfrozen = phase * blocks_per_phase
            print(f"\n=== Phase {phase}: unfreeze top {n_unfrozen} blocks (lr={lr:.2e}) ===")
            _freeze_all(model)
            _unfreeze_top_n_blocks(model, n_unfrozen)
            _unfreeze_head(model)
            trainable, _ = _count_trainable_params(model)
            logger.info(f"Phase {phase} — unfreeze top {n_unfrozen} blocks | trainable: {trainable:,} / {total_params:,} | lr={lr:.2e}")
            mlflow.log_metric("trainable_params", trainable, step=phase)
            best_loss, global_step = _train_phase(
                model, train_loader, val_loader, loss_fn, cfg_glob, cfg_method,
                lr=lr, num_epoch=num_epoch_per_phase,
                phase_idx=phase, save_path=save_path, best_loss=best_loss, global_step=global_step
            )
            run_evaluation(
                timestamp=timestamp, val_loader=val_loader, loss_name=cfg_method["loss_name"],
                method_FT=method_FT, cfg_glob=cfg_glob, cfg_mod=cfg_mod,
                prefix=method_FT, method_kwargs=mkwargs, index=f"phase_{phase}", save_val_csv=False
            )

        # load best checkpoint, log, evaluate, generate submission
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        mlflow.log_artifact(str(save_path))

        _, _, _, df_test = get_challenge_split()

        run_evaluation(
            timestamp=timestamp, val_loader=val_loader, loss_name=cfg_method["loss_name"],
            method_FT=method_FT, cfg_glob=cfg_glob, cfg_mod=cfg_mod,
            prefix=method_FT, method_kwargs=mkwargs, index=None
        )

        test_loader = get_challenge_test_loader(df_test, cfg_glob["BATCH_SIZE"], NUM_WORKERS, model_name=cfg_mod)
        run_test(timestamp, test_loader, method_FT, cfg_mod, method_kwargs=mkwargs)
        save_split_predictions(timestamp, train_loader, "train", method_FT, cfg_mod, mkwargs)

    print(f"fin d'entrainement par {method_FT}")
    return run.info.run_id, method_FT