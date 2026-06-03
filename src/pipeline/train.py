import logging
import mlflow
import pandas as pd
import torch
import torch.nn as nn
import time

from pathlib import Path
from tqdm import tqdm
from src.config import*

from src.data.data_utils import get_challenge_split
from src.metrics import error_fn, metric_fn
from src.models.loss import UniversalLossWrapper, LOSS_MAPPING
from src.models.models import get_model


def run_train(timestamp: str, train_loader, val_loader, cfg_mod, cfg_glob, cfg_method,
              precedent_run_id, precedent_method, prefix: str | None = None,
              pretrained_checkpoint_path: Path | None = None,
              log_competition_metrics: bool = False) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    """
    Pipe d'entrainement complet du modèle défini dans config.py:
    - extraire les poids du run_train précédent
    - instancier le modèle avec les poids du modèle base ou ceux du précédent entrainement
    - préparer les données
    - dataAugmentation définie dans src.transforms
    - backpropagation suivant les paramètres de configuration de config.py
    - optimisation du learning rate avec un cosine scheduler
    - sauvegarde des poids/métriques/paramètres en local et sur le Dashboard MLFlow
    """
    
    # création des dossiers locaux
    CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True)
    HISTORY_DIR.mkdir(parents=True,exist_ok=True)

    # charger les paramètres yaml
    learning_rate = cfg_method["learning_rate"]
    l2_weight_decay = cfg_method.get("l2_weight_decay",0)
    num_epoch = cfg_method["num_epoch"]
    loss_name = cfg_method["loss_name"]
    method_FT = cfg_method["method_FT"]
    patience = cfg_glob["PATIENCE"]
    model_tag = f"{cfg_mod}_{method_FT}"
    
    # # load dataframes
    _, df_val_raw, df_val_samp, df_test = get_challenge_split()
    
    # extraction des poids précédents
    if precedent_run_id:
        precedent_tag = f"{cfg_mod}_{precedent_method}"
        weights = mlflow.artifacts.download_artifacts(
            run_id=precedent_run_id,
            artifact_path=f"{timestamp}_{precedent_tag}.pt")
    else:
        weights = None

    # instancier le modèle
    cfg_method_kwargs = cfg_method.get("method_kwargs") or {}
    model = get_model(cfg_mod, num_classes=1, method=method_FT, weights=weights, **cfg_method_kwargs)

    if pretrained_checkpoint_path is not None:
        state = torch.load(pretrained_checkpoint_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        mlflow.log_param("pretrained_checkpoint", pretrained_checkpoint_path.name)
        print(f"Loaded weights from {pretrained_checkpoint_path.name}")

    # -> DEVICE
    model = model.to(DEVICE)
    
    if cfg_glob.get("COMPILE",False) :
        print("Compiling model")
        # compile for faster run but first epoch is slower
        if DEVICE.type == 'mps':
            model = torch.compile(model, backend="aot_eager")
        else:   
            model = torch.compile(model)

    # GD
    base_loss = LOSS_MAPPING[loss_name]()
    loss_fn = UniversalLossWrapper(base_loss)
        # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_weight_decay)
        # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=num_epoch,eta_min=0,last_epoch=-1)
    
    # paramétrisatio MLFlow
    hyper_params = {**cfg_glob, **{k: v for k, v in cfg_method.items() if k != "method_kwargs"}, **cfg_method_kwargs}
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hyper_params.update({
        "model": cfg_mod,
        "model_tag": model_tag,
        "time_stamp": timestamp,
        "prefix": prefix,
        "total_params": total_params,
        "trainable_params": trainable_params,
    })
    mlflow.log_params(hyper_params)
    
    # entrainement
    save_path = CHECKPOINT_DIR / f"{timestamp}_{model_tag}.pt"
    best_loss = float('inf')

    # initialisation early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    train_start = time.time()

    for n in range(num_epoch):
        epoch_start = time.time()
        print(f"Epoch {n+1}/{num_epoch}")
        model.train()
        running_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Entraînement")

        for batch_idx, batch in progress_bar:
            X = batch[0].to(DEVICE)
            # normalize y to shape [B,1]
            y = batch[1].to(DEVICE).float()
            y = y.squeeze()
            y = y.view(-1, 1)

            # fixer les coefficients par défaut
            iw = None
            pi = None

            # extraction des coefficients en fonction de la loss appelée
            if loss_name == "nLiteMSE":
                    iw = batch[4].to(DEVICE).unsqueeze(1).float()
            elif loss_name == "nMSE":
                iw = batch[4].to(DEVICE).unsqueeze(1).float()
                pi = batch[5].to(DEVICE).unsqueeze(1).float()

            # with torch.autocast(device_type="mps", dtype=torch.float16):
            #     y_pred = model(X)
            #     loss = loss_fn(y_pred, y, iw, pi)
            y_pred = model(X)
            
            loss = loss_fn(y_pred, y, iw, pi)

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = running_loss/len(train_loader)

        # boucle d'évaluation
        model.eval()
        val_loss = 0
        val_records = [] if log_competition_metrics else None
        with torch.inference_mode():
            for batch in val_loader:
                X_val = batch[0].to(DEVICE)
                # normalize y_val to shape [B,1]
                y_val = batch[1].to(DEVICE).float()
                y_val = y_val.squeeze()
                y_val = y_val.view(-1, 1)

                # validation: handle possible presence of iw/pi
                iw_val = None
                pi_val = None
                if loss_name == "nLiteMSE":
                    iw_val = batch[4].to(DEVICE).unsqueeze(1).float()
                elif loss_name == "nMSE":
                    iw_val = batch[4].to(DEVICE).unsqueeze(1).float()
                    pi_val = batch[5].to(DEVICE).unsqueeze(1).float()

                y_pred_val = model(X_val)
                loss_v = loss_fn(y_pred_val, y_val, iw_val, pi_val)
                val_loss += loss_v.item()

                if log_competition_metrics:
                    gender_val = batch[2]
                    if torch.is_tensor(gender_val):
                        gender_val = gender_val.cpu().tolist()
                    preds_cpu = y_pred_val.squeeze(1).cpu().tolist()
                    gt_cpu = y_val.squeeze(1).cpu().tolist()
                    for g, p, gt in zip(gender_val, preds_cpu, gt_cpu):
                        val_records.append({"gender": float(g), "pred": p, "FaceOcclusion": gt})

        final_val_loss = val_loss / len(val_loader)

        # enregistrement des métriques sur MLflow
        mlflow.log_metric(key="lr", value=scheduler.get_last_lr()[0], step=n)
        mlflow.log_metric(key="train_loss", value=final_loss, step=n)
        mlflow.log_metric(key="val_loss", value=final_val_loss, step=n)

        if log_competition_metrics:
            val_df = pd.DataFrame(val_records)
            female_df = val_df[val_df["gender"] == 0.0]
            male_df = val_df[val_df["gender"] == 1.0]
            mlflow.log_metric("val_err_female", error_fn(female_df), step=n)
            mlflow.log_metric("val_err_male", error_fn(male_df), step=n)
            mlflow.log_metric("val_score", metric_fn(female_df, male_df), step=n)

        # log the time to run the epoch
        epoch_time = time.time() - epoch_start
        mlflow.log_metric("epoch_time_s", epoch_time, step=n)
        # update du scheduler
        scheduler.step()

        # sauvegarde du modèle en local et mlflow
        if final_val_loss < best_loss:
            best_loss = final_val_loss
            patience_counter = 0
            print(f"modèle sauvegardé à l'époque {n+1} - Val Loss: {final_val_loss:.4f}")
            state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
            torch.save(state_dict, save_path)
            #torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        # sauvegarde loss en local
        log_path = HISTORY_DIR / f"{timestamp}_train_history_loss_{model_tag}.csv"
        row_dict = hyper_params.copy()

        row_dict.update({
            "id_run": mlflow.active_run().info.run_id,
            "date": timestamp,
            "modèle": cfg_mod,
            "tag": model_tag,
            "epoch": n + 1,
            "final_train_loss": final_loss,
            "final_val_loss": final_val_loss
        })
        new_row = pd.DataFrame([row_dict])

        if log_path.exists():
            new_row.to_csv(log_path, mode='a', header=False, index=False)
        else:
            new_row.to_csv(log_path, index=False)

        if patience_counter >= patience:
            print(f"stagnation de l'entraînement - arrêt à l'époque {n+1}")
            break

    # sauvegarde des poids sur MLFlow (si existants)
    if save_path.exists():
        # handle compiled models
        if hasattr(model, '_orig_mod'):
            model._orig_mod.load_state_dict(torch.load(save_path))
        else:
            model.load_state_dict(torch.load(save_path))
    
        mlflow.log_artifact(local_path=str(save_path))

    # log the total training time
    mlflow.log_metric("total_train_time_s", time.time() - train_start)

    # récupérer l'id de run_train (à injeter sur le run_train suivant)
    run_id = mlflow.active_run().info.run_id

    return run_id, df_val_raw, df_val_samp, df_test,log_path

