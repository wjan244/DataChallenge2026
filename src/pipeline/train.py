import inspect
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
from src.models.loss import LOSS_MAPPING, UniversalLossWrapper, build_loss_fn
from src.models.models import get_model
from src.metrics import PWScore



def run_train(timestamp: str, train_loader, val_loader, cfg_mod, cfg_glob, cfg_method,
              precedent_run_id, precedent_method, prefix: str | None = None,) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
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
    num_epoch = cfg_method["num_epoch"]
    loss_name = cfg_method["loss_name"]
    method_FT = cfg_method["method_FT"]
    patience = cfg_glob["PATIENCE"]
    loss_name = cfg_method["loss_name"]

    model_tag = f"{cfg_mod}_{method_FT}"
    
    # # load dataframes
    _, df_val_raw, df_val_samp, df_test = get_challenge_split()

    # instancier le modèle
    cfg_method_kwargs = cfg_method.get("method_kwargs") or {}
    model = get_model(timestamp=timestamp, cfg_mod=cfg_mod, cfg_method=cfg_method, precedent_run_id=precedent_run_id, precedent_method=precedent_method,
                      method=method_FT, **(cfg_method_kwargs or {}))
    
    # -> DEVICE
    model = model.to(DEVICE)

    # GD
    loss_fn = build_loss_fn(loss_name)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epoch, eta_min=0, last_epoch=-1)
    
    # paramétrisatio MLFlow
    hyper_params = {**cfg_glob, **{k: v for k, v in cfg_method.items() if k != "method_kwargs"}, **cfg_method_kwargs}
    hyper_params.update({
        "model": cfg_mod,
        "model_tag": model_tag,
        "time_stamp": timestamp,
        "prefix": prefix
    })
    mlflow.log_params(hyper_params)
    
    # métrique
    score_fn = PWScore()
    # entrainement
    save_path = CHECKPOINT_DIR / f"{timestamp}_{model_tag}.pt"

    # initialisation early stopping
    patience_counter = 0
    # initialize tracking variables
    best_score = float('inf')
    
    train_start = time.time()

    for n in range(num_epoch):
        epoch_start = time.time()
        print(f"Epoch {n+1}")
        model.train()
        running_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Entraînement",leave=True)

        for batch_idx, batch in progress_bar:
         
            X = batch[0].to(DEVICE)
            y = batch[1].float().to(DEVICE).view(-1, 1)
            gender = batch[2].float().to(DEVICE).view(-1, 1)
            pi = batch[5].float().unsqueeze(1).to(DEVICE)
    
            output = model(X)
            # gesion de l'adversarial (output)
            if isinstance(output,dict):
                y_pred = output["head_0"]
                y_pred_genre = output["head_1"]
            else:
                y_pred = output

            loss = loss_fn(y_pred, y, pi, gender)
            # gesion de l'adversarial (loss)
            if isinstance(output,dict):
                criterion_fairness = torch.nn.BCELoss()
                loss_fairness = criterion_fairness(y_pred_genre,gender)
                loss = loss + loss_fairness

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = running_loss/len(train_loader)

        # boucle d'évaluation
        model.eval()
        val_loss = 0
        all_preds, all_targets, all_pi, all_genders = [], [], [], []
        with torch.inference_mode():
            for batch in val_loader:
                # robust unpack for validation batches (same logic as training)
                X_val = batch[0].to(DEVICE)
                y_val = batch[1].float().to(DEVICE).view(-1, 1)
                gender_val = batch[2].float().to(DEVICE).view(-1, 1)
                pi_val = batch[5].float().unsqueeze(1).to(DEVICE)
                
                output_val = model(X_val)

                # gestion de l'adversarial pour la validation
                if isinstance(output_val, dict):
                    y_pred_val = output_val["head_0"]
                    y_pred_genre_val = output_val["head_1"]
                else:
                    y_pred_val = output_val

                batch_loss_val = loss_fn(y_pred_val, y_val, pi_val, gender_val)
                
                # gestion de l'adversarial pour la validation
                if isinstance(output_val, dict):
                    criterion_fairness = torch.nn.BCELoss()
                    loss_fairness_val = criterion_fairness(y_pred_genre_val, gender_val)
                    batch_loss_val = batch_loss_val + loss_fairness_val

                val_loss += batch_loss_val.item()

                all_preds.append(y_pred_val)
                all_targets.append(y_val)
                all_pi.append(pi_val)
                all_genders.append(gender_val)
                
        final_val_loss = val_loss / len(val_loader)

        val_score, val_err_f, val_err_m = score_fn(
            torch.cat(all_preds), torch.cat(all_targets),
            torch.cat(all_pi), torch.cat(all_genders))
        
        val_score = val_score.item()
        val_err_f = val_err_f.item()
        val_err_m = val_err_m.item()

        mlflow.log_metric("lr", optimizer.param_groups[0]["lr"],step=n)
        mlflow.log_metric("val_score", val_score,step=n)
        mlflow.log_metric("val_err_female", val_err_f,step=n)
        mlflow.log_metric("val_err_male", val_err_m,step=n)
        mlflow.log_metric(key="train_loss", value=final_loss, step=n)
        mlflow.log_metric(key="val_loss", value=final_val_loss, step=n)

        # log the time to run the epoch
        epoch_time = time.time() - epoch_start
        mlflow.log_metric("epoch_time_s", epoch_time, step=n)
        
        # update du scheduler
        scheduler.step()

        # sauvegarde du modèle en local et mlflow
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            state_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
            torch.save(state_dict, save_path)
            print(f"checkpoint saved (val_score={val_score:.4f})")
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
        # reload on CPU (device-agnostic) before logging
        model.load_state_dict(torch.load(save_path, map_location='cpu'))
        mlflow.log_artifact(local_path=str(save_path))

    # log the total training time
    mlflow.log_metric("total_train_time_s", time.time() - train_start)

    # récupérer l'id de run_train (à injeter sur le run_train suivant)
    run_id = mlflow.active_run().info.run_id

    return run_id, df_val_raw, df_val_samp, df_test,log_path

