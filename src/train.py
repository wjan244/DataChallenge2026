import mlflow
import pandas as pd
import torch 
import torch.nn as nn

from pathlib import Path
from tqdm import tqdm
from src.config import DEVICE

from src.data_utils import get_challenge_split
from src.loss import WeightedMSELoss, UniversalLossWrapper
from src.metrics import metric_fn
from src.models import get_model
from src.path import *

LOSS_MAPPING = {"MSE":nn.MSELoss,"BCE":nn.BCELoss, "nMSE":WeightedMSELoss}

def run_train(timestamp: str, train_loader, val_loader, cfg_mod, cfg_glob, cfg_method,
              precedent_run_id=None, precedent_method=None, prefix: str | None = None,
              **kwargs) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
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
    model = get_model(cfg_mod, num_classes=1,method=method_FT,weights=weights)
         # -> DEVICE
    model = model.to(DEVICE)
      
    # GD
    base_loss = LOSS_MAPPING[loss_name]()
    loss_fn = UniversalLossWrapper(base_loss)
        # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # scheduler 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=num_epoch,eta_min=0,last_epoch=-1)

    # paramétrisatio MLFlow
    hyper_params = {**cfg_glob, **cfg_method}
    hyper_params.update({
        "model": cfg_mod,
        "model_tag": model_tag,
        "time_stamp": timestamp,
        "prefix": prefix
    })
    mlflow.log_params(hyper_params)

    # entrainement
    save_path = CHECKPOINT_DIR / f"{timestamp}_{model_tag}.pt"
    best_loss = float('inf')

    # initialisation early stopping
    best_loss = float('inf')
    patience_counter = 0

    for n in range(num_epoch):
        print(f"Epoch {n+1}")
        model.train()
        running_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Entraînement")
        
        for batch_idx, batch in progress_bar:
            X, y = batch[0].to(DEVICE), batch[1].to(DEVICE).view(-1, 1)
            
            #uniquement pour Dataset Challenge
            iw = batch[4].to(DEVICE).view(-1, 1) if len(batch) == 5 else None
            
            y_pred = model(X)
            loss = loss_fn(y_pred, y, iw)

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        final_loss = running_loss/len(train_loader)

        # boucle d'évaluation
        model.eval()
        val_loss = 0
        with torch.inference_mode(): 
            for batch in val_loader:
                
                X_val, y_val = batch[0].to(DEVICE), batch[1].to(DEVICE).view(-1, 1)
                iw_val = batch[4].to(DEVICE).view(-1, 1) if len(batch) == 5 else None
                
                y_pred_val = model(X_val)
                loss_v = loss_fn(y_pred_val, y_val, iw_val)
                val_loss += loss_v.item()

        final_val_loss = val_loss / len(val_loader)
        
        # enregistrement des métriques sur MLflow
        mlflow.log_metric(key="lr", value=scheduler.get_last_lr()[0], step=n)
        mlflow.log_metric(key="train_loss", value=final_loss, step=n)
        mlflow.log_metric(key="val_loss", value=final_val_loss, step=n)
        
        # update du scheduler
        scheduler.step()

        # sauvegarde du modèle en local et mlflow
        if final_val_loss < best_loss:
            best_loss = final_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"modèle sauvegardé à l'époque {n+1} - Val Loss: {final_val_loss:.4f}")
        else: 
            patience_counter += 1

        # sauvegarde loss en local
        log_path = HISTORY_DIR / f"{timestamp}_train_history_loss_{model_tag}.csv" 
        row_dict = {**cfg_glob, **cfg_method}
        
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
             
    # sauvegarde des poids sur MLFlow
    model.load_state_dict(torch.load(save_path))
    mlflow.log_artifact(local_path=str(save_path))

    # récupérer l'id de run_train (à injeter sur le run_train suivant)
    run_id = mlflow.active_run().info.run_id

    return run_id, df_val_raw, df_val_samp, df_test,log_path

