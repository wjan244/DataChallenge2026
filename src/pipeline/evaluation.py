import pandas as pd
import timm
import torch
import mlflow

from torch import nn
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm

from src.config import*
from src.metrics import PWScore
from src.models.models import get_model


def run_evaluation(timestamp, val_loader, method_FT, cfg_glob, loss_name = None, cfg_mod=None, prefix=None, method_kwargs: dict | None = None, index:str=None, save_val_csv: bool = True)->None:

    """
    Pipe d'évalualtion:
    - inférence du modèle entrainé
    - calcul du score
    - sauvegarde du score en local et sur le Dashboard MLFlow
    """

    # attribuer le nom au modèle
    model_tag = f"{cfg_mod}_{method_FT}"

    # création des dossiers locaux et checkpoint_path (dossier d'extraction des poids)
    HISTORY_DIR.mkdir(parents=True,exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True,exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{timestamp}_{model_tag}.pt"

    # instanciation du modèle (get_model signature requires timestamp, cfg_mod, cfg_method, precedent_run_id, precedent_method)
    # call with explicit keyword args to avoid duplicate 'num_classes' if present in method_kwargs
    model = get_model(timestamp=timestamp, cfg_mod=cfg_mod, cfg_method=None, precedent_run_id=None, precedent_method=None, method=method_FT, **(method_kwargs or {}))
    
        # -> DEVICE
    model.load_state_dict(torch.load(checkpoint_path,map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # métrique
    score_fn = PWScore()
        
    # gestion du cas général
    all_preds, all_targets, all_genders = [], [], []
    with torch.inference_mode():

        progress_bar = tqdm(val_loader, total=len(val_loader), desc="validation")
        for batch in progress_bar:
            X_val = batch[0].to(DEVICE)
            y_val = batch[1].float().to(DEVICE).view(-1, 1)
            gender_val = batch[2].float().to(DEVICE).view(-1, 1)
            
            output_val = model(X_val)

            # gesion de l'adversarial (output)
            if isinstance(output_val, dict):
                y_pred_val = output_val["head_0"]
            else:
                y_pred_val = output_val
                
            all_preds.append(y_pred_val)
            all_targets.append(y_val)
            all_genders.append(gender_val)

        # Agréger les résultats de tous les lots
        all_y_pred = torch.cat(all_preds)
        all_y_true = torch.cat(all_targets)
        all_gender = torch.cat(all_genders)

        # Calculer le score final avec toutes les données
        val_score, val_err_f, val_err_m = score_fn(
            all_y_pred, all_y_true, all_gender
        )

        # Journaliser les métriques
        metrics_to_log = {"val_score": val_score}
        if loss_name == "PWGLoss":
            metrics_to_log.update({
                "val_err_f": val_err_f,
                "val_err_m": val_err_m,
            })
        
        if prefix:
            metrics_to_log = {f"{prefix}_{k}": v for k, v in metrics_to_log.items()}

        mlflow.log_metrics(metrics_to_log)

        # sauvegarde loss en local
        log_path = HISTORY_DIR / f"{timestamp}_train_history_loss_{model_tag}.csv"

        row ={
            "id_run": mlflow.active_run().info.run_id,
            "date": timestamp,
            "modèle": cfg_mod,
            "tag": model_tag}
        
        new_row = pd.DataFrame([row])

        if log_path.exists():
            new_row.to_csv(log_path, mode='a', header=False, index=False)
        else:
            new_row.to_csv(log_path, index=False)

        
# # ====================================
        
# def save_results(model, timestamp, train_loader, val_loader, test_loader,
#                  loss_name=None, cfg_mod=None, method_FT=None) ->None:
#     """
#     Run inference on train/val/test splits and save per-split CSVs.
#     - train.csv / val.csv : filename, FaceOcclusion (GT), pred, gender, iw
#     - test.csv            : filename, FaceOcclusion (pred), gender='x'  ← submission format
#     Also logs competition scores to MLflow for train and val.
#     """
#     HISTORY_DIR.mkdir(parents=True, exist_ok=True)
#     SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

#     model_tag = f"{cfg_mod}_{method_FT}"
#     out_dir = SUBMISSION_DIR / f"{timestamp}_submission_{model_tag}"
#     out_dir.mkdir(parents=True, exist_ok=True)

#     model.eval()

#     for split, loader in zip(['train', 'val', 'test'], [train_loader, val_loader, test_loader]):
#         is_test = split == 'test'
#         results_list = []

#         with torch.inference_mode():
#             pbar = tqdm(loader, total=len(loader), desc=split)
#             for batch in pbar:
#                 X        = batch[0].to(DEVICE)
#                 y_pred   = model(X)
#                 filename = batch[1] if is_test else batch[3]

#                 if not is_test:
#                     y      = batch[1].float().to(DEVICE).view(-1, 1)
#                     gender = batch[2].float().to(DEVICE).view(-1, 1)
#                     iw = batch[4].float().to(DEVICE).unsqueeze(1) if loss_name in ("nMSE", "nLiteMSE", "PGWLoss", "PGWLossRegularized") else None
#                     pi = batch[5].float().to(DEVICE).unsqueeze(1) if loss_name in ("nMSE", "PGWLoss", "PGWLossRegularized") else None

#                 for i in range(len(X)):
#                     if is_test:
#                         results_list.append({
#                             'filename':      filename[i],
#                             'FaceOcclusion': float(y_pred[i].cpu()),
#                             'gender':        'x',
#                         })
#                     else:
#                         results_list.append({
#                             'filename':      filename[i],
#                             'FaceOcclusion': float(y[i].cpu()),
#                             'pred':          float(y_pred[i].cpu()),
#                             'gender':        float(gender[i].cpu()),
#                             'iw':            float(iw[i].cpu()) if iw is not None else None,
#                             'pi':            float(pi[i].cpu()) if pi is not None else None,
#                         })

#         results_df = pd.DataFrame(results_list)

#         if is_test:
#             # submission format: filename, FaceOcclusion (prediction), gender='x'
#             results_df[["filename", "FaceOcclusion", "gender"]].to_csv(out_dir / "test.csv", index=False)
#         elif split == "val":
#             results_df[["filename", "FaceOcclusion", "pred", "gender", "iw"]].to_csv(out_dir / f"{split}.csv", index=False)

#             # unshifted
#             results_female = results_df[results_df["gender"] == 0.0]
#             results_male   = results_df[results_df["gender"] == 1.0]
#             err_female = error_fn(results_female)
#             err_male   = error_fn(results_male)
#             score      = metric_fn(results_female, results_male)

#             tag = model_tag
#             mlflow.log_metric(f"End_score_{split}_not_shifted", score)
#             mlflow.log_metric(f"End_err_female_{split}_not_shifted", err_female)
#             mlflow.log_metric(f"End_err_male_{split}_not_shifted", err_male)

#             # shifted score (iw*pi weighted) on full split via PWScore
#             if results_df["iw"].notna().all() and results_df["pi"].notna().all():
#                 score_fn = PWScore()
#                 # transform to tensor
#                 _t = lambda col: torch.tensor(results_df[col].values, dtype=torch.float32).view(-1, 1)
#                 score_shifted, err_f_shifted, err_m_shifted = score_fn(
#                     _t("pred"), _t("FaceOcclusion"), _t("iw"), _t("pi"), _t("gender")
#                 )
#                 mlflow.log_metric(f"End_score_{split}_shifted_iw", float(score_shifted))
#                 mlflow.log_metric(f"End_err_female_{split}_shifted_iw", float(err_f_shifted))
#                 mlflow.log_metric(f"End_err_male_{split}_shifted_iw", float(err_m_shifted))