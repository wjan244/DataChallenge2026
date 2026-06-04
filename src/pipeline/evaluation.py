import pandas as pd
import timm
import torch
import mlflow

from torch import nn
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm

from src.config import*
from src.metrics import metric_fn,error_fn,PWScore
from src.models.models import get_model
from src.models.loss import WeightedMSELoss, WeightedLiteMSELoss, PWGLoss, UniversalLossWrapper


def run_evaluation(timestamp, val_loader, method_FT, cfg_glob, loss_name = None, cfg_mod=None, prefix=None, method_kwargs: dict | None = None, index:str=None, save_val_csv: bool = True)->None:

    """
    Pipe d'évalualtion:
    - inférence du modèle entrainé sur le dataset eval
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

    # gestion de l'adaptation de domaine
    if method_FT == "domain_adaptation":
        
        with torch.inference_mode():
            correct =0
            total = 0

            f1_metric = BinaryF1Score(threshold=0.5).to(DEVICE)
            #f1_scores = []
            for batch in val_loader:
                # distinguer deux cas pour pouvoir faire l'évaluation de l'adaptation de domaine sur le Dataset du DataChallenge
                if isinstance(batch, (list, tuple)):
                    X = batch[0]
                    y = batch[1]
                # cas général
                else:
                    X, y = batch
                X = X.to(DEVICE)
                
                y = y.to(DEVICE).float()
                y = y.squeeze() # supprimer les dimensions inutiles
                y = y.view(-1, 1) # mettre sous forme d'une colonne
                y_pred = model(X)

                # calcul de la prédiction (binaire)
                preds = (y_pred > 0.5).int()
                y_int = y.int()

                correct += (preds == y_int).sum().item()
                total += y_int.size(0)

                # calcul f1_score
                f1_metric.update(preds,y_int)

            f1_score = float(f1_metric.compute().cpu())
            accuracy = correct / total 
            
            # éviter que les métriques soient écrasées si évaluation sur deux Dataset (cas de Domain_Adaptation)
            suffix = f"_{index}" if index else ""
            mlflow.log_metric(f"f1_score{suffix}", f1_score)
            mlflow.log_metric(f"accuracy{suffix}", accuracy)
            score = f1_score
        
    # gestion du cas général
    else:
        results_list = []
        with torch.inference_mode():

            progress_bar = tqdm(val_loader, total=len(val_loader), desc="validation")
            for batch in progress_bar:
                X = batch[0].to(DEVICE)
                # normalize y to shape [B,1]
                y = batch[1].to(DEVICE).float()
                y = y.squeeze()
                y = y.view(-1, 1)

                # filename (strings/lists -> CPU)
                filename = batch[3]
                # gender: ensure float32 then move to DEVICE (MPS doesn't support float64)
                gender = batch[2]
                if torch.is_tensor(gender):
                    # convert dtype on CPU then move
                    gender = gender.to(torch.float32).to(DEVICE)
                else:
                    try:
                        gender = torch.tensor(gender, dtype=torch.float32, device=DEVICE)
                    except Exception:
                        gender = torch.tensor(gender, dtype=torch.float32)
                        gender = gender.to(DEVICE)


                # fixer les coefficients par défaut
                iw = None
                pi = None

                # extraction des coefficients en fonction de la loss appelée (s'ils existent)
                if loss_name == "nLiteMSE" and len(batch) > 4:
                    iw = batch[4]

                elif loss_name == "nMSE" and len(batch) > 5:
                    iw = batch[4]
                    pi = batch[5]

                # normaliser iw/pi en tenseurs float32 sur DEVICE si présents
                def _to_tensor_on_device(x):
                    if x is None:
                        return None
                    if torch.is_tensor(x):
                        return x.to(torch.float32).to(DEVICE)
                    try:
                        return torch.tensor(x, dtype=torch.float32, device=DEVICE)
                    except Exception:
                        # fallback: create on CPU then move
                        return torch.tensor(x, dtype=torch.float32).to(DEVICE)

                iw = _to_tensor_on_device(iw)
                pi = _to_tensor_on_device(pi)

                # prédictions
                outputs = model(X)
                if isinstance(outputs, dict):
                    y_pred = outputs["head_0"]
                else:
                    y_pred = outputs

                # helper pour récupérer la valeur scalaire de manière sûre
                def _get_item(arr, idx):
                    if arr is None:
                        return None
                    if torch.is_tensor(arr):
                        return float(arr[idx].cpu())
                    try:
                        return float(arr[idx])
                    except Exception:
                        return None

                for i in range(len(X)):
                    iw_val = _get_item(iw, i)
                    pi_val = _get_item(pi, i)
                    combined = None
                    if (iw_val is not None) and (pi_val is not None):
                        combined = float(iw_val * pi_val)

                    row = {
                        'filename': filename[i],
                        'pred': float(y_pred[i].cpu()),
                        'FaceOcclusion': float(y[i].cpu()),
                        'gender': float(gender[i].cpu()),
                        'iw': iw_val,
                        'pi': pi_val,
                        'combined_weights': combined
                    }
                    results_list.append(row)
                    
        results_df = pd.DataFrame(results_list)

        if save_val_csv:
            val_csv_path = SUBMISSION_DIR / f"{timestamp}_submission_{model_tag}" / "val.csv"
            val_csv_path.parent.mkdir(parents=True, exist_ok=True)
            results_df[["filename", "FaceOcclusion", "pred", "gender", "iw"]].to_csv(val_csv_path, index=False)

        # evaluation
        results_male = results_df.loc[results_df["gender"] == 1.0]
        results_female = results_df.loc[results_df["gender"] == 0.0]

        err_female = error_fn(results_female)
        err_male = error_fn(results_male)
        score = metric_fn(results_female, results_male)

    suffix = f"_{index}" if index else ""
    mlflow.log_metric(f"{prefix}_val_score{suffix}", score)
    mlflow.log_metric(f"{prefix}_err_female{suffix}", err_female)
    mlflow.log_metric(f"{prefix}_err_male{suffix}", err_male)

    # sauvegarde du score dans le journal (en local)
    log_path = HISTORY_DIR / f"{timestamp}_eval_history_{model_tag}.csv"

    new_row = pd.DataFrame([{
        "id_run": mlflow.active_run().info.run_id,
        "date":timestamp,
        "modèle": cfg_mod,
        "method_FT":method_FT,
        "batch_size":cfg_glob["BATCH_SIZE"],
        "score":score}])
        # ajout de la nouvelle ligne si non existante
    if log_path.exists():
        new_row.to_csv(log_path, mode='a', header=False, index=False)
    else:
        new_row.to_csv(log_path, index=False)
        
# ====================================
        
def save_results(model, timestamp, train_loader, val_loader, test_loader,
                 loss_name=None, cfg_mod=None, method_FT=None) ->None:
    """
    Run inference on train/val/test splits and save per-split CSVs.
    - train.csv / val.csv : filename, FaceOcclusion (GT), pred, gender, iw
    - test.csv            : filename, FaceOcclusion (pred), gender='x'  ← submission format
    Also logs competition scores to MLflow for train and val.
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    model_tag = f"{cfg_mod}_{method_FT}"
    out_dir = SUBMISSION_DIR / f"{timestamp}_submission_{model_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    for split, loader in zip(['train', 'val', 'test'], [train_loader, val_loader, test_loader]):
        is_test = split == 'test'
        results_list = []

        with torch.inference_mode():
            pbar = tqdm(loader, total=len(loader), desc=split)
            for batch in pbar:
                X        = batch[0].to(DEVICE)
                y_pred   = model(X)
                filename = batch[1] if is_test else batch[3]

                if not is_test:
                    y      = batch[1].float().to(DEVICE).view(-1, 1)
                    gender = batch[2].float().to(DEVICE).view(-1, 1)
                    iw = batch[4].float().to(DEVICE).unsqueeze(1) if loss_name in ("nMSE", "nLiteMSE", "PGWLoss", "PGWLossRegularized") else None
                    pi = batch[5].float().to(DEVICE).unsqueeze(1) if loss_name in ("nMSE", "PGWLoss", "PGWLossRegularized") else None

                for i in range(len(X)):
                    if is_test:
                        results_list.append({
                            'filename':      filename[i],
                            'FaceOcclusion': float(y_pred[i].cpu()),
                            'gender':        'x',
                        })
                    else:
                        results_list.append({
                            'filename':      filename[i],
                            'FaceOcclusion': float(y[i].cpu()),
                            'pred':          float(y_pred[i].cpu()),
                            'gender':        float(gender[i].cpu()),
                            'iw':            float(iw[i].cpu()) if iw is not None else None,
                            'pi':            float(pi[i].cpu()) if pi is not None else None,
                        })

        results_df = pd.DataFrame(results_list)

        if is_test:
            # submission format: filename, FaceOcclusion (prediction), gender='x'
            results_df[["filename", "FaceOcclusion", "gender"]].to_csv(out_dir / "test.csv", index=False)
        elif split == "val":
            results_df[["filename", "FaceOcclusion", "pred", "gender", "iw"]].to_csv(out_dir / f"{split}.csv", index=False)

            # unshifted
            results_female = results_df[results_df["gender"] == 0.0]
            results_male   = results_df[results_df["gender"] == 1.0]
            err_female = error_fn(results_female)
            err_male   = error_fn(results_male)
            score      = metric_fn(results_female, results_male)

            tag = model_tag
            mlflow.log_metric(f"End_score_{split}_not_shifted", score)
            mlflow.log_metric(f"End_err_female_{split}_not_shifted", err_female)
            mlflow.log_metric(f"End_err_male_{split}_not_shifted", err_male)

            # shifted score (iw*pi weighted) on full split via PWScore
            if results_df["iw"].notna().all() and results_df["pi"].notna().all():
                score_fn = PWScore()
                # transform to tensor
                _t = lambda col: torch.tensor(results_df[col].values, dtype=torch.float32).view(-1, 1)
                score_shifted, err_f_shifted, err_m_shifted = score_fn(
                    _t("pred"), _t("FaceOcclusion"), _t("iw"), _t("pi"), _t("gender")
                )
                mlflow.log_metric(f"End_score_{split}_shifted_iw", float(score_shifted))
                mlflow.log_metric(f"End_err_female_{split}_shifted_iw", float(err_f_shifted))
                mlflow.log_metric(f"End_err_male_{split}_shifted_iw", float(err_m_shifted))