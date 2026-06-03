import pandas as pd
import timm
import torch
import mlflow

from torch import nn
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm

from src.config import*
from src.metrics import metric_fn,error_fn
from src.models.models import get_model
from src.models.loss import WeightedLiteMSELoss,UniversalLossWrapper,WeightedMSELoss

# Loss mapping
LOSS_MAPPING = {"MSE":nn.MSELoss,"BCE":nn.BCELoss, "nMSE":WeightedMSELoss, "nLiteMSE":WeightedLiteMSELoss}

def run_evaluation(timestamp, val_loader, method_FT, cfg_glob, loss_name = None, cfg_mod=None, prefix=None, method_kwargs: dict | None = None, index:str=None)->None:

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

    # instanciation du modèle (passer timestamp + cfg_method=None car seuls method & method_kwargs sont disponibles ici)
    model = get_model(timestamp, cfg_mod, None, None, None, num_classes=cfg_glob["NUM_CLASSES"], method=method_FT, **(method_kwargs or {}))
    
        # -> DEVICE
    model.load_state_dict(torch.load(checkpoint_path,map_location='cpu'))
    model = model.to(DEVICE)
    model.eval()

    # gestion de l'adaptation de domaine
    if method_FT == "domain_adaptation":
        
        with torch.inference_mode():
            correct =0
            total = 0

            f1_metric = BinaryF1Score(threshold=0.5).to(DEVICE)
            
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

            progress_bar = tqdm(enumerate(val_loader),total=len(val_loader),desc="validation")
            for batch_idx, batch in progress_bar:
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
                y_pred = model(X)

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

        # evaluation
        results_male = results_df.loc[results_df["gender"] == 1.0]
        results_female = results_df.loc[results_df["gender"] == 0.0]

        # prise en compte des poids pi et iw pour l'évaluation: construire des vecteurs w_female et w_male si disponibles
        if "combined_weights" in results_df.columns and results_df["combined_weights"].notna().any():
            w_female = results_female["combined_weights"].to_numpy() if not results_female.empty else None
            w_male = results_male["combined_weights"].to_numpy() if not results_male.empty else None
            score = metric_fn(results_female, results_male, w=(w_female, w_male))
        else:
            score = metric_fn(results_female, results_male, w=None)
        
    suffix = f"_{index}" if index else ""
    prefix = f"_{prefix}" if prefix else ""
    metric_name = f"score_DataChallenge_{prefix}_{suffix}"
    # loging mlflow
    mlflow.log_metric(metric_name, score)

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


    