import pandas as pd
import timm
import torch
import mlflow

from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm

from src.config import*
from src.metrics import metric_fn
from src.models.models import get_model


def run_evaluation(timestamp, val_loader, method_FT, cfg_glob, cfg_mod=None, prefix=None)->None:

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

    # instanciation du modèle
    model = get_model(cfg_mod, num_classes=1,method=method_FT)
    
        # -> DEVICE
    model.load_state_dict(torch.load(checkpoint_path,map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # gestion de l'adaptation de domaine
    if method_FT == "domain_adaptation":
        correct = 0
        total = 0
        with torch.inference_mode():
            f1_metric = BinaryF1Score(threshold=0.5)
            f1_scores = []
            for X, y in val_loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE).view(-1, 1)
                y_pred = model(X)

                # calcul de la prédiction (binaire)
                preds = (y_pred > 0.5).int()
                y_int = y.int()

                correct += (preds == y_int).sum().item()
                total += y_int.size(0)

                # calcul f1_score
                f1_score_batch = f1_metric(preds, y_int)
                f1_scores.append(float(f1_score_batch))

            # aggregation des métriques
            f1_score = float(sum(f1_scores) / len(f1_scores))
            accuracy = correct / total 
            # mlflow
            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("accuracy", accuracy)
            score = f1_score
        
    # gestion du cas général
    else:
        results_list = []
        with torch.inference_mode():

            progress_bar = tqdm(enumerate(val_loader),total=len(val_loader),desc="validation")
            for batch_idx, (X, y, gender, filename,*_) in progress_bar:
                X= X.to(DEVICE)
                y_pred = model(X)

                for i in range(len(X)):
                    results_list.append({
                        'filename': filename[i],
                        'pred': float(y_pred[i]),
                        'FaceOcclusion': float(y[i]),
                        'gender': float(gender[i])
                    })
                    
        results_df = pd.DataFrame(results_list)

        # evaluation
        results_male = results_df.loc[results_df["gender"] == 1.0]
        results_female = results_df.loc[results_df["gender"] == 0.0]
        score = metric_fn(results_female,results_male)
        metric_name = f"{prefix}_val_score"
        # mlflow
        mlflow.log_metric(metric_name,score)

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


    