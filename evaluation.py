import pandas as pd
import timm
import torch
import mlflow

from tqdm import tqdm

from src.config import MODEL_NAME, BATCH_SIZE
from src.path import *
from src.metrics import metric_fn
from src.models import get_model

def run_evaluation(timestamp,val_loader,method_FT,prefix)->None:
    """
    Pipe d'évalualtion:
    - inférence du modèle entrainé sur le dataset eval
    - calcul du score
    - sauvegarde du score en local et sur le Dashboard MLFlow
    """
    # attribuer le nom au modèle
    model_tag = f"{MODEL_NAME}_{method_FT}"

    # création des dossiers locaux et checkpoint_path (dossier d'extraction des poids)
    HISTORY_DIR.mkdir(parents=True,exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True,exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{timestamp}_{model_tag}.pt"

    # instanciation du modèle
    model = get_model(MODEL_NAME, num_classes=1,method=method_FT)
    #     # extraire la configuration des données du modèle
    # data_config = timm.data.resolve_model_data_config(model)
    # val_transform = timm.data.create_transform(**data_config, is_training=False) 
        # -> DEVICE
    model.load_state_dict(torch.load(checkpoint_path,map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # gestion de l'adaptation de domaine
    if method_FT == "domain_adaptation":
        correct = 0
        total = 0
        with torch.inference_mode():
        
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                y_pred = model(X)
                preds = (y_pred > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        score = correct / total
        metric_name = f"{prefix}_val_acc_gender"

    # gestion du cas général
    else:
        results_list = []
        with torch.inference_mode():

            progress_bar = tqdm(enumerate(val_loader),total=len(val_loader),desc="validation")
            for batch_idx, (X, y, gender, filename) in progress_bar:
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

    # sauvegarde du score dans le journal (en local)
    log_path = HISTORY_DIR / f"{timestamp}_eval_history_{model_tag}.csv"

    new_row = pd.DataFrame([{
        "id_run": mlflow.active_run().info.run_id,
        "date":timestamp,
        "modèle": MODEL_NAME,
        "method_FT":method_FT,
        "batch_size":BATCH_SIZE,
        "score":score}])
        # ajout de la nouvelle ligne si non existante
    if log_path.exists():
        new_row.to_csv(log_path, mode='a', header=False, index=False)
    else:
        new_row.to_csv(log_path, index=False)

    mlflow.log_metric(metric_name,score)