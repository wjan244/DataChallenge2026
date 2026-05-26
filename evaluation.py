import pandas as pd
import timm
import torch
import mlflow

from tqdm import tqdm

from src.config import DEVICE, HISTORY_DIR,SUBMISSION_DIR, IMG_DIR, MODEL_NAME, CHECKPOINT_DIR, BATCH_SIZE, NUM_WORKERS
from src.dataset import Dataset
from src.metrics import metric_fn
from src.models import get_model

def run_evaluation(timestamp,df_val,method_FT,prefix)->None:
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
        # extraire la configuration des données du modèle
    data_config = timm.data.resolve_model_data_config(model)
    val_transform = timm.data.create_transform(**data_config, is_training=False) 
        # -> DEVICE
    model.load_state_dict(torch.load(checkpoint_path,map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # préparation des données
    validation_set = Dataset(df_val, IMG_DIR, training=True, transform=val_transform)

    params_val = {'batch_size': BATCH_SIZE,
            'shuffle': False,
            'num_workers': NUM_WORKERS}
    
    validation_generator = torch.utils.data.DataLoader(validation_set, **params_val)

    # inférence
    results_list = []
    with torch.inference_mode():

        progress_bar = tqdm(enumerate(validation_generator),total=len(validation_generator),desc="validation")
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

    mlflow.log_metric(f"{prefix}_val_score",score)