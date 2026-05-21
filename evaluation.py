import pandas as pd
import torch
import mlflow

from datetime import datetime
from tqdm import tqdm

from src.config import DEVICE, HISTORY_DIR,SUBMISSION_DIR, IMG_DIR, MODEL_NAME, CHECKPOINT_DIR, BATCH_SIZE, NUM_WORKERS, LEARNING_RATE,LOSS_NAME, NUM_EPOCH, MLFLOW_TRACKING_URI
from src.data_loader import get_challenge_split
from src.dataset import Dataset
from src.metrics import metric_fn
from src.models import get_model

# load data
_, df_val, _ = get_challenge_split()


if __name__ == "__main__":

    HISTORY_DIR.mkdir(parents=True,exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True,exist_ok=True)

# préparation des données
    validation_set = Dataset(df_val, IMG_DIR)

    params_val = {'batch_size': BATCH_SIZE,
            'shuffle': False,
            'num_workers': NUM_WORKERS}
    
    validation_generator = torch.utils.data.DataLoader(validation_set, **params_val)

# instanciation du modèle
    model = get_model(MODEL_NAME, num_classes=1)

    checkpoint_path = CHECKPOINT_DIR / "mobilenetv3_challenge_epoch1.pt"

    model.load_state_dict(torch.load(checkpoint_path,map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

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
    timestamp = f"{datetime.now():%Y-%m-%d_%H:%M}"
    log_path = HISTORY_DIR / "eval_history.csv"

    new_row = pd.DataFrame([{
        "date":timestamp,
        "modèle": MODEL_NAME,
        "learning_rate": LEARNING_RATE,
        "num_epoch": NUM_EPOCH,
        "loss": LOSS_NAME,
        "batch_size":BATCH_SIZE,
        "score":score}])
    
    # ajout de la nouvelle ligne si non existante
    if log_path.exists():
        new_row.to_csv(log_path, mode='a', header=False, index=False)
    else:
        new_row.to_csv(log_path, index=False)

    # MLFlow
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    experiment = mlflow.set_experiment("DataChallenge_2026")

    with mlflow.start_run(experiment_id=experiment.experiment_id,run_name=f"{MODEL_NAME}_{timestamp}"):
        
        params_mlflow = new_row.drop(columns=["score"]).iloc[0].to_dict()

        mlflow.log_params(params_mlflow)
        mlflow.log_metric("val_score",score)