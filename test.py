import mlflow
import timm
import torch
import pandas as pd

from tqdm import tqdm

from src.config import MODEL_NAME
from src.models import get_model
from src.path import*


def run_test(timestamp,test_loader,method_FT)->None:
    """
    Pipe comple de test:
    - instancie le modèle pré entrainé
    - préparation des données de test
    - inférence sur les donées de test
    - sauvegarde du fichier submission en local et sur le dashbord
    """
     # attribuer le nom au modèle
    model_tag = f"{MODEL_NAME}_{method_FT}"
    # création des dossiers locaux
    HISTORY_DIR.mkdir(parents=True,exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True,exist_ok=True)

    checkpoint_path = CHECKPOINT_DIR / f"{timestamp}_{model_tag}.pt"

     # instanciation du modèle
    model = get_model(MODEL_NAME, num_classes=1,method=method_FT)
    # data_config = timm.data.resolve_model_data_config(model)
    # test_transform = timm.data.create_transform(**data_config, is_training=False)

    model.load_state_dict(torch.load(checkpoint_path,map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # inférence
    results_list = []
    with torch.inference_mode():

        progress_bar = tqdm(enumerate(test_loader),total=len(test_loader),desc="test")
        for batch_idx, (X, *_, filename) in progress_bar:
            # Transfer -> device
            X = X.to(DEVICE)
            y_pred = model(X)
            for i in range(len(X)):

                results_list.append({'filename': filename[i],
                                    'FaceOcclusion': float(y_pred[i])
                                    })          
    results_df = pd.DataFrame(results_list)

    # sauvegarde
        # sauvegarde en local
    submission_path = SUBMISSION_DIR / f"{timestamp}_submission_{model_tag}.csv"
    results_df.to_csv(submission_path,index=False)
        # sauvegarde MLFlow
    mlflow.log_artifact(local_path=submission_path,artifact_path="submission")



