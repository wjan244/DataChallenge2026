import mlflow
import timm
import torch
import pandas as pd

from tqdm import tqdm

from src.config import DEVICE,MODEL_NAME,IMG_DIR,HISTORY_DIR,CHECKPOINT_DIR,SUBMISSION_DIR, BATCH_SIZE, NUM_WORKERS
from src.models import get_model
from src.dataset import Dataset

def run_test(timestamp,df_test:pd.DataFrame,method_FT)->None:
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
    data_config = timm.data.resolve_model_data_config(model)
    test_transform = timm.data.create_transform(**data_config, is_training=False)

    model.load_state_dict(torch.load(checkpoint_path,map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # préparation des données
    test_set = Dataset(df_test, IMG_DIR, training=False, transform=test_transform)
        # DataLoader
    params_val = {'batch_size': BATCH_SIZE,
            'shuffle': False,
            'num_workers': NUM_WORKERS}
    
    test_generator = torch.utils.data.DataLoader(test_set, **params_val)

    # inférence
    results_list = []
    with torch.inference_mode():

        progress_bar = tqdm(enumerate(test_generator),total=len(test_generator),desc="test")
        for batch_idx, (X, filename) in progress_bar:
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



