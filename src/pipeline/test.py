import mlflow
import torch
import pandas as pd

from tqdm import tqdm

from src.config import*
from src.models.models import get_model


def save_split_predictions(timestamp, loader, split_name, method_FT, cfg_mod, method_kwargs=None):
    """Run inference on a labeled split; save filename/FaceOcclusion(GT)/pred/gender/iw to CSV."""
    model_tag = f"{cfg_mod}_{method_FT}"
    checkpoint_path = CHECKPOINT_DIR / f"{timestamp}_{model_tag}.pt"

    # cfg_mod may be a model name (str) or a dict of model config; handle both
    if isinstance(cfg_mod, dict):
        num_classes = cfg_mod.get("NUM_CLASSES", 1)
    else:
        num_classes = 1

    model = get_model(timestamp, cfg_mod, None, None, None, num_classes=num_classes, method=method_FT, **(method_kwargs or {}))
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model = model.to(DEVICE)
    model.eval()

    results_list = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"predict {split_name}"):
            X = batch[0].to(DEVICE)
            y_pred = model(X).view(-1)
            y_true = batch[1].view(-1)
            genders = batch[2]
            filenames = batch[3]
            iws = batch[4]
            for i in range(len(X)):
                results_list.append({
                    'filename': filenames[i],
                    'FaceOcclusion': float(y_true[i]),
                    'pred': float(y_pred[i]),
                    'gender': float(genders[i]),
                    'iw': float(iws[i]),
                })

    results_df = pd.DataFrame(results_list)
    save_path = SUBMISSION_DIR / f"{timestamp}_submission_{model_tag}" / f"{split_name}.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(save_path, index=False)


def run_test(timestamp,cfg_glob,test_loader,method_FT,cfg_mod, method_kwargs: dict | None = None)->None:
    """
    Pipe comple de test:
    - instancie le modèle pré entrainé
    - préparation des données de test
    - inférence sur les donées de test
    - sauvegarde du fichier submission en local et sur le dashbord
    """
     # attribuer le nom au modèle
    model_tag = f"{cfg_mod}_{method_FT}"
    # création des dossiers locaux
    HISTORY_DIR.mkdir(parents=True,exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True,exist_ok=True)

    checkpoint_path = CHECKPOINT_DIR / f"{timestamp}_{model_tag}.pt"

     # instanciation du modèle
    model = get_model(timestamp, cfg_mod, None, None, None, num_classes=cfg_glob['NUM_CLASSES'], method=method_FT, **(method_kwargs or {}))
    
    model.load_state_dict(torch.load(checkpoint_path,map_location='cpu'))
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
                                    'FaceOcclusion': float(y_pred[i]),
                                    'gender' : 'x'
                                    })          
    results_df = pd.DataFrame(results_list)

    # sauvegarde
        # sauvegarde en local
    submission_path = SUBMISSION_DIR / f"{timestamp}_submission_{model_tag}" / "test.csv"
    submission_path.parent.mkdir(parents=True,exist_ok=True)
    results_df.to_csv(submission_path,index=False)
        # sauvegarde MLFlow
    mlflow.log_artifact(local_path=submission_path,artifact_path="submission")



