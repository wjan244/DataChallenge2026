import mlflow
import torch
import pandas as pd

from tqdm import tqdm

from src.config import*
from src.models.models import get_model


def _per_sample_aggregate(x):
    """Return a 1-D tensor with one value per batch sample.
    If x has shape [B, T, ...] it collapses non-batch dims by mean.
    If x is scalar-like, returns flattened tensor of length B.
    """
    if torch.is_tensor(x):
        if x.dim() > 1:
            return x.view(x.size(0), -1).mean(dim=1)
        return x.view(-1)
    # fallback: convert to tensor then process
    t = torch.tensor(x)
    if t.dim() > 1:
        return t.view(t.size(0), -1).mean(dim=1)
    return t.view(-1)


def _to_scalar(v):
    """Convert a value (tensor or python) to a float safely.
    If tensor has multiple elements, takes the mean.
    """
    if torch.is_tensor(v):
        v = v.detach().cpu()
        if v.numel() == 1:
            return float(v.item())
        return float(v.view(-1).float().mean().item())
    try:
        return float(v)
    except Exception:
        return float(torch.tensor(v).float().mean().item())


def save_split_predictions(timestamp, loader, split_name, method_FT, cfg_mod, method_kwargs=None):
    """Run inference on a labeled split; save filename/FaceOcclusion(GT)/pred/gender/iw to CSV."""
    model_tag = f"{cfg_mod}_{method_FT}"
    checkpoint_path = CHECKPOINT_DIR / f"{timestamp}_{model_tag}.pt"

    # cfg_mod may be a model name (str) or a dict of model config; handle both
    if isinstance(cfg_mod, dict):
        num_classes = cfg_mod.get("NUM_CLASSES", 1)
    else:
        num_classes = 1

    model = get_model(timestamp=timestamp, cfg_mod=cfg_mod, cfg_method=None, precedent_run_id=None, precedent_method=None, method=method_FT, **(method_kwargs or {}))
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model = model.to(DEVICE)
    model.eval()

    results_list = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"predict {split_name}"):
            X = batch[0].to(DEVICE)

            outputs_split = model(X)
            if isinstance(outputs_split, dict):
                y_pred = outputs_split["head_0"]
            else:
                y_pred = outputs_split

            # ensure one scalar per sample
            y_pred = _per_sample_aggregate(y_pred)
            y_true = _per_sample_aggregate(batch[1])
            genders = _per_sample_aggregate(batch[2])
            filenames = batch[3]
            iws = _per_sample_aggregate(batch[4]) if len(batch) > 4 else None

            for i in range(len(X)):
                results_list.append({
                    'filename': filenames[i],
                    'FaceOcclusion': _to_scalar(y_true[i]),
                    'pred': _to_scalar(y_pred[i]),
                    'gender': _to_scalar(genders[i]),
                    'iw': _to_scalar(iws[i]) if iws is not None else None,
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
    model = get_model(timestamp=timestamp, cfg_mod=cfg_mod, cfg_method=None, precedent_run_id=None, precedent_method=None, method=method_FT, **(method_kwargs or {}))
    
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
            outputs_test = model(X)
            if isinstance(outputs_test, dict):
                y_pred = outputs_test["head_0"]
            else:
                y_pred = outputs_test

            y_pred = _per_sample_aggregate(y_pred)

            for i in range(len(X)):
                results_list.append({'filename': filename[i],
                                     'FaceOcclusion': _to_scalar(y_pred[i]),
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



