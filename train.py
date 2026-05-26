import mlflow
import pandas as pd
import timm
import torch 
import torch.nn as nn

from pathlib import Path
from torchvision.transforms import v2
from tqdm import tqdm

from src.config import CHECKPOINT_DIR,HISTORY_DIR, IMG_DIR, MODEL_NAME, DEVICE, BATCH_SIZE, NUM_WORKERS, PATIENCE
from src.dataset import Dataset
from src.data_utils import get_challenge_split
from src.models import get_model
from src.transforms import get_augmentation_transforms

LOSS_MAPPING = {"MSE":nn.MSELoss,"BCE":nn.BCELoss}

def run_train(timestamp:str,loss_name,method_FT,learning_rate,num_epoch,precedent_run_id=None,precedent_method=None,prefix:str|None=None)->tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    """
    Pipe d'entrainement complet du modèle défini dans config.py:
    - extraire les poids du run_train précédent
    - instancier le modèle avec les poids du modèle base ou ceux du précédent entrainement
    - préparer les données
    - dataAugmentation définie dans src.transforms
    - backpropagation suivant les paramètres de configuration de config.py
    - optimisation du learning rate avec un cosine scheduler
    - sauvegarde des poids/métriques/paramètres en local et sur le Dashboard MLFlow
    """

    # création des dossiers locaux
    CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True)
    HISTORY_DIR.mkdir(parents=True,exist_ok=True)

    # attribuer le nom au modèle
    model_tag = f"{MODEL_NAME}_{method_FT}"
    # load dataframes
    df_train, df_val_raw, df_val_samp, df_test = get_challenge_split()
    
    # extraction des poids précédents
    if precedent_run_id:
        precedent_tag = f"{MODEL_NAME}_{precedent_method}"
        weights = mlflow.artifacts.download_artifacts(
            run_id=precedent_run_id,
            artifact_path=f"{timestamp}_{precedent_tag}.pt")
    else:
        weights = None

    # instancier le modèle
    model = get_model(MODEL_NAME, num_classes=1,method=method_FT,weights=weights)
         # -> DEVICE
    model = model.to(DEVICE)
        # extraire la configuration des données du modèle
    data_config = timm.data.resolve_model_data_config(model)
    timm_transform = timm.data.create_transform(**data_config, is_training=False)
   
    # dataAugmentation
        # instantier les augmentations
    augment_transform = get_augmentation_transforms()
        # pipeLine Transform (model + augmentation)
    transform_pipeline = v2.Compose([augment_transform,timm_transform])

    # préparation des données
        # Data
    training_set = Dataset(df=df_train,image_dir=IMG_DIR,training=True,transform=transform_pipeline)
        # DataLoader
    params_train = {'batch_size': BATCH_SIZE,
            'shuffle': True,
            'num_workers': NUM_WORKERS}
    training_generator = torch.utils.data.DataLoader(training_set, **params_train)

    # GD
    loss_fn = LOSS_MAPPING[loss_name]()
        # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # scheduler 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=num_epoch,eta_min=0,last_epoch=-1)

    # paramétrisatio MLFlow
    hyper_params = {
        "model":MODEL_NAME,
        "tag": model_tag,
        "learnin_rate" : scheduler.get_last_lr()[0],
        "num_epoch": num_epoch,
        "batch_size": BATCH_SIZE,
        "num_worker": NUM_WORKERS,
        "loss": loss_name,
        "training_mode": method_FT,
        "time_stamp":timestamp,
        "prefix":prefix
    }
    mlflow.log_params(hyper_params)

    # entrainement
    save_path = CHECKPOINT_DIR / f"{timestamp}_{model_tag}.pt"
    best_loss = float('inf')

    # initialisation early stopping
    best_loss = float('inf')
    patience = PATIENCE
    patience_counter = 0

    for n in range(num_epoch):
        print(f"Epoch {n+1}")
        model.train()
        running_loss = 0
        progress_bar = tqdm(enumerate(training_generator), total=len(training_generator), desc="Entraînement")
        
        for batch_idx, (X, y, gender, filename) in progress_bar:
            # Transfert -> device
            X, y = X.to(DEVICE), y.to(DEVICE)
            y = y.view(-1, 1)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            running_loss += loss.item()

            if loss.isnan():
                print(filename)
                print('label', y)
                print('y_pred', y_pred)
                break

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        final_loss = running_loss/len(training_generator)
        # métrique mlflow (loss - pas = époque)
        mlflow.log_metric(key="lr",value=scheduler.get_last_lr()[0],step=n)
        mlflow.log_metric(key="loss",value=final_loss,step=n)
        # update du scheduler
        scheduler.step()

        # sauvegarde du modèle en local et mlflow
        if final_loss < best_loss:
            best_loss = final_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"modèle sauvegarde à l'époque {n+1}")
        else: 
            patience_counter +=1

        # sauvegarde loss en local
        log_path = HISTORY_DIR / f"{timestamp}_train_history_loss_{model_tag}.csv" 
        new_row = pd.DataFrame([{
                "id_run": mlflow.active_run().info.run_id,
                "date": timestamp,
                "modèle": MODEL_NAME,
                "tag": model_tag,
                "learning_rate": learning_rate,
                "epoch": n+1,
                "num_epoch": num_epoch,
                "loss_name": loss_name,
                "batch_size": BATCH_SIZE,
                "traing_mode": method_FT,
                "final_train_loss": final_loss
            }])   
        
        if log_path.exists():
            new_row.to_csv(log_path, mode='a', header=False, index=False)
        else:
            new_row.to_csv(log_path, index=False) 

        if patience_counter >= patience:
            print(f"stagnation de l'entraînement - arrêt à l'époque {n+1}")
            break
             
    # sauvegarde des poids sur MLFlow
    model.load_state_dict(torch.load(save_path))
    mlflow.log_artifact(local_path=str(save_path))

    # récupérer l'id de run_train (à injeter sur le run_train suivant)
    run_id = mlflow.active_run().info.run_id

    return run_id, df_val_raw, df_val_samp, df_test,log_path

