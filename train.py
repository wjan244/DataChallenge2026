# modifier train:
#   - ajouter un scheduler
#   - reprendre la sauvegarde en locale (ne prend que la dernière)

import mlflow
import pandas as pd
import timm
import torch 
import torch.nn as nn

from tqdm import tqdm

from src.config import CHECKPOINT_DIR,HISTORY_DIR, IMG_DIR, MODEL_NAME, DEVICE, BATCH_SIZE, NUM_WORKERS, NUM_EPOCH, LEARNING_RATE, LOSS_NAME,TRAINING_MODE
from src.dataset import Dataset
from src.data_loader import get_challenge_split
from src.models import get_model


def run_train(timestamp):

    # création des dossiers locaux
    CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True)
    HISTORY_DIR.mkdir(parents=True,exist_ok=True)

    # load dataframes
    df_train, _, _ = get_challenge_split()

    # instancier le modèle
    model = get_model(MODEL_NAME, num_classes=1)
        # extraire la configuration des données du modèle
    data_config = timm.data.resolve_model_data_config(model)
    timm_transform = timm.data.create_transform(**data_config, is_training=True)
        # -> DEVICE
    model = model.to(DEVICE)

    # préparation des données
        # Data
    training_set = Dataset(df=df_train,image_dir=IMG_DIR,training=True,transform=timm_transform)
        # DataLoader
    params_train = {'batch_size': BATCH_SIZE,
            'shuffle': True,
            'num_workers': NUM_WORKERS}
    training_generator = torch.utils.data.DataLoader(training_set, **params_train)

    # GD
        # loss
    if LOSS_NAME == "MSE":
        loss_fn = nn.MSELoss()
    elif LOSS_NAME == "BCE":
        loss_fn = nn.BCELoss()
        # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # paramétrisatio MLFlow
    hyper_params = {
        "model":MODEL_NAME,
        "learnin_rate" : LEARNING_RATE,
        "num_epoch": NUM_EPOCH,
        "batch_size": BATCH_SIZE,
        "num_worker": NUM_WORKERS,
        "loss": LOSS_NAME,
        "training_mode": TRAINING_MODE
    }
    mlflow.log_params(hyper_params)

    # entrainement
    save_path = CHECKPOINT_DIR / f"{MODEL_NAME}_{TRAINING_MODE}_{timestamp}.pt"
    best_loss = float('inf')

    for n in range(NUM_EPOCH):
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
        mlflow.log_metric(key="loss",value=final_loss,step=n)

        # sauvegarde du modèle en local
        if final_loss < best_loss:
            best_loss = final_loss
            print(f"sauvegarde du modèle à l'époque {n+1}")
            torch.save(model.state_dict(), save_path)

    # sauvegarde loss en local
    log_path = HISTORY_DIR / f"train_history_loss_{MODEL_NAME}_{TRAINING_MODE}.csv" 
    new_row = pd.DataFrame([{
            "id_run": mlflow.active_run().info.run_id,
            "date": timestamp,
            "modèle": MODEL_NAME,
            "learning_rate": LEARNING_RATE,
            "num_epoch": NUM_EPOCH,
            "loss_name": LOSS_NAME,
            "batch_size": BATCH_SIZE,
            "traing_mode": TRAINING_MODE,
            "final_train_loss": final_loss
        }])    
        # ajout de la nouvelle ligne si non existante
    if log_path.exists():
        new_row.to_csv(log_path, mode='a', header=False, index=False)
    else:
        new_row.to_csv(log_path, index=False)