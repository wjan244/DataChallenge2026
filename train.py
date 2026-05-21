import mlflow
import pandas as pd
import torch 
import torch.nn as nn

from datetime import datetime
from tqdm import tqdm

from src.config import CHECKPOINT_DIR,HISTORY_DIR, IMG_DIR, MODEL_NAME, DEVICE, BATCH_SIZE, NUM_WORKERS, NUM_EPOCH, LEARNING_RATE, LOSS_NAME, MLFLOW_TRACKING_URI 
from src.dataset import Dataset
from src.data_loader import get_challenge_split
from src.models import get_model

# Load dataframes
df_train, _, _ = get_challenge_split()

if __name__ == "__main__":

    CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True)
    HISTORY_DIR.mkdir(parents=True,exist_ok=True)

# instancier le modèle
    model = get_model(MODEL_NAME, num_classes=1)
    model = model.to(DEVICE)

# préparation des données
    training_set = Dataset(df_train, IMG_DIR)

    params_train = {'batch_size': BATCH_SIZE,
            'shuffle': True,
            'num_workers': NUM_WORKERS}

    training_generator = torch.utils.data.DataLoader(training_set, **params_train)

# Hyper paramètres
    num_epochs = NUM_EPOCH
    if LOSS_NAME == "MSE":
        loss_fn = nn.MSELoss()
    elif LOSS_NAME == "BCE":
        loss_fn = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# MLFlow
    timestamp = f"{datetime.now():%Y-%m-%d_%H:%M}"
    # logging
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment = mlflow.set_experiment("DataChallenge_2026")
    # start run
    with mlflow.start_run(experiment_id=experiment.experiment_id,run_name=f"{MODEL_NAME}_{timestamp}"):
        hyper_params = {
            "model":MODEL_NAME,
            "learnin_rate" : LEARNING_RATE,
            "num_epoch": NUM_EPOCH,
            "batch_size": BATCH_SIZE,
            "num_worker": NUM_WORKERS,
            "loss": LOSS_NAME 
        }
        mlflow.log_params(hyper_params)

        # entrainement
        global_step = 0
        final_loss = 0

        for n in range(NUM_EPOCH):
            print(f"Epoch {n+1}")
            model.train()
            running_loss = 0
            progress_bar = tqdm(enumerate(training_generator), total=len(training_generator), desc="Entraînement")
            
            for batch_idx, (X, y, gender, filename) in progress_bar:
                # Transfer to device
                X, y = X.to(DEVICE), y.to(DEVICE)
                y = y.view(-1, 1)
                y_pred = model(X)
                loss = loss_fn(y_pred, y)

                running_loss += loss.item()

                # mlflow metric (loss)
                mlflow.log_metric(key="loss",value=loss.item(),step=global_step)
                global_step+=1

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

    # sauvegarde loss en local
        log_path = HISTORY_DIR / "train_history_loss.csv" 
        new_row = pd.DataFrame([{
                "id_run": mlflow.active_run().info.run_id,
                "date": timestamp,
                "modèle": MODEL_NAME,
                "learning_rate": LEARNING_RATE,
                "num_epoch": NUM_EPOCH,
                "loss_name": LOSS_NAME,
                "batch_size": BATCH_SIZE,
                "final_train_loss": final_loss
            }])    
        
        # ajout de la nouvelle ligne si non existante
        if log_path.exists():
            new_row.to_csv(log_path, mode='a', header=False, index=False)
        else:
            new_row.to_csv(log_path, index=False)

    # sauvegarde du modèle en local
        save_path = CHECKPOINT_DIR / f"{MODEL_NAME}_{timestamp}.pt"
        torch.save(model.state_dict(), save_path)
