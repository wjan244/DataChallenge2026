import mlflow
import pandas as pd
import timm
import torch 
import torch.nn as nn

from torchvision.transforms import v2
from tqdm import tqdm

from src.config import CHECKPOINT_DIR,HISTORY_DIR, IMG_DIR, MODEL_NAME, DEVICE, BATCH_SIZE, NUM_WORKERS, NUM_EPOCH, LEARNING_RATE, LOSS_NAME,TRAINING_MODE
from src.dataset import Dataset
from src.data_utils import get_challenge_split
from src.models import get_model
from src.transforms import get_augmentation_transforms


def run_train(timestamp):

    # création des dossiers locaux
    CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True)
    HISTORY_DIR.mkdir(parents=True,exist_ok=True)

    # load dataframes
    df_train, df_val, df_test = get_challenge_split()

    # instancier le modèle
    model = get_model(MODEL_NAME, num_classes=1)
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
        # loss
    if LOSS_NAME == "MSE":
        loss_fn = nn.MSELoss()
    elif LOSS_NAME == "BCE":
        loss_fn = nn.BCELoss()
        # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        # scheduler 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=NUM_EPOCH,eta_min=0,last_epoch=-1)

    # paramétrisatio MLFlow
    hyper_params = {
        "model":MODEL_NAME,
        "learnin_rate" : scheduler.get_last_lr()[0],
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
        mlflow.log_metric(key="lr",value=scheduler.get_last_lr()[0],step=n)
        mlflow.log_metric(key="loss",value=final_loss,step=n)
        # update du scheduler
        scheduler.step()

        # sauvegarde du modèle en local et mlflow
        if final_loss < best_loss:
            best_loss = final_loss
            torch.save(model.state_dict(), save_path)
            print(f"modèle sauvegarde en local à l'époque {n+1}")

        # sauvegarde loss en local
        log_path = HISTORY_DIR / f"train_history_loss_{MODEL_NAME}_{TRAINING_MODE}.csv" 
        new_row = pd.DataFrame([{
                "id_run": mlflow.active_run().info.run_id,
                "date": timestamp,
                "modèle": MODEL_NAME,
                "learning_rate": LEARNING_RATE,
                "epoch": n+1,
                "num_epoch": NUM_EPOCH,
                "loss_name": LOSS_NAME,
                "batch_size": BATCH_SIZE,
                "traing_mode": TRAINING_MODE,
                "final_train_loss": final_loss
            }])   
        
        if log_path.exists():
            new_row.to_csv(log_path, mode='a', header=False, index=False)
        else:
            new_row.to_csv(log_path, index=False) 
             
    # sauvegarde des poids sur MLFlow
    model.load_state_dict(torch.load(save_path))
    mlflow.pytorch.log_model(model,artifact_path=f"{MODEL_NAME}_{TRAINING_MODE}")

    return df_val, df_test

