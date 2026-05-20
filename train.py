import pandas as pd
import torch 
import torch.nn as nn

from tqdm import tqdm

from src.config import CHECKPOINT_DIR,CSV_DIR,HISTORY_DIR, IMG_DIR, MODEL_NAME
from src.dataset import Dataset
from src.models import get_model

# Load dataframes
df_train_raw = pd.read_csv(CSV_DIR / "train.csv", delimiter=',')
df_test_raw = pd.read_csv(CSV_DIR / "test_students.csv", delimiter=',')

# Remove nan values
df_train_clean = df_train_raw.dropna()
df_test = df_test_raw.dropna().reset_index(drop=True)

# Split Dataframe in train and val
df_val = df_train_clean.iloc[:20000].reset_index(drop=True)
df_train = df_train_clean.iloc[20000:].reset_index(drop=True)

if __name__ == "__main__":

    CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True)
    HISTORY_DIR.mkdir(parents=True,exist_ok=True)

# device adaptatif (local ou cluster de Télécom)
    if torch.backends.mps.is_available():
        device = torch.device("mps")         
    elif torch.cuda.is_available():
        device = torch.device("cuda")       
    else:
        device = torch.device("cpu")

# instancier le modèle
    model = get_model(MODEL_NAME, num_classes=1)
    model = model.to(device)

# préparation des données
    training_set = Dataset(df_train, IMG_DIR)
    validation_set = Dataset(df_val, IMG_DIR)
    test_set = Dataset(df_test, IMG_DIR, training=False)

    params_train = {'batch_size': 64,
            'shuffle': True,
            'num_workers': 0}

    params_val = {'batch_size': 64,
            'shuffle': False,
            'num_workers': 0}

    training_generator = torch.utils.data.DataLoader(training_set, **params_train)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params_val)
    test_generator = torch.utils.data.DataLoader(test_set, **params_val)

# Hyper paramètres
    num_epochs = 1
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# entrainement
    for n in range(num_epochs):
        print(f"Epoch {n+1}")
        model.train()

        progress_bar = tqdm(enumerate(training_generator), total=len(training_generator), desc="Entraînement")
        
        for batch_idx, (X, y, gender, filename) in progress_bar:
            # Transfer to GPU
            X, y = X.to(device), y.to(device)
            y = y.view(-1, 1)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            if loss.isnan():
                print(filename)
                print('label', y)
                print('y_pred', y_pred)
                break

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
# sauvegarde
    save_path = CHECKPOINT_DIR / "mobilenetv3_challenge_epoch1.pt"
    torch.save(model.state_dict(), save_path)
