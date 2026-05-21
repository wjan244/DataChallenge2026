import argparse
import torch
import pandas as pd

from tqdm import tqdm

from src.config import DEVICE,MODEL_NAME,IMG_DIR,HISTORY_DIR,CHECKPOINT_DIR,SUBMISSION_DIR, BATCH_SIZE, NUM_WORKERS
from src.models import get_model
from src.dataset import Dataset
from src.data_loader import get_challenge_split

def run_test(timestamp):

    # Load dataframes
    _, _, df_test = get_challenge_split()

    HISTORY_DIR.mkdir(parents=True,exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True,exist_ok=True)

    # préparation des données
    test_set = Dataset(df_test, IMG_DIR, training=False)

    params_val = {'batch_size': BATCH_SIZE,
            'shuffle': False,
            'num_workers': NUM_WORKERS}

    test_generator = torch.utils.data.DataLoader(test_set, **params_val)

    # instanciation du modèle
    model = get_model(MODEL_NAME, num_classes=1)

    checkpoint_path = CHECKPOINT_DIR / f"{MODEL_NAME}_{timestamp}.pt"

    model.load_state_dict(torch.load(checkpoint_path,map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # inférence
    results_list = []
    with torch.inference_mode():

        progress_bar = tqdm(enumerate(test_generator),total=len(test_generator),desc="test")
        for batch_idx, (X, filename) in progress_bar:
            # Transfer to device
            X = X.to(DEVICE)
            y_pred = model(X)
            for i in range(len(X)):

                results_list.append({'filename': filename[i],
                                    'FaceOcclusion': float(y_pred[i])
                                    })
                
    results_df = pd.DataFrame(results_list)

    # sauvegarde
    submission_path = SUBMISSION_DIR / f"submission_{MODEL_NAME}_{timestamp}.csv"
    results_df.to_csv(submission_path,index=False)


