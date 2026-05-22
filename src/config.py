import os
from pathlib import Path

import torch

# chemins vers les dossiers
BASE_DIR = Path(__file__).resolve().parent.parent # pointer directement vers la racine

DATA = BASE_DIR / "data"
IMG_DIR = DATA / "Crop_224_5fp_100K" 
CSV_DIR = DATA / "occlusion_datasets"
SUBMISSION_DIR = BASE_DIR / "submission"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
HISTORY_DIR = BASE_DIR / "history"

# device
if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")         
elif torch.cuda.is_available():
       DEVICE = torch.device("cuda")       
else:
      DEVICE = torch.device("cpu")

# -------------------- PARAMETRES --------------------#
# modèles
MODEL_NAME = 'mobilenetv3_small_075'

# hyper paramètres d'entrainement
LEARNING_RATE = 0.001
NUM_EPOCH = 1
LOSS_NAME = "MSE"

# hyper-paramètres Dataloader
BATCH_SIZE = 16
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()

# fine_tuning - modes d'entrainement
TRAINING_MODE = None
NUM_CLASSES = 1

# distribution des données
N_SAMPLE = 15000