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

# modèles - hyper paramètres d'entrainement
MODEL_NAME = "mobilenetv3_small_100"
LEARNING_RATE = 0.001
NUM_EPOCH = 1
LOSS_NAME = "MSE"

# hyper-paramètres Dataloader
BATCH_SIZE = 64
NUM_WORKERS = 0