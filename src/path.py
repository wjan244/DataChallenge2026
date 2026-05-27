from pathlib import Path

import torch

# chemins vers les dossiers
BASE_DIR = Path(__file__).resolve().parent.parent # pointer directement vers la racine

DATA = BASE_DIR / "data"
IMG_DIR = DATA / "Crop_224_5fp_100K" 
CSV_DIR = DATA / "occlusion_datasets"

SCREENSHOT_PATH = DATA / "test_distribution.png"
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

# distribution des données
N_SAMPLE = 20000

N_SAMPLE = 500
N_BINS = 20