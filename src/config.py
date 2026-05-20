from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent #pointer directement vers la racine

DATA = BASE_DIR / "data"
IMG_DIR = DATA / "Crop_224_5fp_100K" 
CSV_DIR = DATA / "occlusion_datasets"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
HISTORY_DIR = BASE_DIR / "history"


MODEL_NAME = "mobilenetv3_small_100"