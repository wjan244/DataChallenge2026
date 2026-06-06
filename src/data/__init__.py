import pandas as pd

from src.config import*
from src.config_utils import load_config

df_train_raw = pd.read_csv(CSV_DIR / "train.csv", delimiter=',')
df_test_raw = pd.read_csv(CSV_DIR / "test_students.csv", delimiter=',')

cfg_glob            = load_config(CONFIG_DEFAULT).get("globaux")
N_BINS              = cfg_glob["N_BINS"]
N_SAMPLE            = cfg_glob["N_SAMPLE"]
eps                 = float(cfg_glob.get("EPS"))
N_BINS_GENDER       = cfg_glob.get("N_BINS_GENDER", 30)
ALPHA_SMOOTH        = cfg_glob.get("ALPHA_SMOOTH", 50)



cfg_augmentation    = load_config(CONFIG_DEFAULT).get("augmentation", {})
augmentation_factor = cfg_augmentation.get('augmentation_factor')
augmentation_status = cfg_augmentation["status"]
warm_start          = float(cfg_augmentation["warm_start"])
warm_stop           = float(cfg_augmentation["warm_stop"]) 


cfg_loader          = load_config(CONFIG_DEFAULT).get("loader")
val_split           = cfg_loader["val_split"]




