from src.config import *
from src.config_utils import load_config

cfg 				= load_config(CONFIG_DEFAULT)
cfg_glob 			= cfg.get("globaux", {})
cfg_augmentation 	= cfg.get("augmentation",{})

BATCH_SIZE 			= cfg_glob["BATCH_SIZE"]
kwargs_augmentation	= cfg_augmentation.copy()


