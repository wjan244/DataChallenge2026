import os

# méthode d'adaptation de la distribution
CURRENT_METHOD = "dkl" 
AUGMENTATION = False

# Hyper-paramètres entrainement
MODEL_NAME = 'mobilenetv3_small_075'

                  # exemples:
                  # 'beit3_base_patch16_224'
                  # 'mobilenetv3_small_075'
                  # 'vit_small_patch14_reg4_dinov2.lvd142m'
                  # hyper paramètres d'entrainement
PATIENCE = 5

# hyper-paramètres Dataloader
BATCH_SIZE = 32
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
NUM_CLASSES = 1

# Hyper-paramètres LoRA
RANK = 8
ALPHA = 16
DROPOUT = 0.05

from src.data_loader import (get_challenge_train_loader,get_celeba_train_loader, 
                             get_celeba_val_loader, get_challenge_val_loader)

# configuration des méthodes de Fine_Tuning
CONFIG_DOMAINE = {
    "loss_name": "BCE",
    "method_FT": "domain_adaptation",
    "loader_factory": get_celeba_train_loader,
    "val_loader_factory": get_celeba_val_loader,
    "learning_rate": 5e-5,
    "num_epoch": 1 #5
}

CONFIG_LINEAR_PROBING = {
    "loss_name": "MSE",
    "method_FT": "linear_probing",
    "loader_factory": get_challenge_train_loader,
    "val_loader_factory": lambda b, n: get_challenge_val_loader(split="val_samp", batch_size=b, num_workers=n),
    "learning_rate": 1e-3,
    "num_epoch": 1#15
}

CONFIG_LORA_FT = {
    "loss_name": "MSE",
    "method_FT": "LoRA_Transformer",
    "loader_factory": get_challenge_train_loader,
    "val_loader_factory": lambda b, n: get_challenge_val_loader(split="val_samp", batch_size=b, num_workers=n),
    "learning_rate": 2e-4,
    "num_epoch": 1#15
}



