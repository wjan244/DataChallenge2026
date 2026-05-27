# méthode d'adaptation de la distribution
AUGMENTATION = False

# Hyper-paramètres entrainement
MODEL_NAME = 'beit3_base_patch16_224'

                  # exemples:
                  # 'beit3_base_patch16_224'
                  # 'mobilenetv3_small_075'
                  # 'vit_small_patch14_reg4_dinov2.lvd142m'
                  # hyper paramètres d'entrainement
PATIENCE = 5

# hyper-paramètres Dataloader
BATCH_SIZE = 32

# Hyper-paramètres LoRA
RANK = 16
ALPHA = 16
DROPOUT = 0.2

from src.data_loader import (get_challenge_train_loader,get_celeba_train_loader, 
                             get_celeba_val_loader, get_challenge_val_loader)

# configuration des méthodes de Fine_Tuning
CONFIG_DOMAINE = {
    "loss_name": "BCE",
    "method_FT": "domain_adaptation",
    "loader_factory": get_celeba_train_loader,
    "val_loader_factory": get_celeba_val_loader,
    "learning_rate": 5e-5,
    "num_epoch": 5 #5
}

CONFIG_LINEAR_PROBING = {
    "loss_name": "nMSE",
    "method_FT": "linear_probing",
    "loader_factory": get_challenge_train_loader,
    "val_loader_factory": lambda b, n: get_challenge_val_loader(split="val_samp", batch_size=b, num_workers=n),
    "learning_rate": 1e-3,
    "num_epoch": 8#15
}

CONFIG_LORA_FT = {
    "loss_name": "nMSE",
    "method_FT": "LoRA_Transformer",
    "loader_factory": get_challenge_train_loader,
    "val_loader_factory": lambda b, n: get_challenge_val_loader(split="val_samp", batch_size=b, num_workers=n),
    "learning_rate": 2e-4,
    "num_epoch": 15#15
}



