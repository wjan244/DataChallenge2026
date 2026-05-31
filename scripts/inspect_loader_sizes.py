from pathlib import Path
import timm
import torch
from src.config_utils import load_config
from src.config import CONFIG_DEFAULT, DEVICE, NUM_WORKERS
from src.data.data_loader import get_challenge_train_loader, get_challenge_val_loader, get_challenge_test_loader, get_celeba_train_loader, get_celeba_val_loader
from src.data.data_utils import get_challenge_split


def loader_info(loader):
    ds = loader.dataset
    try:
        dataset_len = len(ds)
    except Exception:
        dataset_len = None
    batch_size = loader.batch_size
    num_batches = len(loader)
    return dataset_len, batch_size, num_batches


def main():
    cfg = load_config('vit_tiny_patch16_224.yaml')
    cfg_glob = cfg['globaux']
    cfg_mod = cfg['model']
    batch_size = cfg_glob['BATCH_SIZE']

    print('CONFIG BATCH_SIZE', batch_size)

    # challenge loaders
    df_train, df_val_raw, df_val_samp, df_test = get_challenge_split()
    chal_train_loader = get_challenge_train_loader(batch_size=batch_size, num_workers=NUM_WORKERS, model_name=cfg_mod, augmentation=True)
    chal_val_loader = get_challenge_val_loader(split='val_samp', batch_size=batch_size, num_workers=NUM_WORKERS, model_name=cfg_mod)
    chal_test_loader = get_challenge_test_loader(df_test, batch_size=batch_size, num_workers=NUM_WORKERS, model_name=cfg_mod)

    print('Challenge train loader:', loader_info(chal_train_loader))
    print('Challenge val loader:', loader_info(chal_val_loader))
    print('Challenge test loader:', loader_info(chal_test_loader))

    # celeba loaders
    celeba_train_loader = get_celeba_train_loader(batch_size=batch_size, num_workers=NUM_WORKERS, model_name=cfg_mod, augmentation=True)
    celeba_val_loader = get_celeba_val_loader(batch_size=batch_size, num_workers=NUM_WORKERS, model_name=cfg_mod)

    print('CelebA train loader:', loader_info(celeba_train_loader))
    print('CelebA val loader:', loader_info(celeba_val_loader))

if __name__ == '__main__':
    main()
