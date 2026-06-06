import os
import pandas as pd
import timm

from torch.utils.data import DataLoader
from torchvision.transforms import v2

from . import augmentation_factor,val_split
from src.data.transforms import SchedulerTransform, warm_stop
from src.config import*
from src.config_utils import load_config
from src.data.dataset import Dataset, ChallengeTrain
from src.data.data_utils import get_challenge_split
    
    
def get_challenge_train_loader(batch_size: int, num_workers: int = NUM_WORKERS, model_name: str = None, **kwargs_augmentation) -> DataLoader:
    """Génère le DataLoader d'entraînement"""
    df_train, _, _, _ = get_challenge_split()
    data_config = timm.data.resolve_model_data_config(timm.create_model(model_name, pretrained=True))
    data_transform = timm.data.create_transform(**data_config, is_training=True)

    augmentation_transform = None
    augmentation_factor = 1
    transform_pipeline = data_transform

    # gestion de l'augmentation
    augmentation_status = kwargs_augmentation.get("status")
    if augmentation_status is True:
        warm_stop = kwargs_augmentation.get("warm_stop", 12) 
        augmentation_factor = kwargs_augmentation.get("augmentation_factor", 1)
        augmentation_transform = SchedulerTransform(warm_stop=warm_stop)
        transform_pipeline = v2.Compose([data_transform, augmentation_transform])
        print(f"augmentation_{augmentation_status}_avec un facteur_{augmentation_factor}")
    else:
        print("pas d'augmentation activée")
    
    # transformation des données
    raw_dataset = Dataset(df=df_train, image_dir=IMG_DIR, training=True, transform=transform_pipeline,
                          augmentation_factor=augmentation_factor)
    standard_dataset = ChallengeTrain(raw_dataset,augmentation_transform)
    
    return DataLoader(standard_dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers)
        

def get_challenge_val_loader(split: str, batch_size: int, num_workers: int = NUM_WORKERS, model_name: str = None) -> DataLoader:
    
    _, df_val_raw, df_val_samp, _ = get_challenge_split()
    df_val = df_val_samp if split == val_split else df_val_raw
    
    data_config = timm.data.resolve_model_data_config(timm.create_model(model_name, pretrained=True))
    val_transform = timm.data.create_transform(**data_config, is_training=False)

    val_set = Dataset(df_val, IMG_DIR, training=True, transform=val_transform)
    return DataLoader(val_set, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)


def get_challenge_test_loader(df_test: pd.DataFrame, batch_size: int, num_workers: int = NUM_WORKERS, model_name: str = None) -> DataLoader:

    data_config = timm.data.resolve_model_data_config(timm.create_model(model_name, pretrained=True))
    test_transform = timm.data.create_transform(**data_config, is_training=False)

    test_set = Dataset(df_test, IMG_DIR, training=False, transform=test_transform)
    
    return DataLoader(test_set, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)


if __name__ == "__main__":
# vérification des dimensions
    cfg_glob = load_config(CONFIG_DEFAULT).get("globaux")
    model_name = load_config(CONFIG_DEFAULT).get("model")
    batch_size = cfg_glob.get("BATCH_SIZE")
    
    train_dataloader = get_challenge_train_loader(batch_size, NUM_WORKERS, model_name)
    val_dataloader = get_challenge_val_loader('val_samp', batch_size, NUM_WORKERS, model_name)
    print(f"dim dataloader: {len(train_dataloader)}")
    print("\n--------------")
    print(f"dim dataloader: {len(val_dataloader)}")