import os
import pandas as pd
import timm

from torch.utils.data import DataLoader
from torchvision.transforms import v2

from src.config import*

_PIN = DEVICE.type == "cuda"
_PW  = NUM_WORKERS > 0
from src.data.dataset import Dataset, ChallengeTrain, CelebA
from src.data.data_utils import get_challenge_split
from src.data.transforms import get_augmentation_finetuning_transforms,get_augmentation_pretrained_transforms


def get_challenge_train_loader(batch_size: int, num_workers: int = NUM_WORKERS, model_name: str = None, augmentation: bool = None) -> DataLoader:
    """Génère le DataLoader d'entraînement pour le challenge (Format: image, target)."""
    df_train, _, _, _ = get_challenge_split()
    data_config = timm.data.resolve_model_data_config(timm.create_model(model_name, pretrained=True))

    # transforms
    data_transform = timm.data.create_transform(**data_config, is_training=True)
    augmentation_transform = get_augmentation_finetuning_transforms() if augmentation else None
    transforms = [data_transform, augmentation_transform] if augmentation_transform else [data_transform]
    transform_pipeline = v2.Compose(transforms)

    raw_dataset = Dataset(df=df_train, image_dir=IMG_DIR, training=True, transform=transform_pipeline)
    standard_dataset = ChallengeTrain(raw_dataset)
    
    return DataLoader(standard_dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=_PIN, persistent_workers=_PW)
        

def get_celeba_train_loader(batch_size: int, num_workers: int = NUM_WORKERS, model_name: str = None, augmentation: bool = None) -> DataLoader:
    """Génère le DataLoader CelebA d'entraînement en utilisant la classe locale CelebA."""
    data_config = timm.data.resolve_model_data_config(timm.create_model(model_name, pretrained=True))
    # transform (data + augmentation -> pipe)
    data_transform = timm.data.create_transform(**data_config, is_training=True)
    augmentation_transform = get_augmentation_pretrained_transforms() if augmentation==True else None 
    transform_pipeline = v2.Compose([data_transform,augmentation_transform])

    celeba_dataset = CelebA(split="train", transform=transform_pipeline,path="./data/celeba")
    
    return DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=_PIN, persistent_workers=_PW)


def get_challenge_val_loader(split: str, batch_size: int, num_workers: int = NUM_WORKERS, model_name: str = None) -> DataLoader:
    
    _, df_val_raw, df_val_samp, _ = get_challenge_split()
    df_val = df_val_samp if split == "val_samp" else df_val_raw
    
    data_config = timm.data.resolve_model_data_config(timm.create_model(model_name, pretrained=True))
    val_transform = timm.data.create_transform(**data_config, is_training=False)

    val_set = Dataset(df_val, IMG_DIR, training=True, transform=val_transform)
    return DataLoader(val_set, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=_PIN, persistent_workers=_PW)

def get_celeba_val_loader(batch_size: int, num_workers: int = NUM_WORKERS, model_name: str = None) -> DataLoader:
    """Génère le DataLoader de validation CelebA en utilisant la classe locale CelebA."""
    data_config = timm.data.resolve_model_data_config(timm.create_model(model_name, pretrained=True))
    val_transform = timm.data.create_transform(**data_config, is_training=False)

    celeba_dataset = CelebA(
        split="valid", 
        transform=val_transform,
        path="./data/celeba"
    )
    return DataLoader(celeba_dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=_PIN, persistent_workers=_PW)


def get_challenge_test_loader(df_test: pd.DataFrame, batch_size: int, num_workers: int = NUM_WORKERS, model_name: str = None) -> DataLoader:

    data_config = timm.data.resolve_model_data_config(timm.create_model(model_name, pretrained=True))
    test_transform = timm.data.create_transform(**data_config, is_training=False)

    test_set = Dataset(df_test, IMG_DIR, training=False, transform=test_transform)
    
    return DataLoader(test_set, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=_PIN, persistent_workers=_PW)