import os
from pathlib import Path
from typing import Literal

import numpy as np 
import numpy.typing as npt
import pandas as pd
import torch

from PIL import Image
from torchvision import transforms
from torchvision.transforms.v2 import Transform
from typing import Callable, Optional, Union
from src.data.data_utils import lookup_gender_weights, compute_gender_weights


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, df:pd.DataFrame, image_dir:str, training:bool=True, transform:Optional[Callable]=True)->None:
         self.training_or_validation = training
         self.image_dir = image_dir
         self.df = df
         self.transform = transform if transform else transforms.ToTensor()
         if self.training_or_validation and "FaceOcclusion" in df.columns and "gender" in df.columns:
             y_all = torch.tensor(df["FaceOcclusion"].values, dtype=torch.float32)
             g_all = torch.tensor(df["gender"].values, dtype=torch.float32)
             self.W_F, self.W_M = compute_gender_weights(y_all, g_all)
         else:
             self.W_F = self.W_M = None
         
    def __len__(self)->int:
        
        return len(self.df)

    def __getitem__(self, idx:int)->tuple[torch.Tensor,np.float32,str,str] | tuple[torch.Tensor,str]:
        
        # Select sample
        real_idx = idx % len(self.df) # forcer l'index à revenir à 0 dès que taille max atteinte (N_SAMPLES)
        row = self.df.iloc[real_idx]
        filename = row['filename']

        # Load data and get label
        img_path = self.image_dir / filename
        img = Image.open(img_path).convert('RGB')

        X = self.transform(img)

        if self.training_or_validation:
            y = row['FaceOcclusion']
            y = np.float32(y)
            gender = row['gender']
            iw = np.float32(row['iw']) if 'iw' in self.df.columns else np.float32(1.0) #poids de pondération
            pi = 1/30 + y # poids du score
            gw = lookup_gender_weights(y, gender, self.W_F, self.W_M)
            return X, y, gender, filename, iw, pi, gw
        else:
            y = None
            gender = None
            return X, filename
        
class ChallengeTrain(torch.utils.data.Dataset):
    def __init__(self, raw_challenge_dataset: Dataset):
        self.dataset = raw_challenge_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        X, y, gender, filename, iw, pi, gw = self.dataset[idx]
        target = torch.tensor([y], dtype=torch.float32) # Shape [1] pour BCE

        return X, target, gender, filename, iw, pi, gw


class CelebA(torch.utils.data.Dataset):
    def __init__(
        self,
        split: Literal["train", "test", "valid"],
        transform: Transform | None = None,
        path: str = "./data/celeba",
    ):
        partition = np.loadtxt(os.path.join(path, "Eval", "list_eval_partition.txt"), dtype=str)
        identity = np.loadtxt(os.path.join(path, "Anno", "identity_CelebA.txt"), dtype=str)
        attributes = np.loadtxt(os.path.join(path, "Anno", "list_attr_celeba.txt"), dtype=str)

        match split:
            case "train":
                flag = 0
            case "test":
                flag = 1
            case "valid":
                flag = 2
            case _:
                raise ValueError(f"Unknown split {split}")

        mask = partition[:, 1].astype(int) == flag

        self.transform = transform
        self.root = Path(os.path.join(path, "Img"))
        self.paths: npt.NDArray[np.str_] = partition[mask, 0]
        self.identities: npt.NDArray[np.int_] = identity[mask, 1].astype(int)
        self.attr_names: npt.NDArray[np.str_] = attributes[0, 1:]
        
        # Identification de l'index de la colonne 'Male' pour extraire la bonne cible binaire
        male_col_idx = np.where(self.attr_names == "Male")[0][0]
        self.attributes: npt.NDArray[np.bool_] = attributes[1:, 1:][mask, male_col_idx] == "1"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.root / self.paths[index]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # Extraction binaire au format float32 attendu par le modèle (0.0 ou 1.0)
        label = 1.0 if self.attributes[index] else 0.0
        return img, torch.tensor([label], dtype=torch.float32)

