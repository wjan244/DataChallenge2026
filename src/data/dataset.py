import numpy as np 
import pandas as pd
import torch

from PIL import Image
from torchvision import transforms
from typing import Callable, Optional

from src.data.transforms import SchedulerTransform


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, df:pd.DataFrame, image_dir:str, training:bool=True,
                  transform:Optional[Callable]=True, augmentation_factor:int=1,
                  )->None:

         self.training_or_validation = training
         self.image_dir = image_dir
         self.df = df
         self.transform = transform if transform else transforms.ToTensor()
         self.augmentation_factor = augmentation_factor

        
    def __len__(self)->int:
        return int(len(self.df)*self.augmentation_factor)

    def __getitem__(self, idx:int)->tuple[torch.Tensor,np.float32,str,str] | tuple[torch.Tensor,str]:
        
        # Select sample
        real_idx = idx % len(self.df)
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
            iw = np.float32(row['iw']) if 'iw' in self.df.columns else None #poids de pondération
            pi = 1/30 + y # poids du score
            return X, y, gender, filename, iw, pi
        else:
            y = None
            gender = None
            return X, filename
        
class ChallengeTrain(torch.utils.data.Dataset):
    def __init__(self, raw_challenge_dataset: Dataset, scheduler_transform:Optional[SchedulerTransform]=None):
        self.dataset = raw_challenge_dataset
        self.scheduler_transform = scheduler_transform

    def step(self,epoch):
        if self.scheduler_transform is not None:
            self.scheduler_transform.step(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        X, y, gender, filename, iw, pi = self.dataset[idx]
        return X, y, gender, filename, iw, pi
