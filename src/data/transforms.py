from torchvision.transforms import v2

from src.config import*
from . import warm_start,warm_stop,augmentation_status

def get_augmentation_finetuning_transforms()->None:
    """définit le transform de data_augmentation avec les transformations
    de base appelées de manière composée et/ou aléatoire:
    - RandomRotation
    - HorizontalFlip
    - ColorJitter (constraste et luminosité)
    - FlouGaussien
    """
    # option 1:
    return v2.Compose([v2.RandomHorizontalFlip(p=0.5),
                       v2.RandomApply([
                           v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)], p=0.8),
                             v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0))], p=0.3)])
    # option 2:  
    # return v2.RandomChoice([v2.Identity(),
    #                        v2.RandomChoice([v2.RandomRotation(degrees=25),
    #                                         v2.RandomHorizontalFlip(p=0.5),
    #                                         v2.ColorJitter(brightness=0.2),
    #                                         v2.ColorJitter(contrast=0.3),
    #                                         v2.GaussianBlur(kernel_size=(3, 5), sigma=(0.5, 2.0))])],
    #                         p=[0.2,0.8])    # 20% de identité et 80% de random_choice

class SchedulerTransform(torch.nn.Module):
    def __init__(self,warm_stop:float,warm_start:int=1):
        super().__init__()
        self.epoch = 0
        self.warm_stop = warm_stop
        self.warm_start = warm_start
        self.augment = get_augmentation_finetuning_transforms()
        self.identity = v2.Identity()
        self.augmentation_status = augmentation_status

    def step(self, epoch: int):
        self.epoch = epoch

    def forward(self,img):
        if self.augmentation_status==True and warm_start<=self.epoch<self.warm_stop:
            return self.augment(img)
        else:
            return self.identity(img)


