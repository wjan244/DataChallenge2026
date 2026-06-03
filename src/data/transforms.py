from torchvision.transforms import v2

def get_augmentation_pretrained_transforms()->None:
    """définit le transform de data_augmentation pour l'étape  de FT sur les données
    """
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.RandomRotation(degrees=15)], p=0.3),
        v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)], p=0.5),
        v2.RandomGrayscale(p=0.05),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
        v2.RandomErasing(p=0.3),
    ])

def get_augmentation_finetuning_transforms()->None:
    """définit le transform de data_augmentation avec les transformations
    de base appelées de manière composée et/ou aléatoire:
    - RandomRotation
    - HorizontalFlip
    - ColorJitter (constraste et luminosité)
    - FlouGaussien
    """
    return v2.Compose(
        [
        v2.RandomRotation(degrees=10),
        v2.RandomHorizontalFlip(p=0.5),

        v2.RandomChoice(
            [
        v2.ColorJitter(brightness=0.2),
        v2.ColorJitter(contrast=0.3),
        v2.GaussianBlur(kernel_size=(3, 5), sigma=(0.5, 2.0)),
        v2.Identity()
            ])
        ])