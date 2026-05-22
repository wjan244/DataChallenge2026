from torchvision.transforms import v2

def get_augmentation_transforms():
    """défini le transform de data_augmentation avec les transformation
    de base appelée de manière composée et/ou aléatoire:
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