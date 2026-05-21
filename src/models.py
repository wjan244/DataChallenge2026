import timm
import torch

from torchinfo import summary


def get_model (model_name,num_classes=1):
    """instancier n'importe quel modèle"""

    model = timm.create_model( # timm permet de donner accés à quasiment tous les modèle. Il suffit juste de spécifier le bon nom.
        model_name,
        pretrained=True,
        num_classes=num_classes
    )

    return model

if __name__ == "__main__":
    from src.config import MODEL_NAME

    model = get_model(MODEL_NAME,num_classes=1)
    summary(model)

