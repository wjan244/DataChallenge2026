import timm

from torchinfo import summary

from src.config import TRAINING_MODE,NUM_CLASSES


def _linear_probing(model):

    for params in model.parameters():
        params.requires_grad = False
    
    for params in model.head.parameters():
        params.requires_grad=True

    return model
    
TRAIN_MODE = {"linear_probing":_linear_probing
              }

def get_model (model_name,num_classes=NUM_CLASSES,method=TRAINING_MODE):
    """instancier n'importe quel modèle prenant en compte plusieurs méthodes d'entrainement:
    - Linear_probing
    - LoRA
    """
    # récupérer le modèle
    model = timm.create_model( # timm permet de donner accés à quasiment tous les modèle. Il suffit juste de spécifier le bon nom.
        model_name,
        pretrained=True,
        num_classes=num_classes
    )
    
    # attribuer la méthode d'entrainement
    if method in TRAIN_MODE:
        train_method = TRAIN_MODE[method]
        model = train_method(model)

    return model

if __name__ == "__main__":
    from src.config import MODEL_NAME

    model = get_model(MODEL_NAME,num_classes=1)
    summary(model)

