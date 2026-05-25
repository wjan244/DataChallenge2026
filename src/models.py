import timm

from torch import nn
from torchinfo import summary

from src.config import TRAINING_MODE,NUM_CLASSES,RANK,DROPOUT,ALPHA
from src.finetuning import inject_linear_probing, inject_lora_transformer

class OcclusionModel(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model
        self.sigmoide = nn.Sigmoid()

    def forward(self,x):
        return self.sigmoide(self.model(x))

    
TRAIN_MODE = {"linear_probing":inject_linear_probing,
              "LoRA_Transformer": lambda model: inject_lora_transformer(model, rank=RANK, alpha=ALPHA, dropout=DROPOUT)
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

    return OcclusionModel(model)

if __name__ == "__main__":
    from src.config import MODEL_NAME

    model = get_model(MODEL_NAME,num_classes=1)
    summary(model)
    # afficher le nom des couches
    for name, _ in list(model.named_parameters())[:]:
        print(name)

