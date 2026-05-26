from pathlib import Path
import timm

import torch
from torch import nn
from torchinfo import summary

from src.config import NUM_CLASSES,RANK,DROPOUT,ALPHA
from src.finetuning import inject_linear_probing, inject_lora_transformer

class OcclusionModel(nn.Module):
    """
    - instancie le modèle défini dans config.py
    - ajoute une sigmoïde en sortie de du modèle
    """
    def __init__(self,model:nn.Module)->None:
        super().__init__()
        self.model = model
        self.sigmoide = nn.Sigmoid()

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.sigmoide(self.model(x))

def setup_domain_adaptation(model: nn.Module) -> nn.Module:
    return inject_lora_transformer(model, rank=RANK, alpha=ALPHA, dropout=DROPOUT)


def setup_linear_probing_with_lora(model: nn.Module) -> nn.Module:
    model = inject_lora_transformer(model, rank=RANK, alpha=ALPHA, dropout=DROPOUT)
    for param in model.parameters():
        param.requires_grad = False
    return model

def setup_lora_finetuning(model: nn.Module) -> nn.Module:
    model = inject_lora_transformer(model, rank=RANK, alpha=ALPHA, dropout=DROPOUT)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


TRAIN_MODE = {
    "domain_adaptation": setup_domain_adaptation, # renvoyer modèle avec injection mat LoRA
    "linear_probing": setup_linear_probing_with_lora, # renvoyer modèle avec injection mat LoRA et gèle LoRA + Backbone
    "LoRA_Transformer": setup_lora_finetuning} # Fine tuning par LoRA

# TRAIN_MODE = {"linear_probing":inject_linear_probing,
#               "LoRA_Transformer": lambda model: inject_lora_transformer(model, rank=RANK, alpha=ALPHA, dropout=DROPOUT), # renvoyer modèle avec injection mat LoRA
#               "domain_adaptation":lambda model:model} # renvoyer le modèle sans modif

def get_model (model_name:str,num_classes=NUM_CLASSES,method:str|None=None,weights:str|Path|None=None)->nn.Module:
    """instancier le modèle défini dans config.py et lui affecter une méthode de FineTuning:
    - Linear_probing
    - LoRA
    """
    # récupérer le modèle
    model = timm.create_model( # timm permet de donner accés à quasiment tous les modèle. Il suffit juste de spécifier le bon nom.
        model_name,
        pretrained=True,
        num_classes=num_classes
    )
    
    # ajout de la sigmoïde au modèle chargé
    model = OcclusionModel(model)

   # attribuer la méthode d'entrainement
    if method in TRAIN_MODE:
        train_method = TRAIN_MODE[method]
        model.model = train_method(model.model)

    # charger les nouveaux poids du modèle entrainé à l'étape d'avant
    if weights is not None:
        state_dict = torch.load(weights,map_location='cpu')
        model.load_state_dict(state_dict)

    return model

if __name__ == "__main__":
    from src.config import MODEL_NAME

    model = get_model(MODEL_NAME,num_classes=1,method="domain_adaptation")
    summary(model)
    # afficher le nom des couches
    for name, params in list(model.named_parameters())[:]:
        print(name)

