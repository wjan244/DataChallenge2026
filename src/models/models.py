from pathlib import Path
import timm

import torch
from torch import nn
from torchinfo import summary

from src.models.finetuning import inject_linear_mlp_probing,inject_lora_transformer

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

def _setup_domain_adaptation(model: nn.Module, rank: int = 8, alpha: int = 16, dropout: float = 0.0, **kwargs) -> nn.Module:
    # injecter les poids LoRA si pas déjà présent (une seule injection)
    if not any("lora_" in name for name, _ in model.named_parameters()):
        model = inject_lora_transformer(model, rank=rank, alpha=alpha, dropout=dropout)
    # geler/ libérer les poids souhaités
    for name, param in model.named_parameters():
        if "lora_" in name or "head" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def _setup_probing(model: nn.Module, rank: int = 8, alpha: int = 16, 
                   dropout: float = 0.0, probing_type:str=None,hidden_size:str=None, **kwargs) -> nn.Module:
    # injecter les poids LoRA si pas déja prsents
    if not any("lora_" in name for name, _ in model.named_parameters()):
        model = inject_lora_transformer(model, rank=rank, alpha=alpha, dropout=dropout)
    model = inject_linear_mlp_probing(model,probing_type,hidden_size)

    # geler/ libérer les poids souhaités
    for name, param in model.named_parameters():
        if "head" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def _setup_lora_finetuning(model: nn.Module, rank: int = 8, alpha: int = 16, dropout: float = 0.0, **kwargs) -> nn.Module:
    # injecter les poids LoRA si pas déja prsents (une seule injection)
    if not any("lora_" in name for name, _ in model.named_parameters()):
        model = inject_lora_transformer(model, rank=rank, alpha=alpha, dropout=dropout)
    # geler/ libérer les poids souhaités
    for name, param in model.named_parameters():
        if "lora_" in name or "head" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def get_model(model_name: str, num_classes=1, method: str | None = None, weights: str | Path | None = None, **method_kwargs) -> nn.Module:
    """instancier le modèle défini dans config.py et lui affecter une méthode de FineTuning:
    - Linear_probing
    - LoRA
    """
    METHOD_MAPPING = {"domain_adaptation":_setup_domain_adaptation,"probing_training":_setup_probing,"lora_training":_setup_lora_finetuning} 
    # récupérer le modèle
    model = timm.create_model( # timm permet de donner accés à quasiment tous les modèle. Il suffit juste de spécifier le bon nom.
        model_name,
        pretrained=True,
        num_classes=num_classes)
    
    # ajout de la sigmoïde au modèle chargé
    model = OcclusionModel(model)
    # si aucune méthode spécifiée, retourner le modèle tel quel
    if method is None:
        return model
    # accepter quelques alias courts éventuels puis valider
    SHORT_METHODS = {"probing": "probing_training", "lora": "lora_training", "domain": "domain_adaptation"}
    if method in SHORT_METHODS:
        method = SHORT_METHODS[method]
    if method not in METHOD_MAPPING:
        raise ValueError(f"Unknown method {method!r}. Supported: {list(METHOD_MAPPING.keys())}")
    # injecter la méthode
    model = METHOD_MAPPING[method](model,**method_kwargs)
    return model


if __name__ == "__main__":
    from src.config_utils import load_config
    
    cfg = load_config("beit3_base_patch16_224.yaml")
    
    # On teste en passant le dictionnaire d'hyperparamètres du YAML via **
    model = get_model(
        model_name=cfg["model"], 
        num_classes=1, 
        method="lora_training",
        **cfg["lora_training"].get("method_kwargs")
    )
    
    summary(model)
