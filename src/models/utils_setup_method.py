import torch
from torch import nn

from src.models.utils_probing import inject_linear_mlp_probing
from src.models.utils_gradientreversal import inject_adversarial_probing

def setup_probing(model: nn.Module, rank: int = 8, alpha: int = 16, 
                   dropout: float = 0.0, probing_type:str=None,hidden_size:str=None, **kwargs) -> nn.Module:
    model = inject_linear_mlp_probing(model,probing_type,hidden_size)

    # geler/ libérer les poids souhaités
    for name, param in model.named_parameters():
        if "head" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def setup_adversarial_probing(model:torch.nn.Module, **method_kwargs) -> nn.Module:
    model = inject_adversarial_probing(model, **method_kwargs)

    # geler/ libérer les poids souhaités
    for name, param in model.named_parameters():
        if "head" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def setup_domain_adaptation(model: nn.Module, rank: int = 8, alpha: int = 16, dropout: float = 0.0, 
                             probing_type:str=None,hidden_size:str=None, **kwargs) -> nn.Module:
    # injecter une nouvelle tête de classification
    model = inject_linear_mlp_probing(model,probing_type,hidden_size)
    # geler/ libérer les poids souhaités
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def setup_lora_finetuning(model: nn.Module, rank: int = 8, alpha: int = 16, dropout: float = 0.0, **kwargs) -> nn.Module:
    # geler/ libérer les poids souhaités
    for name, param in model.named_parameters():
        if "lora_" in name or "head" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model