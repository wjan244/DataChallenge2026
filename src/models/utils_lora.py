import math
import logging
import torch


from torch import nn


class LoRALinear(nn.Module):
    """
    Encapsule une couche linéaire gelée et ajoute des matrices
    de mise à jour de rang inférieur (A et B) entraînables pour le finetuning.
    """

    def __init__(self, in_dim: int = None, out_dim: int = None, rank: int = 4, alpha: float = 1.0, dropout: float = 0.0, bias: bool = False, base_linear: nn.Linear = None, device=None, dtype=None) -> None:
        super().__init__()

        # gestion du cas s'il existe une couche linéaire fournie
        if base_linear is not None:
            self.linear = base_linear
            
            in_dim = getattr(self.linear, 'in_features', in_dim)
            out_dim = getattr(self.linear, 'out_features', out_dim)
        else:
            # sinon créer une couche linéaire
            self.linear = nn.Linear(in_dim, out_dim, bias=bias)
            if device is not None or dtype is not None:
                try:
                    self.linear = self.linear.to(device=device, dtype=dtype)
                except Exception:
                    pass
        
        # création des matrices A/B (couches linéaires)
        self.lora_a = nn.Linear(in_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_dim, bias=False)
        
        # initialisation des biais
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5)) # Initialisation standard pour A
        nn.init.zeros_(self.lora_b.weight)                           # Initialisation stricte à 0 pour B afin de garantir la neutralité au départ

        # définir r/alpha/dropout
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout)

        # Contrôle de poids freezés/défreezés par défaut avec LoRA -> adapté en fonction de la méthode dans utils_setup_method
        try:
            self.linear.weight.requires_grad = False
            if getattr(self.linear, 'bias', None) is not None:
                self.linear.bias.requires_grad = False  # geler des couches linéaires s'ils existent
        except Exception:
            pass
        try:
            self.lora_a.weight.requires_grad = True
            self.lora_b.weight.requires_grad = True
        except Exception:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen_out = self.linear(x)
        lora_out = self.lora_b(self.lora_a(self.dropout(x)))
        return frozen_out + (self.alpha / max(1, self.rank)) * lora_out


def inject_lora_transformer(model: torch.nn.modules, rank: int = None, alpha: int = None, dropout: float = None) -> torch.nn.Module:
    """
    - capter les q, k, v
    - remplacer les poids dans chaque couche par les nouveaux poids (w_backbone + w_lora)"""

    # Idempotence: skip if already injected or if any parameter name contains 'lora'
    if getattr(model, '_lora_injected', False):
        return model
    if any('lora' in name for name, _ in model.named_parameters()):
        setattr(model, '_lora_injected', True)
        return model

    replaced = 0
    targets = ("qkv", "query", "key", "value")

    # Parcourir les modules nommés et remplacer simplement les Linear ciblés
    for full_name, module in list(model.named_modules()):
        
        # cibler les nn.Linear purs qui correspondent aux mots-clés
        if not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
            continue
        if not any(t in full_name for t in targets):
            continue

        # récupération dynamique du parent 
        *parent_path, layer_name = full_name.split(".")
        parent = model
        for path_segment in parent_path:
            parent = getattr(parent, path_segment)

        # extraction sécurisée des propriétés de l'ancienne couche
        has_bias = module.bias is not None
        dev = module.weight.device
        dtype = module.weight.dtype
        new_layer = LoRALinear(in_dim=module.in_features, out_dim=module.out_features, 
                               rank=rank,alpha=alpha,dropout=dropout,bias=has_bias,
                               base_linear=module,device=dev,dtype=dtype)

        setattr(parent, layer_name, new_layer)
        replaced += 1

    if replaced > 0:
        setattr(model, '_lora_injected', True)

    return model