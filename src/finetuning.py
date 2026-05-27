import torch

from src.lora import LoRALinear

def inject_linear_probing(model:torch.nn.Module)->torch.nn.Module:
    """exécute un FineTuning de type LinearProbing:
    - gèle du backbone
    - dégèle de la tête de classification"""

    for params in model.parameters():
        params.requires_grad = False
    
    for params in model.head.parameters():
        params.requires_grad=True

    return model

def inject_lora_transformer(model:torch.nn.modules, rank:int, alpha:int, dropout:torch.nn.modules)->torch.nn.Module:
    """
    - capter les q, k, v dans chacune des couches de SSAST
    - remplacer les poids dans chaque couche par les nouveaux poids (w_backbone + w_lora)"""
    for params in model.parameters():
        params.requires_grad = False

    # extraction des Q,K,V de chacune des 12 couches
    for name, module in model.named_modules(): 
        if any(target in name for target in ["qkv","query", "key", "value"]):
            parent_name = ".".join(name.split(".")[:-1]) # nom du chemin vers l'attribut
            layer_name = name.split(".")[-1]    # layer_name: nom de l'attribut (exp: "query")
            parent = dict(model.named_modules())[parent_name] # objet réel pointé par le chemin. parent = tout le bloc d'attention par couche
            
            # Création de la couche LoRA avec les poids originaux
            old_layer = getattr(parent, layer_name) # aller chercher l'attribut (objet) dans le chemin layer_name

            has_bias = old_layer.bias is not None

            new_layer = LoRALinear(
                in_dim=old_layer.in_features,
                out_dim=old_layer.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                bias=has_bias
            )
            
            # transfert des poids du Linear Probing vers la partie fixe de LoRA
            new_layer.linear.weight.data = old_layer.weight.data.clone()
            # gestion des biais absents
            if has_bias:
                new_layer.linear.bias.data = old_layer.bias.data.clone()
        
            setattr(parent, layer_name, new_layer)

    
    return model