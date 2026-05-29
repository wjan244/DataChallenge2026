import torch

from src.models.lora import LoRALinear

def inject_linear_mlp_probing(model:torch.nn.Module,probing_type:str,hidden_size:int=None)->torch.nn.Module:
    # trouver le bon container qui expose la tête (head/classifier)
    container = model
    if hasattr(model, 'model') and (hasattr(model.model, 'head') or hasattr(model.model, 'classifier')):
        container = model.model

    # déterminer le nom de l'attribut de sortie
    if hasattr(container, 'head'):
        head_attr = 'head'
    elif hasattr(container, 'classifier'):
        head_attr = 'classifier'
    
    head = getattr(container, head_attr)
    in_features = getattr(head, 'in_features', None)
    out_features = getattr(head, 'out_features', None)
    
    # cast hidden_size if provided as string from YAML
    if hidden_size is not None:
        hidden_size = int(hidden_size)

    if probing_type == "linear_probing":
        new_head = torch.nn.Linear(in_features, out_features)
    elif probing_type == "mlp_probing":
        new_head = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, out_features)
        )
    else:
        raise ValueError("erreur dans le choix des paramètres de probing")

    setattr(container, head_attr, new_head)
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

            # Si la couche est déjà une LoRALinear, l'ignorer
            if isinstance(old_layer, LoRALinear):
                continue

            # Assurer que la couche ressemble à un Linear (has in/out features)
            if not (hasattr(old_layer, 'in_features') and hasattr(old_layer, 'out_features')):
                # on ne sait pas comment injeter LoRA ici -> skip
                continue

            has_bias = getattr(old_layer, 'bias', None) is not None

            new_layer = LoRALinear(
                in_dim=getattr(old_layer, 'in_features'),
                out_dim=getattr(old_layer, 'out_features'),
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                bias=has_bias
            )
            # transfert des poids du Linear Probing vers la partie fixe de LoRA
            if hasattr(old_layer, 'weight') and new_layer.linear.weight.shape == old_layer.weight.shape:
                new_layer.linear.weight.data = old_layer.weight.data.clone()
            
            # gestion des biais absents
            if has_bias and hasattr(old_layer, 'bias') and hasattr(new_layer.linear, 'bias'):
                new_layer.linear.bias.data = old_layer.bias.data.clone()
                
            setattr(parent, layer_name, new_layer)
    
    return model