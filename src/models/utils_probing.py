import torch

from torch import nn


def inject_linear_mlp_probing(model:torch.nn.Module, probing_type:str, hidden_size:int=None, device=None)->torch.nn.Module:
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
    # mettre une couche linéaire par défaut
    if probing_type is None or probing_type == "linear_probing":
        # prendre en compte la gestion du cas ou probing_training est False
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
