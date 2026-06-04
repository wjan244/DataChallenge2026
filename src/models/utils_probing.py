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

def inject_multi_probing(model:torch.nn.Module, probing_type:str, hidden_size:int=None,num_head:int=None)->torch.nn.Module:
    # trouver le bon container qui expose la tête (head/classifier)
    container = model
    if hasattr(model, 'model') and (hasattr(model.model, 'head') or hasattr(model.model, 'classifier')):
        container = model.model

    # déterminer le nom de l'attribut de sortie
    if hasattr(container, 'head'):
        head_attr = 'head'
    elif hasattr(container, 'classifier'):
        head_attr = 'classifier'
    
    # récupérer les dimensions de la tête (in/out_dim)
    head = getattr(container, head_attr)
    in_features = getattr(head, 'in_features', None)
    out_features = getattr(head, 'out_features', None)
    
    # cast hidden_size if provided as string from YAML
    if hidden_size is not None:
        hidden_size = int(hidden_size)

    multiple_new_head = torch.nn.ModuleDict()
        # mettre une couche linéaire par défaut
    for i in range(num_head):
        if probing_type is None or probing_type == "linear_probing":
            multiple_new_head [f"{head_attr}_{i}"] = torch.nn.Linear(in_features, out_features)
        # créer des têtes MLP
        elif probing_type == "mlp_probing":
            multiple_new_head[f"{head_attr}_{i}"] = torch.nn.Sequential(torch.nn.Linear(in_features, hidden_size),
                                                                 torch.nn.ReLU(),
                                                                 torch.nn.Linear(hidden_size, out_features))
        else:
            raise ValueError("erreur dans le choix des paramètres de probing")

    setattr(container, head_attr, multiple_new_head)

    old_forward = container.forward         # prédiction de l'ancienne tête

    # aiguiller les entrées vers toutes les têtes
    def _switch_forward(*args, **kwargs):
        """permet d'aiguiller les entrées vers les différentes têtes"""
        setattr(container,head_attr,torch.nn.Identity())    # nouvelle tête neutre
        container.head = torch.nn.Identity()    
        features = old_forward(*args, **kwargs)             # capter les features de l'ancienne tête
        setattr(container,head_attr,multiple_new_head)      # instancier la nouvelle tête avec toutes les têtes créées
        
        output = {}                             # attribuer les features à chaque tête
        for i in range(num_head):
            cle = f"{head_attr}_{i}"        # nom de la tête
            head = multiple_new_head[cle]   # tête
            output[cle] = head(features)
        return output

    container.forward = _switch_forward   # récupérer l'output des nouvelles têtes

    return model
