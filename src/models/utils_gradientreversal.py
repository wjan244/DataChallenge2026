from torch.autograd import Function

import torch
from torch import nn


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None


revgrad = GradientReversal.apply


class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)
    def forward(self, x):
        return revgrad(x, self.alpha)

  
def inject_adversarial_probing(model:torch.nn.Module, **method_kwargs)->torch.nn.Module:
    num_head = method_kwargs["num_head"]
    alpha_grl = method_kwargs["lambda"]
    hidden_size = method_kwargs["hidden_size"]
    num_head = method_kwargs["num_head"]

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

    # couche GRL
    container.grl = GradientReversal(alpha=alpha_grl)

    # gestion des têtes
    multiple_new_head = torch.nn.ModuleDict()
        # mettre une couche linéaire par défaut
    for i in range(num_head):
        multiple_new_head[f"{head_attr}_{i}"] = torch.nn.Sequential(torch.nn.Linear(in_features, hidden_size),
                                                                    torch.nn.ReLU(),
                                                                    torch.nn.Linear(hidden_size, out_features))
    setattr(container, head_attr, multiple_new_head)

    old_forward = container.forward                         # prédiction de l'ancienne tête

    # aiguiller les entrées vers toutes les têtes
    def _switch_forward(*args, **kwargs):
        """permet d'aiguiller les entrées vers les différentes têtes"""
        setattr(container,head_attr,torch.nn.Identity())    # nouvelle tête neutre
        container.head = torch.nn.Identity()    
        features = old_forward(*args, **kwargs)             # capter les features de l'ancienne tête
        setattr(container,head_attr,multiple_new_head)      # instancier la nouvelle tête avec toutes les têtes créées
        
        output = {}                                          # attribuer les features à chaque tête
        for i in range(num_head):
            cle = f"{head_attr}_{i}"                        # nom de la tête
            head_layer = multiple_new_head[cle]             # tête
            if i==1:                                        # appliquer le GRL à la première tête
                features_head = container.grl(features)
            else:
                features_head = features
        
            output[cle] = head_layer(features_head)
        return output

    container.forward = _switch_forward                     # récupérer l'output des nouvelles têtes

    return model
