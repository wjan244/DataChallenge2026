import inspect
import torch
import torch.nn as nn

import torch 
import torch.nn as nn

from src.config import*
from src.config_utils import load_config

cfg_glob = load_config(CONFIG_DEFAULT).get("globaux", {})
EPS= float(cfg_glob.get("EPS", 1e-8))
    
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, iw, pi):
        try:
            eps = 1e-8
            combined_weights = iw * pi   # w_i * p(y_i)
            numerator = torch.sum(combined_weights * (y_true - y_pred) ** 2)    # numérateur
            denominator = torch.sum(combined_weights)   #dénominateur
        except ValueError:
            print("coefficients mal définis")
        
        return numerator / (denominator + eps)

class WeightedLiteMSELoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, iw):
        try:
            return (iw * (y_pred - y_true) ** 2).mean()
        except ValueError as e:
            print("coefficient de reweighting indéfinis", e)
            return None


# class UniversalLossWrapper(nn.Module):
#     def __init__(self, base_loss):
#         super().__init__()
#         self.base_loss = base_loss

#     def forward(self, y_pred, y_true, iw=None, w_pdf=None, gw=None, gender=None):
#         if isinstance(self.base_loss, WeightedMSELoss):
#             return self.base_loss(y_pred, y_true, iw, w_pdf)
#         if isinstance(self.base_loss, WeightedLiteMSELoss):
#             return self.base_loss(y_pred, y_true, iw)
#         if PWGLoss is not None and isinstance(self.base_loss, PWGLoss):
#             return self.base_loss(y_pred, y_true, iw, w_pdf, gw, gender)

#         # default: try calling with (y_pred, y_true)
#         # Some built-in losses use .view() internally which fails for
#         # non-contiguous tensors; make tensors contiguous and use
#         # reshape(-1) which is safe for arbitrary strides.
#         try:
#             if torch.is_tensor(y_pred):
#                 y_pred_t = y_pred.contiguous().reshape(-1)
#             else:
#                 y_pred_t = y_pred
#             if torch.is_tensor(y_true):
#                 y_true_t = y_true.contiguous().reshape(-1)
#             else:
#                 y_true_t = y_true
#             return self.base_loss(y_pred_t, y_true_t)
#         except Exception:
#             return self.base_loss(y_pred, y_true)

class UniversalLossWrapper(nn.Module):
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, y_pred, y_true, iw=None, w_pdf=None, gw=None, gender=None):
        if isinstance(self.base_loss, WeightedMSELoss):
            return self.base_loss(y_pred, y_true, iw, w_pdf)
        if isinstance(self.base_loss, WeightedLiteMSELoss):
            return self.base_loss(y_pred, y_true, iw)
        if PWGLoss is not None and isinstance(self.base_loss, PWGLoss):
            return self.base_loss(y_pred, y_true, iw, w_pdf, gw, gender)

        # Remplacement du bloc par défaut pour gérer le reshape et la contiguïté en 2D [Batch, 1]
        try:
            if torch.is_tensor(y_pred):
                # .reshape(-1, 1) est plus robuste que .view() et compatible avec le stride
                y_pred_t = y_pred.contiguous().reshape(-1, 1)
            else:
                y_pred_t = y_pred
                
            if torch.is_tensor(y_true):
                y_true_t = y_true.contiguous().reshape(-1, 1)
            else:
                y_true_t = y_true

            # Sécurité : Si les tailles ne correspondent pas (ex: résidu d'un dictionnaire mal extrait), 
            # on aligne y_pred_t sur la taille du batch de y_true_t
            if torch.is_tensor(y_pred_t) and torch.is_tensor(y_true_t):
                if y_pred_t.shape[0] != y_true_t.shape[0]:
                    y_pred_t = y_pred_t[:y_true_t.shape[0]]

            return self.base_loss(y_pred_t, y_true_t)
        except Exception:
            return self.base_loss(y_pred, y_true)
        
            
class PWGLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, iw, pi, gw, gender):
        # gender not used but added for eeasier calling of the function without if
        combined_weights = iw * pi * gw
        return torch.sum(combined_weights * (y_true - y_pred) ** 2) / (torch.sum(combined_weights)+EPS)
    
def build_loss_fn(cfg_method):
    # Accept either a config dict (with 'loss_name') or a direct loss name string
    if isinstance(cfg_method, str):
        loss_name = cfg_method
        cfg = {}
    elif isinstance(cfg_method, dict):
        cfg = cfg_method
        loss_name = cfg.get("loss_name")


    loss_cls = LOSS_MAPPING[loss_name]
    supported = inspect.signature(loss_cls.__init__).parameters
    loss_kwargs = {"alpha": cfg.get("loss_alpha")} if "loss_alpha" in cfg and "alpha" in supported else {}

    loss_kwargs = {k: v for k, v in loss_kwargs.items() if v is not None}
    return UniversalLossWrapper(loss_cls(**loss_kwargs))

# Loss mapping
LOSS_MAPPING = {
    "MSE": nn.MSELoss,
    "BCE": nn.BCELoss,
    "SL1":nn.SmoothL1Loss,
    "nMSE": WeightedMSELoss,
    "nLiteMSE": WeightedLiteMSELoss,
    "PWGLoss": PWGLoss}