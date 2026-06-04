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
        if isinstance(self.base_loss, (PWGLoss,HuberPWGLossRegularized)):
            return self.base_loss(y_pred, y_true, iw, w_pdf, gw, gender)
        if isinstance(self.base_loss, WeightedMSELoss):
            return self.base_loss(y_pred, y_true, iw, w_pdf)
        if isinstance(self.base_loss, WeightedLiteMSELoss):
            return self.base_loss(y_pred, y_true, iw)
        return self.base_loss(y_pred, y_true)

class HuberPWGLossRegularized(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def HuberLoss(self, y_true, y_pred, w):
        delta = torch.abs(y_true - y_pred)
        
        l = torch.where(delta < self.beta,
                0.5 * w * delta**2,
                w * self.beta * (delta - 0.5 * self.beta))
        
        return torch.sum(l) / (torch.sum(w)+EPS)

    def forward(self, y_pred, y_true, iw, pi, gw, gender):
        w  = (iw * pi * gw).view(-1)
        g  = gender.view(-1)
        mask_f = g == 0.0
        mask_m = g == 1.0
        
        err_f = self.HuberLoss(y_true[mask_f], y_pred[mask_f], w[mask_f])
        err_m = self.HuberLoss(y_true[mask_m], y_pred[mask_m], w[mask_m])
        return (err_f + err_m) / 2 + self.alpha * torch.sqrt(torch.square(err_f - err_m) + EPS)

    
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
    "SL1":lambda **kwargs: nn.SmoothL1Loss(beta=0.1, **kwargs),
    "nMSE": WeightedMSELoss,
    "nLiteMSE": WeightedLiteMSELoss,
    "PWGLoss": PWGLoss,
    "HuberPWGLossRegularized": HuberPWGLossRegularized}