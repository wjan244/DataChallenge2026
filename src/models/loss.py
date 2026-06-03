import torch
import torch.nn as nn

import torch
import torch.nn as nn

from src.config import*
from src.config_utils import load_config

cfg_glob = load_config(CONFIG_DEFAULT).get("globaux", {})
EPS= cfg_glob.get("EPS", 1e-8)
    
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

class UniversalLossWrapper(nn.Module):
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss
        self.is_weighted = isinstance(base_loss, WeightedMSELoss)
        self.is_lite_weighted = isinstance(base_loss,WeightedLiteMSELoss)

    def forward(self, y_pred, y_true, iw=None, w_pdf=None):
        # Accept flexible args/kwargs to support custom losses (e.g. PWGLoss)
        # If the wrapped loss is one of the known weighted variants, call it using
        # the canonical signature; otherwise forward all args to the base loss.
        if self.is_weighted:
            return self.base_loss(y_pred, y_true, iw, w_pdf)
        elif self.is_lite_weighted:
            return self.base_loss(y_pred, y_true, iw)

        # Fallback: forward any additional positional/keyword args directly.
        try:
            return self.base_loss(y_pred, y_true, iw, w_pdf)
        except TypeError:
            # Last-resort: call with flexible signature
            return self.base_loss(y_pred, y_true, iw, w_pdf)

    
class PWGLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, iw, pi, gw, gender):
        # gender not used but added for eeasier calling of the function without if
        combined_weights = iw * pi * gw
        return torch.sum(combined_weights * (y_true - y_pred) ** 2) / (torch.sum(combined_weights)+EPS)
    


# Loss mapping
LOSS_MAPPING = {
    "MSE": nn.MSELoss,
    "BCE": nn.BCELoss,
    "nMSE": WeightedMSELoss,
    "nLiteMSE": WeightedLiteMSELoss,
    "PWGLoss": PWGLoss}