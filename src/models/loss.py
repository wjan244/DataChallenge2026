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

class UniversalLossWrapper(nn.Module):
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss
        self.is_weighted = isinstance(base_loss, WeightedMSELoss)
        self.is_lite_weighted = isinstance(base_loss,WeightedLiteMSELoss)

    def forward(self, *args, **kwargs):
       
        # Try direct forwarding first
        try:
            return self.base_loss(*args, **kwargs)
        except TypeError:
            # Build from positional args when possible
            if len(args) >= 2:
                y_pred_loc, y_true_loc = args[0], args[1]
            else:
                # Can't dispatch without at least y_pred/y_true
                raise

            # Weighted loss signature: (y_pred, y_true, iw, pi)
            if self.is_weighted:
                iw_loc = args[2] if len(args) > 2 else kwargs.get('iw')
                pi_loc = args[3] if len(args) > 3 else kwargs.get('pi')
                return self.base_loss(y_pred_loc, y_true_loc, iw_loc, pi_loc)

            # Lite weighted: (y_pred, y_true, iw)
            if self.is_lite_weighted:
                iw_loc = args[2] if len(args) > 2 else kwargs.get('iw')
                return self.base_loss(y_pred_loc, y_true_loc, iw_loc)

            # PWG-like or other custom losses: attempt to map common extra args
            iw_loc = args[2] if len(args) > 2 else kwargs.get('iw')
            pi_loc = args[3] if len(args) > 3 else kwargs.get('pi')
            gw_loc = args[4] if len(args) > 4 else kwargs.get('gw')
            gender_loc = args[5] if len(args) > 5 else kwargs.get('gender')

            try:
                return self.base_loss(y_pred_loc, y_true_loc, iw_loc, pi_loc, gw_loc, gender_loc)
            except TypeError:
                # Last resort: call with just y_pred, y_true
                return self.base_loss(y_pred_loc, y_true_loc)

    
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
    "nMSE": WeightedMSELoss,
    "nLiteMSE": WeightedLiteMSELoss,
    "PWGLoss": PWGLoss}