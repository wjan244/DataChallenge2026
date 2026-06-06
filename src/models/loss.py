import inspect
import torch
import torch.nn as nn

import torch 
import torch.nn as nn

from . import eps
from src.config import*

    

class UniversalLossWrapper(nn.Module):
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, y_pred, y_true, pi=None, gender=None):
        if isinstance(self.base_loss,nn.SmoothL1Loss):
            return self.base_loss(y_pred, y_true)
        if isinstance(self.base_loss, HuberLossRegularized):
            return self.base_loss(y_pred, y_true, pi, gender)


class HuberLossRegularized(nn.Module):
    """
    Perte de Huber pondérée, régularisée pour l'équité de genre.
    Implémente la métrique officielle du challenge :
    Err = Sum(w_i * (p_i - GT_i)^2) / Sum(w_i)
    Score = (Err_F + Err_M)/2 + alpha * |Err_F - Err_M|
    """
    def __init__(self, alpha=1.0, beta=0.1, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        
    def _weighted_huber_per_group(self, y_true, y_pred, weights):
        if len(y_true) == 0:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        delta = torch.abs(y_true - y_pred)
        
        # fonction Huber
        huber_loss = torch.where(
            delta < self.beta,
            0.5 * (delta ** 2),
            self.beta * (delta - 0.5 * self.beta))
        
        weighted_loss = torch.sum(weights * huber_loss) / (torch.sum(weights) + self.eps)
        return weighted_loss

    def forward(self, y_pred, y_true, pi, gender):
        # Extract prediction tensor
        if isinstance(y_pred, dict):
            y_pred_occ = list(y_pred.values())[0]
        else:
            y_pred_occ = y_pred

        # Convert to tensor and aggregate per-sample if necessary
        if torch.is_tensor(y_pred_occ):
            # If model returns per-token/patch predictions [B, T,...], aggregate to one value per sample
            if y_pred_occ.dim() > 1:
                # collapse all non-batch dims by mean
                y_pred_occ = y_pred_occ.view(y_pred_occ.size(0), -1).mean(dim=1)
            else:
                y_pred_occ = y_pred_occ.view(-1)
        else:
            y_pred_occ = torch.tensor(y_pred_occ, dtype=torch.get_default_dtype())
            y_pred_occ = y_pred_occ.view(-1)

        # Ensure y_true is 1-D and aligned with predictions
        if torch.is_tensor(y_true):
            y_true = y_true.view(-1)
        else:
            y_true = torch.tensor(y_true, dtype=y_pred_occ.dtype)
            y_true = y_true.view(-1)

        # Align lengths: keep the minimum length to avoid indexing errors
        min_len = min(y_pred_occ.numel(), y_true.numel())
        y_pred_occ = y_pred_occ[:min_len]
        y_true = y_true[:min_len]

        # Normalize weights (pi) and gender to be 1-D tensors of length min_len
        if pi is None:
            weights = torch.ones(min_len, dtype=y_pred_occ.dtype, device=y_pred_occ.device)
        else:
            if torch.is_tensor(pi):
                weights = pi.view(-1).to(dtype=y_pred_occ.dtype, device=y_pred_occ.device)[:min_len]
            else:
                weights = torch.tensor(pi, dtype=y_pred_occ.dtype, device=y_pred_occ.device).view(-1)[:min_len]

        if gender is None:
            g = torch.zeros(min_len, dtype=torch.float32, device=y_pred_occ.device)
        else:
            if torch.is_tensor(gender):
                g = gender.view(-1).to(device=y_pred_occ.device)[:min_len]
            else:
                g = torch.tensor(gender, device=y_pred_occ.device).view(-1)[:min_len]

        # ensure numeric dtype for mask comparisons
        g = g.to(dtype=torch.float32)

        mask_f = (g == 0.0)
        mask_m = (g == 1.0)

        # If no samples for a group, the helper will return 0.0
        err_f = self._weighted_huber_per_group(y_true[mask_f], y_pred_occ[mask_f], weights[mask_f])
        err_m = self._weighted_huber_per_group(y_true[mask_m], y_pred_occ[mask_m], weights[mask_m])

        diff = err_f - err_m
        loss_fairness = (err_f + err_m) / 2.0 + self.alpha * torch.sqrt(torch.square(diff) + self.eps)

        return loss_fairness
    

def build_loss_fn(loss_descriptor):
    """Build a loss module from either a config dict or a direct loss name string.

    Accepts either:
      - loss_descriptor: dict with key 'loss_name' and optional 'loss_kwargs'
      - loss_descriptor: string giving the loss name
    Returns a UniversalLossWrapper wrapping the instantiated loss module.
    """
    # determine loss_name and kwargs
    if isinstance(loss_descriptor, dict):
        loss_name = loss_descriptor.get("loss_name")
        loss_kwargs = loss_descriptor.get("loss_kwargs") or {}
    elif isinstance(loss_descriptor, str):
        loss_name = loss_descriptor
        loss_kwargs = {}
    else:
        raise ValueError("build_loss_fn expects a dict or a string as input")

    if loss_name not in LOSS_MAPPING:
        raise ValueError(f"Unknown loss: {loss_name}")

    loss_ctor = LOSS_MAPPING[loss_name]
    # instantiate loss (LOSS_MAPPING entries are callables)
    try:
        loss_obj = loss_ctor(**(loss_kwargs or {}))
    except TypeError:
        # fallback: no kwargs accepted
        loss_obj = loss_ctor()

    # ensure returned object is an nn.Module
    if not isinstance(loss_obj, nn.Module):
        # if mapping produced a function, wrap it into a Module via lambda-loss wrapper
        raise TypeError("loss constructor did not return an nn.Module")

    return UniversalLossWrapper(loss_obj)

# Loss mapping
LOSS_MAPPING = {
    "MSE": nn.MSELoss,
    "BCE": nn.BCELoss,
    "HuberLossRegularized":HuberLossRegularized}