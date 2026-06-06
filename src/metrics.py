import numpy as np
import pandas as pd
import torch

import torch.nn as nn

from src.config import*
from src.config_utils import load_config

cfg_glob = load_config(CONFIG_DEFAULT).get("globaux", {})
eps = float(cfg_glob.get("EPS", 1e-8))

def error_fn(df: pd.DataFrame, w=None) -> float:
    pred = df.loc[:, "pred"].astype(float).to_numpy()
    ground_truth = df.loc[:, "FaceOcclusion"].astype(float).to_numpy()

    if w is None:
        weight = 1/30 + ground_truth
        return np.sum(((pred - ground_truth)**2) * weight, axis=0) / np.sum(weight, axis=0)
    else:
        return np.sum(((pred - ground_truth)**2) * w, axis=0) / np.sum(w, axis=0)


def metric_fn(female:pd.DataFrame, male:pd.DataFrame)->float:
    err_male = error_fn(male)
    err_female = error_fn(female)
    return (err_male + err_female) / 2 + abs(err_male - err_female)


class PWScore(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, pi, gender):
        # Normalize predictions: extract tensor and aggregate per-sample if needed
        if isinstance(y_pred, dict):
            y_pred_t = list(y_pred.values())[0]
        else:
            y_pred_t = y_pred

        if torch.is_tensor(y_pred_t):
            if y_pred_t.dim() > 1:
                # collapse non-batch dims (mean over tokens/patches)
                y_pred_t = y_pred_t.view(y_pred_t.size(0), -1).mean(dim=1)
            else:
                y_pred_t = y_pred_t.view(-1)
        else:
            y_pred_t = torch.tensor(y_pred_t)
            y_pred_t = y_pred_t.view(-1)

        # ensure y_true tensor
        if torch.is_tensor(y_true):
            y_true_t = y_true.view(-1)
        else:
            y_true_t = torch.tensor(y_true)
            y_true_t = y_true_t.view(-1)

        # weights and gender to 1-D tensors
        if pi is None:
            w = torch.ones_like(y_true_t)
        else:
            w = pi.view(-1) if torch.is_tensor(pi) else torch.tensor(pi).view(-1)

        if gender is None:
            g = torch.zeros_like(y_true_t)
        else:
            g = gender.view(-1) if torch.is_tensor(gender) else torch.tensor(gender).view(-1)

        # align all to the same minimum length to avoid indexing errors
        min_len = int(min(y_pred_t.numel(), y_true_t.numel(), w.numel(), g.numel()))
        if min_len == 0:
            # nothing to score
            zero = torch.tensor(0.0, device=y_pred_t.device if torch.is_tensor(y_pred_t) else None)
            return zero, zero, zero

        y_pred_t = y_pred_t[:min_len].to(dtype=torch.float32)
        y_true_t = y_true_t[:min_len].to(dtype=torch.float32)
        w = w[:min_len].to(dtype=torch.float32)
        g = g[:min_len].to(dtype=torch.float32)

        se = ((y_true_t - y_pred_t) ** 2).view(-1)

        mask_f = (g == 0.0)
        mask_m = (g == 1.0)

        err_f = torch.sum(w[mask_f] * se[mask_f]) / (torch.sum(w[mask_f]) + eps)
        err_m = torch.sum(w[mask_m] * se[mask_m]) / (torch.sum(w[mask_m]) + eps)
        return (err_f + err_m) / 2 + torch.abs(err_f - err_m), err_f, err_m