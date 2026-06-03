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

def metric_fn(female: pd.DataFrame, male: pd.DataFrame, w=None) -> float:
    if w is None:
        err_female = error_fn(female, None)
        err_male = error_fn(male, None)
    else:
        if isinstance(w, (list, tuple)) and len(w) == 2:
            w_female, w_male = w
            err_female = error_fn(female, w_female)
            err_male = error_fn(male, w_male)
        else:
            err_female = error_fn(female, w)
            err_male = error_fn(male, w)

    return float((err_male + err_female) / 2 + abs(err_male - err_female))


class PWScore(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, iw, pi, gender):
        w  = (iw * pi).view(-1)
        se = ((y_true - y_pred) ** 2).view(-1)
        g  = gender.view(-1)
        mask_f = g == 0.0
        mask_m = g == 1.0
        err_f = torch.sum(w[mask_f] * se[mask_f]) / (torch.sum(w[mask_f]) + eps)
        err_m = torch.sum(w[mask_m] * se[mask_m]) / (torch.sum(w[mask_m]) + eps)
        return (err_f + err_m) / 2 + torch.abs(err_f - err_m), err_f, err_m