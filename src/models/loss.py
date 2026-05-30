import torch
import torch.nn as nn

import torch
import torch.nn as nn

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
        if self.is_weighted:
            return self.base_loss(y_pred, y_true, iw, w_pdf)
        elif self.is_lite_weighted:
            return self.base_loss(y_pred, y_true, iw)

        return self.base_loss(y_pred, y_true)
