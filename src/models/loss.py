import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """Weighted MSE with optional importance weights (iw) and pdf weights (w_pdf).
    If iw or w_pdf is None, they default to 1 for all samples.
    Returns a scalar: sum(w * p * (y - y_pred)^2) / sum(w * p)
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, iw=None, w_pdf=None):
        eps = 1e-8
        if iw is None:
            iw = torch.ones_like(y_true)
        if w_pdf is None:
            w_pdf = torch.ones_like(y_true)

        combined_weights = iw * w_pdf
        numerator = torch.sum(combined_weights * (y_true - y_pred) ** 2)
        denominator = torch.sum(combined_weights)
        return numerator / (denominator + eps)


class WeightedLiteMSELoss(nn.Module):
    """Lite weighted MSE: requires iw to be provided. If iw is None, raise a ValueError.
    Returns the mean of iw * (y - y_pred)^2.
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, iw=None):
        if iw is None:
            raise ValueError("coefficient de reweighting indéfinis")
        return (iw * (y_pred - y_true) ** 2).mean()


class UniversalLossWrapper(nn.Module):
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss
        self.is_weighted = isinstance(base_loss, WeightedMSELoss)
        self.is_lite_weighted = isinstance(base_loss, WeightedLiteMSELoss)

    def forward(self, y_pred, y_true, iw=None, w_pdf=None):
        if self.is_weighted:
            return self.base_loss(y_pred, y_true, iw, w_pdf)
        elif self.is_lite_weighted:
            return self.base_loss(y_pred, y_true, iw)
        return self.base_loss(y_pred, y_true)
