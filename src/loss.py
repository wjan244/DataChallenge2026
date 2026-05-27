import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, iw=None):
        if iw is None:
            iw = torch.ones_like(y_true)
        return (iw * (y_pred - y_true) ** 2).mean()


class UniversalLossWrapper(nn.Module):
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss
        self.is_weighted = isinstance(base_loss, WeightedMSELoss)

    def forward(self, y_pred, y_true, iw):
        if self.is_weighted:
            return self.base_loss(y_pred, y_true, iw)
        return self.base_loss(y_pred, y_true)
