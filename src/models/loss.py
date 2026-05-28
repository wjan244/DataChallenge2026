import torch
import torch.nn as nn

import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, iw=None, w_pdf=None):
        if iw is None:
            iw = torch.ones_like(y_true)
        if w_pdf is None:
            w_pdf = torch.ones_like(y_true)
        eps = 1e-8
        combined_weights = iw * w_pdf   # w_i * p(y_i)
        numerator = torch.sum(combined_weights * (y_true - y_pred) ** 2)    # numérateur
        denominator = torch.sum(combined_weights)   #dénominateur
        
        return numerator / (denominator + eps)


class UniversalLossWrapper(nn.Module):
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss
        self.is_weighted = isinstance(base_loss, WeightedMSELoss)

    def forward(self, y_pred, y_true, iw=None, w_pdf=None):
        if self.is_weighted:
            return self.base_loss(y_pred, y_true, iw, w_pdf)
        return self.base_loss(y_pred, y_true)

# class WeightedMSELoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, y_pred, y_true, iw=None):
#         if iw is None:
#             iw = torch.ones_like(y_true)
#         return (iw * (y_pred - y_true) ** 2).mean()


# class UniversalLossWrapper(nn.Module):
#     def __init__(self, base_loss):
#         super().__init__()
#         self.base_loss = base_loss
#         self.is_weighted = isinstance(base_loss, WeightedMSELoss)

#     def forward(self, y_pred, y_true, iw):
#         if self.is_weighted:
#             return self.base_loss(y_pred, y_true, iw)
#         return self.base_loss(y_pred, y_true)
