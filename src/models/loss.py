import torch
import torch.nn as nn

EPS = 1e-8

    
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

#TODO check that iw and pi are not inversed
class WeightedLiteMSELoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, iw):
        try:
            return (iw * (y_pred - y_true) ** 2).mean()
        except ValueError as e:
            print("coefficient de reweighting indéfinis", e)
            return None

class PLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, iw, pi, gw, gender):
        # gender not used but added for eeasier calling of the function without if
        combined_weights = pi
        return torch.sum(combined_weights * (y_true - y_pred) ** 2) / (torch.sum(combined_weights)+EPS)
    
class PWLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, iw, pi, gw, gender):
        # gender not used but added for eeasier calling of the function without if
        combined_weights = iw * pi
        return torch.sum(combined_weights * (y_true - y_pred) ** 2) / (torch.sum(combined_weights)+EPS)
    
    
class PWGLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, iw, pi, gw, gender):
        # gender not used but added for eeasier calling of the function without if
        combined_weights = iw * pi * gw
        return torch.sum(combined_weights * (y_true - y_pred) ** 2) / (torch.sum(combined_weights)+EPS)
    
    
class PWGLossRegularized(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true, iw, pi, gw, gender):
        w  = (iw * pi * gw).view(-1)
        se = ((y_true - y_pred) ** 2).view(-1)
        g  = gender.view(-1)
        mask_f = g == 0.0
        mask_m = g == 1.0
        err_f = torch.sum(w[mask_f] * se[mask_f]) / (torch.sum(w[mask_f]) + EPS)
        err_m = torch.sum(w[mask_m] * se[mask_m]) / (torch.sum(w[mask_m]) + EPS)
        return (err_f + err_m) / 2 + self.alpha * torch.sqrt(torch.square(err_f - err_m) + EPS)


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


class CompoundLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma= 1.0, kappa=1.0):
        # alpha -> 0 : no regularisation
        # beta -> 1: no err M vs F
        # gamma -> 0 : Poss
        # kappa -> 0 : PWLoss
        #gamma kappa -> 1 PWGLoss
        
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.kappa = kappa
        
    def HuberLoss(self, y_true, y_pred, w):
        delta = torch.abs(y_true - y_pred)
        
        l = torch.where(delta < self.beta,
                0.5 * delta**2,
                self.beta * (delta - 0.5 * self.beta))
        
        return torch.sum(w * l) / (torch.sum(w)+EPS)
        

    def forward(self, y_pred, y_true, iw, pi, gw, gender):
        w  = (iw**self.gamma * pi * gw**self.kappa).view(-1)
        g  = gender.view(-1)
        mask_f = g == 0.0
        mask_m = g == 1.0
        
        # Guard against empty gender groups
        # if mask_f.sum() == 0 or mask_m.sum() == 0:
        #     return self.HuberLoss(y_true, y_pred, w)
        
        err_f = self.HuberLoss(y_true[mask_f], y_pred[mask_f], w[mask_f])
        err_m = self.HuberLoss(y_true[mask_m], y_pred[mask_m], w[mask_m])
        return (err_f + err_m) / 2 + self.alpha * torch.sqrt(torch.square(err_f - err_m) + EPS)

    
class UniversalLossWrapper(nn.Module):
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, y_pred, y_true, iw=None, w_pdf=None, gw=None, gender=None):
        if isinstance(self.base_loss, (PWGLoss, PWGLossRegularized, HuberPWGLossRegularized, PWLoss, PLoss, CompoundLoss)):
            return self.base_loss(y_pred, y_true, iw, w_pdf, gw, gender)
        if isinstance(self.base_loss, WeightedMSELoss):
            return self.base_loss(y_pred, y_true, iw, w_pdf)
        if isinstance(self.base_loss, WeightedLiteMSELoss):
            return self.base_loss(y_pred, y_true, iw)
        return self.base_loss(y_pred, y_true)


class PWScore(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, iw, pi, gender):
        w  = (iw * pi).view(-1)
        se = ((y_true - y_pred) ** 2).view(-1)
        g  = gender.view(-1)
        mask_f = g == 0.0
        mask_m = g == 1.0
        err_f = torch.sum(w[mask_f] * se[mask_f]) / (torch.sum(w[mask_f]) + EPS)
        err_m = torch.sum(w[mask_m] * se[mask_m]) / (torch.sum(w[mask_m]) + EPS)
        return (err_f + err_m) / 2 + torch.abs(err_f - err_m), err_f, err_m
    
    
LOSS_MAPPING = {
    "MSE": nn.MSELoss,
    "BCE": nn.BCELoss,
    "nMSE": WeightedMSELoss,
    "nLiteMSE": WeightedLiteMSELoss,
    "PLoss": PLoss,
    "PWLoss": PWLoss,
    "PWGLoss": PWGLoss,
    "PWGLossRegularized": PWGLossRegularized,
    "HuberPWGLossRegularized": HuberPWGLossRegularized,
    "CompoundLoss": CompoundLoss
}