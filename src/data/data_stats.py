import numpy as np
import pandas as pd

from . import N_BINS,N_BINS_GENDER,ALPHA_SMOOTH,eps
from PIL import Image
from scipy.stats import beta
from src.config import*


bins = np.linspace(0, 1, N_BINS + 1)
bin_center = (bins[:-1] + bins[1:]) / 2

BINS = torch.linspace(0, 1, steps=31, dtype=torch.float64)

def get_test_distribution_from_screenshot(screenshot_path, n_bins=N_BINS):
    """
    screenshot_path : capture d'écran cadrée sur le graphique test uniquement
    (juste la zone des barres, sans axes ni labels)
    """

    arr = np.array(Image.open(screenshot_path).convert("RGB"))
    _, plot_width = arr.shape[:2]
    
    # détection des barres bleues
    blue_mask = (arr[:, :, 2].astype(int) - arr[:, :, 0].astype(int)) > 5

    # agréger en n_bins
    bin_counts = np.zeros(n_bins)
    bw = plot_width / n_bins
    for i in range(n_bins):
        s = int(i * bw)
        e = max(s + 1, int((i + 1) * bw))
    
        # rectangle de hauteur plot_height, largeur (e-s)
        patch = blue_mask[:, s:e]          # shape: (plot_height, e-s)
        bin_counts[i] = patch.mean()       # proportion de pixels bleus dans le rectangle

    return (bin_counts + eps) / (bin_counts.sum() + eps * n_bins)


def distribution_adaptation_reweight(n_sample, df, test_distribution):
    """ Adaptation basée sur la distribution estimée par screenshot (Reweighting) """

    df_distribution, _ = np.histogram(df["FaceOcclusion"], bins=N_BINS, density=True) 
    df_distribution = (df_distribution+eps) / (np.sum(df_distribution) + eps)

    if test_distribution is None:
        test_distribution = beta.pdf(bin_center, a=1.5, b=5) + eps
        test_distribution = test_distribution / (np.sum(test_distribution) + eps)

    ratio_KL = test_distribution / (df_distribution)
    bin_categorie = pd.cut(df["FaceOcclusion"], bins=bins, include_lowest=True, ordered=True, labels=False)
    df['D_KL'] = ratio_KL[bin_categorie]

    df_reweight = df.sample(n=n_sample, weights='D_KL', replace=True, random_state=42)
    df_reweight = df_reweight.rename(columns={"D_KL": "iw"})
    return df_reweight, test_distribution, df_distribution


def compute_gender_weights(y_all, gender_all):
    """Compute per-bin × gender balancing weights from the full training set.

    y_all, gender_all: 1-D float32 tensors over the full training set.
    Returns w_f, w_m: 1-D float32 tensors of length N_BINS_GENDER.
    """
    device = y_all.device
    bins = BINS.float().to(device)
    bin_idx = (torch.bucketize(y_all.float(), bins, right=False) - 1).clamp(0, N_BINS_GENDER - 1)

    female = (gender_all == 0.0).float()
    male   = (gender_all == 1.0).float()
    n_f = torch.zeros(N_BINS_GENDER, device=device).scatter_add(0, bin_idx, female)
    n_m = torch.zeros(N_BINS_GENDER, device=device).scatter_add(0, bin_idx, male)

    n_total = n_f + n_m
    w_f = (n_total + 2 * ALPHA_SMOOTH) / (2 * (n_f + ALPHA_SMOOTH))
    w_m = (n_total + 2 * ALPHA_SMOOTH) / (2 * (n_m + ALPHA_SMOOTH))
    return w_f, w_m


def lookup_gender_weights(y, gender, w_f, w_m) -> np.float32:
    """Look up precomputed gender bin weight for a single sample.

    y, gender: scalar (float, np.float32, etc.).
    w_f, w_m: 1-D float32 tensors of length N_BINS_GENDER (from Dataset.W_F/W_M).
    Returns: np.float32 — compatible with iw/pi in Dataset.__getitem__.
    """
    y_t = torch.tensor([float(y)], dtype=torch.float32)
    bin_idx = (torch.bucketize(y_t, BINS.float(), right=False) - 1).clamp(0, N_BINS_GENDER - 1)
    w = w_f[bin_idx] if float(gender) == 0.0 else w_m[bin_idx]
    return np.float32(w.item())
