import numpy as np
import pandas as pd

from PIL import Image
from scipy.stats import beta
from src.config import*
from src.config_utils import load_config

cfg_glob = load_config(CONFIG_DEFAULT).get("globaux", {})
N_BINS = cfg_glob.get("N_BINS", 20)

bins = np.linspace(0, 1, N_BINS + 1)
bin_center = (bins[:-1] + bins[1:]) / 2
eps = 1e-6

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
    df_distribution = (df_distribution + eps) / (np.sum(df_distribution) + eps)

    if test_distribution is None:
        test_distribution = beta.pdf(bin_center, a=1.5, b=5) + eps
        test_distribution = test_distribution / (np.sum(test_distribution) + eps)

    ratio_KL = test_distribution / (df_distribution + 1e-6)
    bin_categorie = pd.cut(df["FaceOcclusion"], bins=bins, include_lowest=True, ordered=True, labels=False)
    df['D_KL'] = ratio_KL[bin_categorie]

    df_reweight = df.sample(n=n_sample, weights='D_KL', replace=False, random_state=42)
    df_reweight = df_reweight.rename(columns={"D_KL": "iw"})
    return df_reweight, test_distribution, df_distribution


def compute_gender_weights(df, n_bins=N_BINS, alpha=cfg_glob['ALPHA_SMOOTH']):
    """
    Calcule des poids par bin × genre pour équilibrer les distributions H/F.
    Retourne : bins (frontières), w_f (poids femmes par bin), w_m (poids hommes).
    """
    gt     = df["FaceOcclusion"].values
    gender = df["gender"].values

    bins    = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(gt, bins, right=False) - 1, 0, n_bins - 1)

    n_f = np.zeros(n_bins)
    n_m = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_idx == b
        n_f[b] = np.sum((gender == 0.0) & mask)
        n_m[b] = np.sum((gender == 1.0) & mask)

    n_total = n_f + n_m
    w_f = (n_total + 2 * alpha) / (2 * (n_f + alpha))
    w_m = (n_total + 2 * alpha) / (2 * (n_m + alpha))

    return bins, w_f, w_m