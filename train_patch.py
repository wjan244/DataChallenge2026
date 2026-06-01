"""
train_patch.py — Classification par patch token, par dessus le checkpoint LoRA.

Architecture :
    patch_tokens (B, 196, 768)
    → tête partagée 768→128→3 (fond / visible / occludé)
    → softmax
    → ratio sum(p_occluded) / (sum(p_visible) + sum(p_occluded))  ∈ [0,1]

Poids d'entraînement : iw × pi × w_genre  (rééquilibrage genre par bin Bayésien)

Lancer depuis la racine du projet :
    python train_patch.py
"""

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from src.config import IMG_DIR, NUM_WORKERS
from src.data.data_utils import get_challenge_split
from src.data.dataset import Dataset as ChallengeDataset
from src.models.finetuning import inject_lora_transformer

# ==============================================================================
# CONFIG
# ==============================================================================

MODEL_NAME   = "vit_base_patch16_dinov3.lvd1689m"
LORA_CKPT    = Path("checkpoints/lora_best.pt")
SAVE_PATH    = Path("checkpoints/patch_best.pt")
OUTPUT_CSV   = Path("submission/patch_val_preds.csv")

BATCH_SIZE   = 64
LEARNING_RATE = 2e-4
NUM_EPOCHS   = 20
PATIENCE     = 5
RANK         = 8
ALPHA_LORA   = 16
DROPOUT      = 0.2
HIDDEN_SIZE  = 128   # tête patch : 768 → 128 → 3
SEED         = 42

# Poids d'équité genre
N_BINS       = 30
ALPHA_SMOOTH = 50    # pseudo-comptes Bayésien

# Indices des tokens de patch dans forward_features
# forward_features → (B, 201, 768) : [CLS, reg×4, patch×196]
PATCH_START  = 5

# ==============================================================================
# DEVICE
# ==============================================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Device : {DEVICE}")
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==============================================================================
# POIDS D'ÉQUITÉ GENRE
# ==============================================================================

def compute_gender_weights(df, n_bins=N_BINS, alpha=ALPHA_SMOOTH):
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


def lookup_gender_weights(y_np, gender_np, bins, w_f, w_m):
    """Retourne le poids genre de chaque exemple du batch (numpy → tensor)."""
    bin_idx = np.clip(np.digitize(y_np, bins, right=False) - 1, 0, len(w_f) - 1)
    weights = np.where(gender_np == 0.0, w_f[bin_idx], w_m[bin_idx])
    return torch.tensor(weights, dtype=torch.float32)

# ==============================================================================
# MODÈLE
# ==============================================================================

class PatchHead(nn.Module):
    """
    Tête partagée appliquée indépendamment à chaque patch token.
    Entrée  : (B, 196, 768)
    Sortie  : (B, 1)  — ratio occludé / visage
    """
    def __init__(self, hidden=HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, patch_tokens):
        logits = self.net(patch_tokens)          # (B, 196, 3)
        probs  = torch.softmax(logits, dim=-1)   # (B, 196, 3)

        p_visible  = probs[:, :, 1]              # (B, 196)
        p_occluded = probs[:, :, 2]              # (B, 196)

        # Ratio géométrique : occludé / (visible + occludé)
        ratio = p_occluded.sum(dim=1) / (
            p_visible.sum(dim=1) + p_occluded.sum(dim=1) + 1e-8
        )
        return ratio.unsqueeze(1)                # (B, 1)


class PatchOcclusionModel(nn.Module):
    """Backbone LoRA + tête patch. Même nommage (self.model) pour charger le checkpoint."""
    def __init__(self, backbone):
        super().__init__()
        self.model = backbone
        self.head  = PatchHead()

    def forward(self, x):
        features     = self.model.forward_features(x)   # (B, 201, 768)
        patch_tokens = features[:, PATCH_START:, :]     # (B, 196, 768)
        return self.head(patch_tokens)                  # (B, 1)


def build_model():
    """
    Construit le backbone LoRA et charge le checkpoint lora_best.pt.
    Gèle tout sauf la tête patch.
    """
    backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1)
    model    = PatchOcclusionModel(backbone)

    # Injection LoRA identique au run précédent
    model = inject_lora_transformer(model, rank=RANK, alpha=ALPHA_LORA, dropout=DROPOUT)

    # Chargement checkpoint LoRA
    # strict=False : le checkpoint a une clé "sigmoid.*" absente ici → ignorée
    #                la clé "head.*" est nouvelle → initialisée aléatoirement
    state_dict = torch.load(LORA_CKPT, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Clés manquantes  (head.*, attendu) : {len(missing)}")
    print(f"Clés inattendues (sigmoid.*)       : {len(unexpected)}")

    # Geler tout sauf la tête patch (le backbone LoRA reste gelé)
    for name, param in model.named_parameters():
        param.requires_grad = "head." in name

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres entraînables : {n:,}")

    return model

# ==============================================================================
# LOSS ET MÉTRIQUE
# ==============================================================================

def weighted_mse(pred, target, iw, pi, w_genre):
    """Loss nMSE pondérée par iw × pi × w_genre."""
    w = iw * pi * w_genre
    return (w * (pred - target) ** 2).sum() / (w.sum() + 1e-8)


def challenge_score(df):
    """Score officiel split genre (plus bas = mieux)."""

    def error(sub):
        gt   = sub["gt"].values
        pred = sub["pred"].values
        iw   = sub["iw"].values
        pi   = 1 / 30 + gt
        w    = iw * pi
        return np.sum(w * (pred - gt) ** 2) / (np.sum(w) + 1e-8)

    females = df[df["gender"] == 0.0]
    males   = df[df["gender"] == 1.0]
    if len(females) == 0 or len(males) == 0:
        return float("inf"), float("inf"), float("inf")

    err_f = error(females)
    err_m = error(males)
    return (err_f + err_m) / 2 + abs(err_f - err_m), err_f, err_m

# ==============================================================================
# DATA
# ==============================================================================

def get_loaders():
    df_train, _, df_val, _ = get_challenge_split()

    data_config     = timm.data.resolve_model_data_config(
        timm.create_model(MODEL_NAME, pretrained=False)
    )
    train_transform = timm.data.create_transform(**data_config, is_training=False)
    val_transform   = timm.data.create_transform(**data_config, is_training=False)

    train_set = ChallengeDataset(df_train, IMG_DIR, training=True, transform=train_transform)
    val_set   = ChallengeDataset(df_val,   IMG_DIR, training=True, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
    )

    return train_loader, val_loader, df_train

# ==============================================================================
# BOUCLES
# ==============================================================================

def train_epoch(model, loader, optimizer, bins, w_f, w_m):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="  Train"):
        X      = batch[0].to(DEVICE)
        y      = batch[1].to(DEVICE).float().view(-1, 1)
        gender = batch[2]
        iw     = batch[4].to(DEVICE).float().view(-1, 1)
        pi     = batch[5].to(DEVICE).float().view(-1, 1)


        # Poids d'équité genre (lookup vectorisé, CPU → GPU)
        w_genre = lookup_gender_weights(
            batch[1].numpy(), gender.numpy(), bins, w_f, w_m
        ).to(DEVICE).view(-1, 1)

        optimizer.zero_grad()
        pred = model(X)
        loss = weighted_mse(pred, y, iw, pi, w_genre)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader):
    model.eval()
    records = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="  Val  "):
            X        = batch[0].to(DEVICE)
            y        = batch[1]
            gender   = batch[2]
            filename = batch[3]
            iw       = batch[4]

            pred = model(X).cpu().squeeze()
            if pred.dim() == 0:
                pred = pred.unsqueeze(0)

            for i in range(len(X)):
                records.append({
                    "filename": filename[i],
                    "gt":       float(y[i]),
                    "pred":     float(pred[i]),
                    "gender":   float(gender[i]),
                    "iw":       float(iw[i]),
                })

    return pd.DataFrame(records)

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    state_dict = torch.load(LORA_CKPT, map_location="cpu")
    model = build_model()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing:", missing)
    print("Unexpected:", unexpected)

    print("\n--- Chargement des données ---")
    train_loader, val_loader, df_train = get_loaders()

    print("\n--- Calcul des poids d'équité genre ---")
    bins, w_f, w_m = compute_gender_weights(df_train)
    print(f"Poids F : min={w_f.min():.3f}, max={w_f.max():.3f}")
    print(f"Poids H : min={w_m.min():.3f}, max={w_m.max():.3f}")

    print("\n--- Construction du modèle ---")
    model = build_model().to(DEVICE)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_score       = float("inf")
    patience_counter = 0

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Entraînement ({NUM_EPOCHS} epochs max, patience={PATIENCE}) ---\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, bins, w_f, w_m)
        df_val     = validate(model, val_loader)
        score, err_f, err_m = challenge_score(df_val)

        scheduler.step()

        print(f"  train loss : {train_loss:.5f}")
        print(f"  val score  : {score:.5f}  (F={err_f:.5f}, M={err_m:.5f})")

        if score < best_score:
            best_score       = score
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
            df_val.to_csv(OUTPUT_CSV, index=False)
            print(f"  ✅ Meilleur score — checkpoint et CSV sauvegardés")
        else:
            patience_counter += 1
            print(f"  Pas d'amélioration ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping à l'epoch {epoch}.")
            break

    print(f"\nTerminé. Meilleur score val : {best_score:.5f}")
    print(f"Checkpoint : {SAVE_PATH}")
    print(f"Prédictions val : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()