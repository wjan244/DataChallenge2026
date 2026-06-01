"""
train_lora.py — Fine-tuning LoRA de DINOv3 à partir du checkpoint probing.

Lancer depuis la racine du projet :
    python train_lora.py

Sortie :
    checkpoints/lora_best.pt       — meilleur checkpoint
    submission/lora_val_preds.csv  — prédictions val avec gt, pred, gender
"""

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

# Réutilisation minimale du code de William : data utils et LoRA
from src.config import IMG_DIR, NUM_WORKERS
from src.data.data_utils import get_challenge_split
from src.data.dataset import Dataset as ChallengeDataset
from src.models.finetuning import inject_lora_transformer, inject_linear_mlp_probing

# ==============================================================================
# CONFIG — tout ce qu'on peut changer est ici, rien ailleurs
# ==============================================================================

MODEL_NAME        = "vit_base_patch16_dinov3.lvd1689m"
PROBING_CKPT      = Path("checkpoints/2026-05-31_01-38-19_vit_base_patch16_dinov3.lvd1689m_probing_training.pt")
SAVE_PATH         = Path("checkpoints/lora_best.pt")
OUTPUT_CSV        = Path("submission/lora_val_preds.csv")

BATCH_SIZE        = 64
LEARNING_RATE     = 2e-4
NUM_EPOCHS        = 20
PATIENCE          = 5
RANK              = 8
ALPHA             = 16
DROPOUT           = 0.2
HIDDEN_SIZE       = 512
SEED              = 42

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
# MODÈLE
# ==============================================================================

class OcclusionModel(nn.Module):
    """Backbone + sigmoid — même structure que William."""
    def __init__(self, backbone):
        super().__init__()
        self.model = backbone
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))


def build_model():
    """
    Construit DINOv3 + LoRA + tête MLP, puis charge le checkpoint probing.
    Les matrices LoRA (nouvelles) restent à leur initialisation aléatoire.
    """
    backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1)
    model = OcclusionModel(backbone)

    # Injection LoRA dans les couches qkv (12 blocs d'attention)
    model = inject_lora_transformer(model, rank=RANK, alpha=ALPHA, dropout=DROPOUT)

    # Tête MLP identique au probing
    model = inject_linear_mlp_probing(model, probing_type="mlp_probing", hidden_size=HIDDEN_SIZE)

    # Chargement du checkpoint probing
    # strict=False car le checkpoint probing ne contient pas les matrices LoRA
    # → elles sont chargées telles quelles (initialisées aléatoirement)
    state_dict = torch.load(PROBING_CKPT, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Paramètres manquants (matrices LoRA, attendu) : {len(missing)}")
    print(f"Paramètres inattendus : {len(unexpected)}")

    # Geler tout sauf LoRA et tête
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name or "head" in name or "classifier" in name

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres entraînables : {n:,}")

    return model


# ==============================================================================
# LOSS
# ==============================================================================

def weighted_mse(pred, target, iw, pi):
    """Loss nMSE : somme pondérée par iw × pi, normalisée par la somme des poids."""
    w = iw * pi
    return (w * (pred - target) ** 2).sum() / (w.sum() + 1e-8)


# ==============================================================================
# MÉTRIQUE DU CHALLENGE
# ==============================================================================

def challenge_score(df):
    """
    Score officiel (plus bas = mieux) :
        Err = Σ pi*(pred-gt)² / Σ pi      avec pi = 1/30 + gt
        Score = (Err_F + Err_M) / 2 + |Err_F - Err_M|
    """
    def error(sub):
        gt   = sub["gt"].values
        pred = sub["pred"].values
        pi   = 1 / 30 + gt
        return np.sum(pi * (pred - gt) ** 2) / (np.sum(pi) + 1e-8)

    females = df[df["gender"] == 0.0]
    males   = df[df["gender"] == 1.0]

    if len(females) == 0 or len(males) == 0:
        return float("inf")

    err_f = error(females)
    err_m = error(males)
    score = (err_f + err_m) / 2 + abs(err_f - err_m)

    return score, err_f, err_m


# ==============================================================================
# DATA
# ==============================================================================

def get_loaders():
    # get_challenge_split() retourne : df_train, df_val_raw, df_val_samp, df_test
    df_train, _, df_val, _ = get_challenge_split()

    # Transforms recommandées par TIMM pour ce modèle
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

    return train_loader, val_loader


# ==============================================================================
# BOUCLES
# ==============================================================================

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="  Train"):
        X        = batch[0].to(DEVICE)
        y        = batch[1].to(DEVICE).float().view(-1, 1)
        # ⚠️  Vérifier l'ordre avec : print([b.shape for b in batch])
        # D'après train.py de William : pi = batch[4], iw = batch[5]
        pi       = batch[4].to(DEVICE).float().view(-1, 1)
        iw       = batch[5].to(DEVICE).float().view(-1, 1)

        optimizer.zero_grad()
        pred = model(X)
        loss = weighted_mse(pred, y, iw, pi)
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
            X       = batch[0].to(DEVICE)
            y       = batch[1]
            gender  = batch[2]
            filename = batch[3]

            pred = model(X).cpu().squeeze()
            if pred.dim() == 0:
                pred = pred.unsqueeze(0)

            for i in range(len(X)):
                records.append({
                    "filename": filename[i],
                    "gt":       float(y[i]),
                    "pred":     float(pred[i]),
                    "gender":   float(gender[i]),
                })

    return pd.DataFrame(records)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n--- Chargement des données ---")
    train_loader, val_loader = get_loaders()

    print("\n--- Construction du modèle ---")
    model = build_model().to(DEVICE)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_score      = float("inf")
    patience_counter = 0

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Entraînement ({NUM_EPOCHS} epochs max, patience={PATIENCE}) ---\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer)
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
