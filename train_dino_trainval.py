"""
train_clean_trainval.py — DINOv3 fine-tuné END-TO-END + tête patch.
                           Entraîné sur TRAIN + VAL (pas d'early stopping),
                           avec exclusion des images aux annotations douteuses.

Architecture :
    backbone.forward_features(x)          → (B, N, 768) = [CLS, reg×4, patch×196]
    on jette les tokens prefix (CLS + register)         → (B, 196, 768)
    tête partagée 768→128→3 (fond/visible/occludé) par patch
    softmax → ratio = Σ p_occ / (Σ p_vis + Σ p_occ)  ∈ [0,1]

Deux phases :
    Phase 1 (warmup)   : backbone GELÉ, on entraîne seulement la tête patch.
                         Durée : PHASE1_EPOCHS (fixe — pas d'early stopping).
    Phase 2 (fine-tune): backbone ENTIÈREMENT DÉGELÉ, fine-tuning end-to-end.
                         Durée : PHASE2_EPOCHS (fixe — pas d'early stopping).

Différences vs train_finetune_trainval.py :
    - Les 17 images aux annotations douteuses sont exclues du dataset combiné
      (filtrées avant la concaténation train + val)

Différences vs train_cleandata.py :
    - train + val concaténés en un seul loader d'entraînement
    - pas de val loader → pas de challenge_score pendant l'entraînement
    - pas d'early stopping : durées fixes calées sur le meilleur run précédent
    - on sauvegarde l'état final de chaque phase (pas le "meilleur")
    - export test immédiat à la fin (état final phase 2)

Pondérations :
    - Loss d'ENTRAÎNEMENT : iw × pi_train   (pi_train aplatissable via PI_LOSS_FLOOR)
    - Inférence TEST      : aucune pondération

Lancer depuis la racine du projet :
    python train_clean_trainval.py
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torchvision.transforms import v2
from tqdm import tqdm

from src.config import IMG_DIR, NUM_WORKERS, SUBMISSION_DIR
from src.data.data_utils import get_challenge_split
from src.data.dataset import Dataset as ChallengeDataset

# ==============================================================================
# FICHIERS À EXCLURE
# Ces images sont exclues de train ET de val avant concaténation (annotations douteuses)
# ==============================================================================

EXCLUDE_FILES = {
    "database3/database3/m.017m44/19-FaceId-0_align.webp",
    "database3/database3/m.017yfz/70-FaceId-0_align.webp",
    "database3/database3/m.019x6k/87-FaceId-0_align.webp",
    "database3/database3/m.01c56w/50-FaceId-54_align.webp",
    "database3/database3/m.01flb2/71-FaceId-0_align.webp",
    "database3/database3/m.01m98bv/89-FaceId-0_align.webp",
    "database3/database3/m.01mxqdc/52-FaceId-0_align.webp",
    "database3/database3/m.01n57q/82-FaceId-0_align.webp",
    "database3/database3/m.01npms/2-FaceId-0_align.webp",
    "database3/database3/m.01wzwfb/78-FaceId-0_align.webp",
    "database3/database3/m.01y_15/49-FaceId-0_align.webp",
    "database3/database3/m.01z1pc/41-FaceId-0_align.webp",
    "database3/database3/m.0256sx/61-FaceId-0_align.webp",
    "database3/database3/m.0266wpt/31-FaceId-0_align.webp",
    "database3/database3/m.026nflj/116-FaceId-1_align.webp",
    "database3/database3/m.026qq00/34-FaceId-2_align.webp",
    "database3/database3/m.02mcr6/92-FaceId-0_align.webp",
    "database3/database3/m.014k1v/113-FaceId-0_align.webp",
}

# ==============================================================================
# CONFIG
# ==============================================================================

MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
METHOD_FT  = "clean_trainval"
MODEL_TAG  = f"{MODEL_NAME}_{METHOD_FT}"
TIMESTAMP  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

SAVE_PATH        = Path("checkpoints/clean_trainval_final.pt")   # état final phase 2
PHASE1_SAVE_PATH = Path("checkpoints/clean_trainval_phase1.pt")  # état final phase 1

BATCH_SIZE  = 64
HIDDEN_SIZE = 128          # tête patch : 768 → 128 → 3
SEED        = 42

# Phase 1 — warmup tête (backbone gelé)
# Durée calée sur l'époque du meilleur checkpoint du run de référence (0.00188).
PHASE1_EPOCHS = 5
LR_HEAD_P1    = 1e-3       # tête neuve → LR franc

# Phase 2 — fine-tuning end-to-end (backbone dégelé)
# Idem : calée sur l'époque du meilleur checkpoint du run de référence.
PHASE2_EPOCHS = 6
LR_HEAD_P2    = 5e-4       # tête déjà dégrossie → on l'ajuste
LR_BACKBONE   = 2e-5       # backbone : LR BAS pour ne pas détruire le pré-entraînement.

# --- Pondération de la loss d'entraînement -----------------------------------
PI_LOSS_FLOOR = 1.0 / 30.0   # identique au run de référence

# ==============================================================================
# DEVICE / SEED
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

class PatchHead(nn.Module):
    """
    Tête partagée appliquée indépendamment à chaque patch token.
    Entrée  : (B, n_patch, 768)
    Sortie  : (B, 1) — ratio occludé / visage  ∈ [0,1]
    """
    def __init__(self, in_dim=768, hidden=HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),   # 3 classes : fond / visible / occludé
        )

    def forward(self, patch_tokens):
        logits = self.net(patch_tokens)          # (B, n_patch, 3)
        probs  = torch.softmax(logits, dim=-1)   # (B, n_patch, 3)

        p_visible  = probs[:, :, 1]              # (B, n_patch)
        p_occluded = probs[:, :, 2]              # (B, n_patch)

        ratio = p_occluded.sum(dim=1) / (
            p_visible.sum(dim=1) + p_occluded.sum(dim=1) + 1e-8
        )
        return ratio.unsqueeze(1)                # (B, 1)


class PatchOcclusionModel(nn.Module):
    """Backbone DINOv3 + tête patch. self.model = backbone (nom stable)."""
    def __init__(self, backbone):
        super().__init__()
        self.model = backbone
        embed_dim  = getattr(backbone, "embed_dim", 768)
        self.num_prefix = getattr(backbone, "num_prefix_tokens", 5)
        self.head = PatchHead(in_dim=embed_dim)

    def forward(self, x):
        features     = self.model.forward_features(x)    # (B, N, D)
        patch_tokens = features[:, self.num_prefix:, :]  # (B, n_patch, D)
        return self.head(patch_tokens)                   # (B, 1)


def build_model():
    """DINOv3 pré-entraîné timm, num_classes=0 (on n'utilise que forward_features)."""
    backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model    = PatchOcclusionModel(backbone)
    print(f"num_prefix_tokens : {model.num_prefix}")
    return model


def set_trainable(model, train_backbone: bool):
    """
    La tête est TOUJOURS entraînable. Le backbone l'est selon train_backbone.
    Phase 1 : train_backbone=False (gelé). Phase 2 : train_backbone=True (dégelé).
    """
    for name, param in model.named_parameters():
        is_backbone = name.startswith("model.")
        param.requires_grad = (not is_backbone) or train_backbone
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres entraînables : {n:,}")

# ==============================================================================
# AUGMENTATION (paramètres de William, sans RandomErasing)
# ==============================================================================

def build_transforms():
    """
    Base déterministe (is_training=False → pas de RandomResizedCrop qui changerait
    le ratio d'occlusion). Augmentation appliquée à tout l'ensemble train+val.
    RandomErasing RETIRÉ : il simule une occlusion sans mettre à jour le GT.
    """
    data_config = timm.data.resolve_model_data_config(
        timm.create_model(MODEL_NAME, pretrained=False)
    )
    base = timm.data.create_transform(**data_config, is_training=False)

    augmentation = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.RandomRotation(degrees=15)], p=0.3),
        v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)], p=0.5),
        v2.RandomGrayscale(p=0.05),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
        # v2.RandomErasing(p=0.3),   ← RETIRÉ volontairement
    ])

    train_transform = v2.Compose([base, augmentation])
    return train_transform, base   # base = val_transform (pour export test)

# ==============================================================================
# DATA
# ==============================================================================

def get_train_loader(train_transform):
    """
    Concatène train + val en un seul dataset d'entraînement,
    après exclusion des images aux annotations douteuses (EXCLUDE_FILES).
    Les deux splits utilisent training=True pour avoir accès à iw, pi, gender.
    Les deux reçoivent train_transform (augmentation identique).
    """
    df_train, _, df_val_samp, _ = get_challenge_split()

    # Exclusion des fichiers aberrants AVANT concaténation
    n_before_train = len(df_train)
    n_before_val   = len(df_val_samp)
    df_train    = df_train[~df_train["filename"].isin(EXCLUDE_FILES)].reset_index(drop=True)
    df_val_samp = df_val_samp[~df_val_samp["filename"].isin(EXCLUDE_FILES)].reset_index(drop=True)
    print(f"  Exclusions train : {n_before_train - len(df_train)} images retirées "
          f"({len(df_train)} restantes)")
    print(f"  Exclusions val   : {n_before_val - len(df_val_samp)} images retirées "
          f"({len(df_val_samp)} restantes)")

    df_combined = pd.concat([df_train, df_val_samp], ignore_index=True)
    print(f"  Total après exclusions : {len(df_combined)} images")

    combined_set = ChallengeDataset(
        df_combined, IMG_DIR, training=True, transform=train_transform
    )
    loader = torch.utils.data.DataLoader(
        combined_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
    )
    return loader

# ==============================================================================
# LOSS
# ==============================================================================

def weighted_mse(pred, target, iw, pi):
    """Loss d'entraînement : nMSE pondérée par iw × pi (pas de clamp ici)."""
    w = iw * pi
    return (w * (pred - target) ** 2).sum() / (w.sum() + 1e-8)

# ==============================================================================
# BOUCLES
# ==============================================================================

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  Train", leave=False):
        X  = batch[0].to(DEVICE)
        y  = batch[1].to(DEVICE).float().view(-1, 1)
        iw = batch[4].to(DEVICE).float().view(-1, 1)
        gt = y
        pi = (PI_LOSS_FLOOR + gt)

        optimizer.zero_grad()
        pred = model(X)
        loss = weighted_mse(pred, y, iw, pi)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def run_phase(model, train_loader, optimizer, n_epochs, phase_name):
    """
    Entraîne une phase sur un nombre fixe d'époques (pas d'early stopping).
    Retourne l'état final du modèle.
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(1, n_epochs + 1):
        print(f"\n[{phase_name}] Epoch {epoch}/{n_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer)
        scheduler.step()
        print(f"  train loss : {train_loss:.5f}")

    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return final_state

# ==============================================================================
# EXPORT TEST (format soumission)
# ==============================================================================

def export_test_predictions(model, val_transform):
    """
    Inférence sur le test set, format soumission :
        submission/{TIMESTAMP}_submission_{MODEL_TAG}/test.csv
        colonnes : filename, FaceOcclusion, gender='x'
    Aucune pondération. Clamp [0,1].
    Note : le test set n'est PAS filtré (on ne connaît pas les GT, on prédit tout).
    """
    _, _, _, df_test = get_challenge_split()
    test_set = ChallengeDataset(df_test, IMG_DIR, training=False, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
    )

    model.eval()
    records = []
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="  Test ", leave=False):
            X        = batch[0].to(DEVICE)
            filename = batch[1]
            pred = model(X).cpu().view(-1).clamp(0, 1)
            for i in range(len(X)):
                records.append({
                    "filename":      filename[i],
                    "FaceOcclusion": float(pred[i]),
                    "gender":        "x",
                })

    df = pd.DataFrame(records)[["filename", "FaceOcclusion", "gender"]]
    out_dir  = SUBMISSION_DIR / f"{TIMESTAMP}_submission_{MODEL_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test.csv"
    df.to_csv(out_path, index=False)
    print(f"Soumission test : {out_path}  ({len(df)} lignes)")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n--- Données (train + val combinés, images aberrantes exclues) ---")
    train_transform, val_transform = build_transforms()
    train_loader = get_train_loader(train_transform)

    print("\n--- Modèle (DINOv3 pré-entraîné, tête patch) ---")
    print(f"PI_LOSS_FLOOR = {PI_LOSS_FLOOR:.4f}  (1/30={1/30:.4f} → métrique identique)")
    model = build_model().to(DEVICE)

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ----- Phase 1 : warmup tête (backbone gelé) -----
    print(f"\n=== PHASE 1 : warmup tête (backbone gelé) — {PHASE1_EPOCHS} époques ===")
    set_trainable(model, train_backbone=False)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR_HEAD_P1
    )
    phase1_state = run_phase(model, train_loader, optimizer, PHASE1_EPOCHS, "Phase 1")
    torch.save(phase1_state, PHASE1_SAVE_PATH)
    print(f"Checkpoint phase 1 sauvegardé : {PHASE1_SAVE_PATH}")
    model.load_state_dict(phase1_state)   # repartir de l'état final phase 1

    # ----- Phase 2 : fine-tuning end-to-end (backbone dégelé) -----
    print(f"\n=== PHASE 2 : fine-tuning end-to-end (backbone dégelé) — {PHASE2_EPOCHS} époques ===")
    set_trainable(model, train_backbone=True)

    head_params     = [p for n, p in model.named_parameters()
                       if n.startswith("head.") and p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters()
                       if n.startswith("model.") and p.requires_grad]
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params,     "lr": LR_HEAD_P2},
    ])
    final_state = run_phase(model, train_loader, optimizer, PHASE2_EPOCHS, "Phase 2")
    torch.save(final_state, SAVE_PATH)
    print(f"Checkpoint final sauvegardé : {SAVE_PATH}")

    # ----- Export test à partir de l'état final -----
    print("\n=== EXPORT TEST (état final phase 2) ===")
    model.load_state_dict(final_state)
    export_test_predictions(model, val_transform)


if __name__ == "__main__":
    main()

# ==============================================================================
# NOTES
# ==============================================================================
# Combinaison de train_cleandata.py et train_finetune_trainval.py.
#
# L'exclusion porte sur 17 images aux labels implausibles (fort iw, occlusion
# déclarée élevée mais visuellement nulle ou inversée). Elles sont filtrées
# AVANT la concaténation train+val pour ne pas polluer la loss.
#
# Le test set n'est pas filtré : on ne connaît pas les GT, et l'EXCLUDE_FILES
# a été constitué sur la base des labels — il n'y a pas de raison d'écarter
# des images dont on ne connaît pas le label.
#
# Durées fixes (5+6 époques) calées sur le meilleur run train seul (0.00188).
# Le dataset étant ~25-30 % plus grand, le modèle voit plus d'exemples par
# époque — en particulier dans la queue haute occlusion (rare et critique pour
# la métrique). Les durées peuvent être réévaluées si le score s'améliore.