from pathlib import Path
import timm

import mlflow
import torch
from torch import nn

from . import METHOD_MAPPING
from src.models.utils_lora import inject_lora_transformer


class PatchHead(nn.Module):
    """
    Tête partagée appliquée indépendamment à chaque patch token.
    Entrée  : (B, n_patch, D)
    Sortie  : (B, 1) — ratio occludé / visage ∈ [0,1]
    """
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 3)) # fond / visible / occludé

    def forward(self, patch_tokens):
        logits = self.net(patch_tokens)          # (B, n_patch, 3)
        probs = torch.softmax(logits, dim=-1)    # (B, n_patch, 3)
        p_visible = probs[:, :, 1]               # (B, n_patch)
        p_occluded = probs[:, :, 2]              # (B, n_patch)
        ratio = p_occluded.sum(dim=1) / (
            p_visible.sum(dim=1) + p_occluded.sum(dim=1) + 1e-8)
        return ratio.unsqueeze(1)                # (B, 1)
    

class PatchOcclusionModel(nn.Module):
    """Backbone DINO/timm + tête patch."""

    def __init__(self, backbone):
        super().__init__()
        self.model = backbone

        embed_dim = (
            getattr(backbone, "embed_dim", None)
            or getattr(backbone, "embed_dims", None)
            or getattr(backbone, "num_features", None)
            or 768
        )

        self.num_prefix = (
            getattr(backbone, "num_prefix_tokens", None)
            or getattr(backbone, "num_extra_tokens", None)
            or 5
        )

        self.head = PatchHead(in_dim=embed_dim)

    def forward(self, x):
        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(x)
        else:
            features = self.model(x)

        if isinstance(features, dict):
            if "x_norm_patchtokens" in features:
                patch_tokens = features["x_norm_patchtokens"]
            elif "last_hidden_state" in features:
                tokens = features["last_hidden_state"]
                patch_tokens = tokens[:, self.num_prefix:, :]
            else:
                raise KeyError(
                    f"Format dict inattendu. Clés disponibles : {features.keys()}"
                )

        elif isinstance(features, (tuple, list)):
            features = features[-1]

            if features.ndim == 4:
                patch_tokens = features.flatten(2).transpose(1, 2)
            elif features.ndim == 3:
                patch_tokens = features[:, self.num_prefix:, :]
            else:
                raise ValueError(f"Format tuple inattendu : {features.shape}")

        elif features.ndim == 4:
            patch_tokens = features.flatten(2).transpose(1, 2)

        elif features.ndim == 3:
            patch_tokens = features[:, self.num_prefix:, :]

        else:
            raise ValueError(f"Format inattendu pour features : {features.shape}")

        return self.head(patch_tokens)


class OcclusionModel(nn.Module):
    """
    - instancie le modèle
    - ajoute une sigmoïde en sortie de du modèle
    """
    def __init__(self,model:nn.Module)->None:
        super().__init__()
        self.model = model
        self.sigmoide = nn.Sigmoid()

    def forward(self,x:torch.Tensor)->torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            # gestion du cas multi-têtes
            return {k: self.sigmoide(v) for k, v in out.items()}
        return self.sigmoide(out)
    

def get_model(timestamp, cfg_mod=None, cfg_method=None, precedent_run_id=None, precedent_method=None,
              method: str | None = None, load_checkpoint: bool = False, checkpoint_path: str | None = None,
              num_classes: int | None = None, **method_kwargs) -> nn.Module:
    """instancier le modèle défini et lui affecter une méthode de FineTuning avec les bons poids
    """
    if num_classes is None:
        num_classes = 0
    # récupérer la méthode depuis cfg_method si fourni (exp:"reversal_probing","lora_training"...)
    if cfg_method and hasattr(cfg_method, 'get'):
        method = cfg_method.get("method_FT", method)
    # récupérer le modèle
    model = timm.create_model(cfg_mod,pretrained=True,num_classes=num_classes)

    # insertion de la méthode
        # extraction des poids précédents si entrainement avec méthodes séquentielles
    weights = None
    if precedent_run_id:
        precedent_tag = f"{cfg_mod}_{precedent_method}"
        weights = mlflow.artifacts.download_artifacts(
            run_id=precedent_run_id,
            artifact_path=f"{timestamp}_{precedent_tag}.pt")
        print(f"Poids chargés depuis {precedent_method}: {weights}")

        # injection des poids de la méthode de finetune précédente
    if load_checkpoint and checkpoint_path is not None:
        weights = checkpoint_path

    if weights is not None:
        # charger sur le cpu
        state = torch.load(weights, map_location='cpu')
        model.load_state_dict(state, strict=False)

        # Injecter de la méthode de finetuning suivante
    if method is not None:
        # ajout de LoRA (passé une seule fois dans get_model si la méthode l'exige)
        if method in ("domain_adaptation", "lora_training", "reversal_probing"):
            inj_kwargs = {k: method_kwargs[k] for k in ("rank", "alpha", "dropout") if k in (method_kwargs or {})}
            model = inject_lora_transformer(model, **inj_kwargs)
        # ajout de la méthode de fine tune en cours (contrôle de poids freezé/défreezés)
        model = METHOD_MAPPING[method](model, **(method_kwargs or {}))
   
    # ajout de la sigmoïde au modèle chargé
    model = PatchOcclusionModel(model)
    return model

if __name__ == "__main__":
    from src.config_utils import load_config    
    from torchinfo import summary

    import timm

    cfg = load_config("vit_base_patch16_dinov3.yaml")
    addversarial_section = cfg.get("lora_training")
    cfg_kwargs = addversarial_section.get("method_kwargs") or {}

    model = get_model(
        timestamp=None,
        cfg_mod=cfg["model"], 
        cfg_method=addversarial_section,
        precedent_run_id=None,
        precedent_method=None,
        method="probing_training",
        **cfg_kwargs)
    
    summary(model)
   