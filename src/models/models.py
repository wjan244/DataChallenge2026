from pathlib import Path
import timm

import mlflow
import torch
from torch import nn
from torchinfo import summary

from src.config import DEVICE
from src.models.utils_setup_method import setup_domain_adaptation, setup_probing, setup_lora_finetuning,setup_adversarial_probing
from src.models.utils_lora import inject_lora_transformer

# Wrapper de méthodes d'adaptation
METHOD_MAPPING = {"domain_adaptation":setup_domain_adaptation,
                  "probing_training": setup_probing,
                  "reversial_probing_training": setup_adversarial_probing,
                  "lora_training": setup_lora_finetuning} 

class OcclusionModel(nn.Module):
    """
    - instancie le modèle défini dans config.py
    - ajoute une sigmoïde en sortie de du modèle
    """
    def __init__(self,model:nn.Module)->None:
        super().__init__()
        self.model = model
        self.sigmoide = nn.Sigmoid()

    def forward(self,x:torch.Tensor)->torch.Tensor:
        out = self.model(x)
        # some setup methods (adversarial probing) return a dict of heads
        if isinstance(out, dict):
            # apply sigmoid to each tensor head and return a dict with same keys
            return {k: self.sigmoide(v) for k, v in out.items()}
        return self.sigmoide(out)
    
def get_model(timestamp, cfg_mod=None, cfg_method=None, precedent_run_id=None, precedent_method=None,
              method: str | None = None, load_checkpoint: bool = False, checkpoint_path: str | None = None,
              stage: str | None = None, num_classes: int | None = None, model_name: str | None = None, **method_kwargs) -> nn.Module:
    """instancier le modèle défini dans config.py et lui affecter une méthode de FineTuning:
    - Linear_probing
    - LoRA
    """
    if num_classes is None:
        num_classes = 1
    # récupérer la méthode depuis cfg_method si fourni, sinon garder le param `method`
    if cfg_method is not None:
        # use get to avoid KeyError and preserve explicit `method` if passed
        method = cfg_method.get("method_FT", method)
    
    # récupérer le modèle
    model = timm.create_model(cfg_mod,pretrained=True,num_classes=num_classes)

    # insertion de la méthode
        # valider la méthode
    if method is not None and method not in METHOD_MAPPING:
        raise ValueError(f"Unknown method: {method}")

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
        if method in ("domain_adaptation", "lora_training"):
            inj_kwargs = {k: method_kwargs[k] for k in ("rank", "alpha", "dropout") if k in (method_kwargs or {})}
            model = inject_lora_transformer(model, **inj_kwargs)
        # ajout de la méthode de fine tune en cours (contrôle de poids freezé/défreezés)
        model = METHOD_MAPPING[method](model, **(method_kwargs or {}))
   
    # ajout de la sigmoïde au modèle chargé
    model = OcclusionModel(model)
    return model


if __name__ == "__main__":
    from src.config_utils import load_config    
    cfg = load_config("vit_tiny_patch16_224.yaml")
    cfg_method = cfg.get("method_kwargs") or {}
    cfg_method_kwargs = cfg_method.get("method_kwargs") or {}
    # On teste en passant le dictionnaire d'hyperparamètres du YAML via **

    model = get_model(
        timestamp=None,
        cfg_mod=cfg["model"],
        model_name=cfg["model"], 
        cfg_method=cfg["reversial_probing_training"],
        precedent_run_id=None,
        precedent_method=None,
        method="multi_probing_training",
        **cfg_method_kwargs
    )
    
    summary(model)
