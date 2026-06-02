from pathlib import Path
import timm

import mlflow
import torch
from torch import nn
from torchinfo import summary

from src.config import DEVICE
from src.models.utils_setup_method import setup_domain_adaptation, setup_probing, setup_lora_finetuning
from src.models.utils_lora import inject_lora_transformer

# Wrapper de méthodes d'adaptation
METHOD_MAPPING = {"domain_adaptation":setup_domain_adaptation,"probing_training": setup_probing,"lora_training": setup_lora_finetuning} 

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
        return self.sigmoide(self.model(x))
    
def get_model(timestamp,cfg_mod, cfg_method,precedent_run_id,precedent_method, num_classes=1, method: str | None = None, load_checkpoint: bool = False, checkpoint_path: str | None = None, stage: str | None = None, **method_kwargs) -> nn.Module:
    """instancier le modèle défini dans config.py et lui affecter une méthode de FineTuning:
    - Linear_probing
    - LoRA
    """
    # récupérer la méthode depuis cfg_method si fourni, sinon garder le param `method`
    if cfg_method is not None:
        # use get to avoid KeyError and preserve explicit `method` if passed
        method = cfg_method.get("method_FT", method)
    
    # récupérer le modèle
    model = timm.create_model(cfg_mod,pretrained=True,num_classes=num_classes)

    # ajout de la sigmoïde au modèle chargé
    model = OcclusionModel(model)

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

    return model


if __name__ == "__main__":
    from src.config_utils import load_config
    
    cfg = load_config("beit3_base_patch16_224.yaml")
    
    # On teste en passant le dictionnaire d'hyperparamètres du YAML via **
    model = get_model(
        model_name=cfg["model"], 
        num_classes=1, 
        method="lora_training",
        **cfg["lora_training"].get("method_kwargs")
    )
    
    summary(model)
