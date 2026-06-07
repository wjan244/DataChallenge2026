import os
import yaml
from pathlib import Path

# =====================================================================
# 1. PATCH TRAIN : Mode batch unique
# =====================================================================
from torch.utils.data import DataLoader

_original_iter = DataLoader.__iter__

def patched_iter(self):
    iterator = _original_iter(self)
    try:
        yield next(iterator)
    except StopIteration:
        return

DataLoader.__iter__ = patched_iter


# =====================================================================
# 2. PATCH EVALUATION (Correction de cfg_mod + arguments)
# =====================================================================
# import evaluation module dynamically to avoid executing package __init__ side-effects
import importlib
evaluation = importlib.import_module("src.pipeline.evaluation")

_original_run_evaluation = evaluation.run_evaluation

def patched_run_evaluation(*args, **kwargs):
    # 1. Aligner les noms d'arguments du modèle
    if 'model_name' in kwargs:
        kwargs['cfg_mod'] = kwargs.pop('model_name')
        
    # 2. Supprimer l'argument en trop
    if 'cfg_method' in kwargs:
        kwargs.pop('cfg_method')
        
    # 3. SÉCURITÉ CRUCIALE : Si cfg_mod est None, on le déduit du fichier de config
    if kwargs.get('cfg_mod') is None:
        try:
            # On va chercher le nom du modèle directement dans la config globale active
            if 'cfg_glob' in kwargs and 'MODEL_NAME' in kwargs['cfg_glob']:
                kwargs['cfg_mod'] = kwargs['cfg_glob']['MODEL_NAME']
            else:
                # Valeur de secours basée sur ton fichier de config actuel
                kwargs['cfg_mod'] = 'vit_tiny_patch16_224'
            print(f"⚠️ [PATCH TEST] 'cfg_mod' était None. Restauré à : '{kwargs['cfg_mod']}'")
        except Exception:
            kwargs['cfg_mod'] = 'vit_tiny_patch16_224'

    return _original_run_evaluation(*args, **kwargs)

evaluation.run_evaluation = patched_run_evaluation


# =====================================================================
# 3. Import et exécution
# =====================================================================
from main import main
from src.config import CONFIG_MODELS

ORIGINAL_YAML = CONFIG_MODELS / 'convnextv2.yaml'
TEMP_FILENAME = 'vit_tiny_patch16_224_debug.yaml'
TEMP_YAML_PATH = CONFIG_MODELS / TEMP_FILENAME

if __name__ == "__main__":
    print("🔄 Préparation du fichier de configuration temporaire...")
    
    if not ORIGINAL_YAML.exists():
        raise FileNotFoundError(f"Fichier introuvable : {ORIGINAL_YAML}")
        
    with open(ORIGINAL_YAML, 'r') as f:
        cfg = yaml.safe_load(f)
    
    if "globaux" not in cfg:
        cfg["globaux"] = {}
    cfg["globaux"]["EPOCHS"] = 1
    
    with open(TEMP_YAML_PATH, 'w') as f:
        yaml.safe_dump(cfg, f)
        
    print("🚀 Lancement du pipeline avec mouchard d'évaluation renforcé...\n")
    
    try:
        main(TEMP_FILENAME)
    finally:
        if os.path.exists(TEMP_YAML_PATH):
            os.remove(TEMP_YAML_PATH)
            print("🧹 Fichier temporaire nettoyé.")