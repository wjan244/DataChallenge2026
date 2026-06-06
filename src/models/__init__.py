CUSTOM_MODELS = {}

__all__ = ["CUSTOM_MODELS"]
from src.config import*
from src.config_utils import load_config

from src.models.utils_setup_method import setup_domain_adaptation, setup_probing, setup_lora_finetuning,setup_adversarial_probing

METHOD_MAPPING = {"domain_adaptation":setup_domain_adaptation,
                  "probing_training": setup_probing,
                  "reversal_probing": setup_adversarial_probing,
                  "lora_training": setup_lora_finetuning} 

cfg_glob         = load_config(CONFIG_DEFAULT).get("globaux", {})
eps              = float(cfg_glob.get("EPS", 1e-8))

