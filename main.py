import time
import torch
import numpy as np
import random
import argparse

import dagshub
import mlflow

from datetime import datetime
from src.config import*
from src.config_utils import load_config
from src.pipeline.run_adversarial_probing import run_adversarial_probing
from src.pipeline.run_domain_adaptation import run_domain_adaptation
from src.pipeline.run_probing import run_probing
from src.pipeline.run_lora import run_lora

SEED = load_config(CONFIG_DEFAULT)["globaux"]["SEED"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def main(file_name):
    cfg = load_config(file_name)

    dagshub.init(repo_owner='wjan244', repo_name='DataChallenge2026', mlflow=True)
    experiment_name = "DataChallenge_2026"
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(experiment_name)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_id, method = run_domain_adaptation(cfg,file_name,timestamp,experiment_id,precedent_run_id=None,precedent_method=None)
    run_id, method = run_probing(cfg,timestamp,experiment_id,precedent_run_id=run_id,precedent_method=method)
    run_id, method = run_lora(cfg, timestamp, experiment_id, precedent_run_id=run_id, precedent_method=method)
    run_id, method = run_adversarial_probing(cfg, timestamp, experiment_id, precedent_run_id=run_id, precedent_method=method)

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="vit_base_patch16_dinov3.yaml",
                        help="YAML filename inside config/models/")
    args = parser.parse_args()
    
    main(args.config)