import time
import torch
import numpy as np
import random

import dagshub
import mlflow

from evaluation import run_evaluation
from src.config import MODEL_NAME, CONFIG_DOMAINE, CONFIG_LINEAR_PROBING, CONFIG_LORA_FT, BATCH_SIZE, NUM_WORKERS
from src.data_loader import get_challenge_test_loader
from test import run_test
from train import run_train



if __name__ == "__main__":
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


    dagshub.init(repo_owner='wjan244', repo_name='DataChallenge2026', mlflow=True)
    experiment_name = "DataChallenge_2026"
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(experiment_name)

    timestamp = str(int(time.time()))

    print(f"début d'entrainement par {CONFIG_DOMAINE["method_FT"]}")
    with mlflow.start_run(experiment_id=experiment.experiment_id,run_name=f"{timestamp}_{MODEL_NAME}_domain adaptation"):
        train_loader = CONFIG_DOMAINE["loader_factory"](BATCH_SIZE, NUM_WORKERS)
        val_loader = CONFIG_DOMAINE["val_loader_factory"](BATCH_SIZE, NUM_WORKERS)

        run_id_1, _, _, df_test,_ = run_train(timestamp=timestamp,train_loader=train_loader,val_loader=val_loader,**CONFIG_DOMAINE,precedent_run_id=None,precedent_method=None)
        run_evaluation(timestamp=timestamp,val_loader=val_loader,method_FT=CONFIG_DOMAINE["method_FT"],prefix="1_domain_adaptation_sampDKL")
        test_loader = get_challenge_test_loader(df_test, BATCH_SIZE, NUM_WORKERS)
        run_test(timestamp,test_loader,CONFIG_DOMAINE["method_FT"])
    print(f"fin d'entrainement par {CONFIG_DOMAINE["method_FT"]}")

    print(f"début d'entrainement par {CONFIG_LINEAR_PROBING["method_FT"]}")
    with mlflow.start_run(experiment_id=experiment_id,run_name=f"{timestamp}_{MODEL_NAME}_FT linear probing"):
        train_loader = CONFIG_LINEAR_PROBING["loader_factory"](BATCH_SIZE, NUM_WORKERS)
        val_loader = CONFIG_LINEAR_PROBING["val_loader_factory"](BATCH_SIZE, NUM_WORKERS)

        run_id_2, _, _, df_test,_ = run_train(timestamp=timestamp,train_loader=train_loader,val_loader=val_loader,**CONFIG_LINEAR_PROBING,precedent_run_id=run_id_1,precedent_method=CONFIG_DOMAINE['method_FT'])
        run_evaluation(timestamp=timestamp,val_loader=val_loader,method_FT=CONFIG_LINEAR_PROBING["method_FT"],prefix="2_linear_probing_sampDKL")
        test_loader = get_challenge_test_loader(df_test, BATCH_SIZE, NUM_WORKERS)
        run_test(timestamp,test_loader,CONFIG_LINEAR_PROBING["method_FT"])
    print(f"fin d'entrainement par {CONFIG_LINEAR_PROBING["method_FT"]}")

    print(f"début d'entrainement par {CONFIG_LORA_FT["method_FT"]}")
    with mlflow.start_run(experiment_id=experiment_id,run_name=f"{timestamp}_{MODEL_NAME}_LORA_FT"):
        train_loader = CONFIG_LORA_FT["loader_factory"](BATCH_SIZE, NUM_WORKERS)
        val_loader = CONFIG_LORA_FT["val_loader_factory"](BATCH_SIZE, NUM_WORKERS)

        run_id_3, _, _, df_test,_ = run_train(timestamp=timestamp,train_loader=train_loader,val_loader=val_loader,**CONFIG_LORA_FT,precedent_run_id=run_id_2,precedent_method=CONFIG_LINEAR_PROBING['method_FT'])
        run_evaluation(timestamp=timestamp,val_loader=val_loader,method_FT=CONFIG_LORA_FT["method_FT"],prefix="3_LoRA_sampDKL")
        test_loader = get_challenge_test_loader(df_test, BATCH_SIZE, NUM_WORKERS)
        run_test(timestamp,test_loader,CONFIG_LORA_FT["method_FT"])
    print(f"fin d'entrainement par {CONFIG_LORA_FT["method_FT"]}")   
        
    mlflow.end_run()