import dagshub
import mlflow

from evaluation import run_evaluation
from src.config import MODEL_NAME,CONFIG_DOMAINE,CONFIG_LINEAR_PROBING,CONFIG_LORA_FT
from test import run_test
#from time import time
from train import run_train

import time

if __name__ == "__main__":
    dagshub.init(repo_owner='wjan244', repo_name='DataChallenge2026', mlflow=True)
    experiment_name = "DataChallenge_2026"
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(experiment_name)

    timestamp = str(int(time.time()))
    print(f"début d'entrainement par {CONFIG_DOMAINE["method_FT"]}")
    with mlflow.start_run(experiment_id=experiment.experiment_id,run_name=f"{timestamp}_{MODEL_NAME}_domain adaptation"):
        run_id_1, df_val_raw, df_val_samp, df_test,_ = run_train(timestamp=timestamp,**CONFIG_DOMAINE,precedent_run_id=None,precedent_method=None)
        run_evaluation(timestamp=timestamp,df_val=df_val_samp,method_FT=CONFIG_DOMAINE["method_FT"],prefix="1_domain_adaptation_sampDKL")
        run_test(timestamp,df_test,CONFIG_DOMAINE["method_FT"])
    print(f"fin d'entrainement par {CONFIG_DOMAINE["method_FT"]}")

    print(f"début d'entrainement par {CONFIG_LINEAR_PROBING["method_FT"]}")
    with mlflow.start_run(experiment_id=experiment_id,run_name=f"{timestamp}_{MODEL_NAME}_FT linear probing"):
        run_id_2, df_val_raw, df_val_samp, df_test,_ = run_train(timestamp=timestamp,**CONFIG_LINEAR_PROBING,precedent_run_id=run_id_1,precedent_method=CONFIG_DOMAINE['method_FT'])
        run_evaluation(timestamp=timestamp,df_val=df_val_samp,method_FT=CONFIG_LINEAR_PROBING["method_FT"],prefix="2_linear_probing_sampDKL")
        run_test(timestamp,df_test,CONFIG_LINEAR_PROBING["method_FT"])
    print(f"fin d'entrainement par {CONFIG_LINEAR_PROBING["method_FT"]}")

    print(f"début d'entrainement par {CONFIG_LORA_FT["method_FT"]}")
    with mlflow.start_run(experiment_id=experiment_id,run_name=f"{timestamp}_{MODEL_NAME}_LORA_FT"):
        run_id_3, df_val_raw, df_val_samp, df_test,_ = run_train(timestamp=timestamp,**CONFIG_LORA_FT,precedent_run_id=run_id_2,precedent_method=CONFIG_LINEAR_PROBING['method_FT'])
        run_evaluation(timestamp=timestamp,df_val=df_val_samp,method_FT=CONFIG_LORA_FT["method_FT"],prefix="3_LoRA_sampDKL")
        run_test(timestamp,df_test,CONFIG_LORA_FT["method_FT"])
    print(f"fin d'entrainement par {CONFIG_LORA_FT["method_FT"]}")   
        
    mlflow.end_run()