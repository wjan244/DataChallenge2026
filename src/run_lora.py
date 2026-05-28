import mlflow

from evaluation import run_evaluation
from src.config import *
from src.data_loader import*
from src.test import run_test
from src.train import run_train


def run_lora(cfg, timestamp, experiment_id, precedent_run_id=None, precedent_method=None):
    cfg_mod = cfg["model"]
    cfg_glob = cfg["globaux"]
    cfg_method = cfg["lora_training"]

    print(f"début d'entrainement par {cfg_method['method_FT']}")
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{timestamp}_{cfg_mod}_{cfg_method['method_FT']}"):
        get_challenge_train_loader = globals()[cfg_method["loader_factory"]]
        get_challenge_val_loader = globals()[cfg_method["val_loader_factory"]]

        train_loader = get_challenge_train_loader(batch_size=cfg_glob["BATCH_SIZE"],
                                                  num_workers=NUM_WORKERS)
        
        val_loader = get_challenge_val_loader(split="val_samp",batch_size=cfg_glob["BATCH_SIZE"],
                                              num_workers=NUM_WORKERS,model_name=cfg_mod)

        cfg_method_lora = cfg_method.copy()
        cfg_method_lora.pop("loader_factory", None)
        cfg_method_lora.pop("val_loader_factory", None)
        cfg_method_lora.pop("method_FT", None)
        cfg_method_lora.pop("val_split", None)

        run_id, _, _, df_test, _ = run_train(timestamp, train_loader, val_loader, cfg_mod, cfg_glob, cfg_method, precedent_run_id, precedent_method, prefix=None)
        
        run_evaluation(timestamp=timestamp, val_loader=val_loader, method_FT=cfg_method["method_FT"], cfg_glob=cfg_glob, cfg_method=cfg_method, model_name=cfg_mod, prefix=None)
        return_method = cfg_method["method_FT"]
        test_loader = get_challenge_test_loader(df_test, cfg_glob["BATCH_SIZE"], NUM_WORKERS)
        run_test(timestamp, test_loader, cfg_method["method_FT"])

    print(f"fin d'entrainement par {cfg_method['method_FT']}")

    return run_id, return_method