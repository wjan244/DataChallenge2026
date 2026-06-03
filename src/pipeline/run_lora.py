import mlflow

from src.pipeline.evaluation import run_evaluation
from src.config import *
from src.data.data_loader import*
from src.pipeline.test import run_test, save_split_predictions
from src.pipeline.train import run_train


def run_lora(cfg, timestamp, experiment_id, precedent_run_id=None, precedent_method=None):
    cfg_mod = cfg["model"]
    cfg_glob = cfg["globaux"]
    cfg_method = cfg["lora_training"]
    loss_name = cfg_method.get("loss_name")

    if cfg_method["run_execution"]==True:
        print(f"début d'entrainement par {cfg_method['method_FT']}")
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"{timestamp}_{cfg_mod}_{cfg_method['method_FT']}") as run:
            print(f"MLflow run: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}/runs/{run.info.run_id}")
            get_challenge_train_loader = globals()[cfg_method["loader_factory"]]
            get_challenge_val_loader = globals()[cfg_method["val_loader_factory"]]

            train_loader = get_challenge_train_loader(batch_size=cfg_glob["BATCH_SIZE"],
                                                    num_workers=NUM_WORKERS,model_name=cfg_mod,
                                                    augmentation=cfg_method["augmentation"])
            
            val_loader = get_challenge_val_loader(split="val_samp",batch_size=cfg_glob["BATCH_SIZE"],
                                                num_workers=NUM_WORKERS,model_name=cfg_mod)

            cfg_method_lora = cfg_method.copy()
            cfg_method_lora.pop("loader_factory", None)
            cfg_method_lora.pop("val_loader_factory", None)
            cfg_method_lora.pop("method_FT", None)
            cfg_method_lora.pop("val_split", None)

            run_id, _, _, df_test, _ = run_train(timestamp, train_loader, val_loader, cfg_mod, cfg_glob, cfg_method, precedent_run_id, precedent_method, prefix=None)
            
            run_evaluation(timestamp=timestamp, cfg_glob=cfg_glob, val_loader=val_loader, loss_name=loss_name, method_FT=cfg_method["method_FT"], cfg_mod=cfg_mod, prefix=None, method_kwargs=cfg_method_lora.get("method_kwargs"), index=None)
            return_method = cfg_method["method_FT"]
            test_loader = get_challenge_test_loader(df_test, cfg_glob["BATCH_SIZE"], NUM_WORKERS,model_name=cfg_mod)
            run_test(timestamp, cfg_glob, test_loader, cfg_method["method_FT"], cfg_mod, method_kwargs=cfg_method_lora.get("method_kwargs"))
            save_split_predictions(timestamp, train_loader, "train", cfg_method["method_FT"], cfg_mod, cfg_method.get("method_kwargs"))

        print(f"fin d'entrainement par {cfg_method['method_FT']}")

        return run_id, return_method
    else:
        return precedent_run_id,precedent_method