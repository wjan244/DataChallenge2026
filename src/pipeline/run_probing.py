import mlflow

from src.pipeline.evaluation import run_evaluation
from src.config import *
from src.data.data_loader import*
from src.pipeline.test import run_test, save_split_predictions
from src.pipeline.train import run_train


def run_probing(cfg,timestamp, experiment_id, precedent_run_id=None, precedent_method=None):
    cfg_mod = cfg["model"]
    cfg_glob = cfg["globaux"]

    # strict: require the 'probing_training' section in the YAML
    cfg_method = cfg["probing_training"]
    method_ft = "probing_training"
    loss_name = cfg_method.get("loss_name")

    if cfg_method["run_execution"] == True:
        print(f"début d'entrainement par {method_ft}")
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"{timestamp}_{cfg_mod}_probing_training"):
            get_challenge_train_loader = globals()[cfg_method["loader_factory"]]
            get_challenge_val_loader = globals()[cfg_method["val_loader_factory"]]

            train_loader = get_challenge_train_loader(batch_size=cfg_glob["BATCH_SIZE"],
                                                    num_workers=NUM_WORKERS, model_name=cfg_mod,
                                                    augmentation=cfg_method["augmentation"])

            val_loader = get_challenge_val_loader(split="val_samp", batch_size=cfg_glob["BATCH_SIZE"],
                                                num_workers=NUM_WORKERS, model_name=cfg_mod)

            cfg_method_lp = cfg_method.copy()
            cfg_method_lp.pop("loader_factory", None)
            cfg_method_lp.pop("val_loader_factory", None)
            cfg_method_lp.pop("method_FT", None)
            cfg_method_lp.pop("val_split", None)

            run_id, _, _, df_test, _ = run_train(timestamp, train_loader, val_loader, cfg_mod, cfg_glob, cfg_method, precedent_run_id, precedent_method, prefix=None)

            run_evaluation(timestamp=timestamp, cfg_glob=cfg_glob, val_loader=val_loader, loss_name=loss_name, method_FT=method_ft, cfg_mod=cfg_mod, prefix=None, method_kwargs=cfg_method_lp.get("method_kwargs"), index=None)
            return_method = method_ft
            test_loader = get_challenge_test_loader(df_test, cfg_glob["BATCH_SIZE"], NUM_WORKERS, model_name=cfg_mod)
            run_test(timestamp, cfg_glob, test_loader, "probing_training", cfg_mod, method_kwargs=cfg_method_lp.get("method_kwargs"))
            save_split_predictions(timestamp, train_loader, "train", method_ft, cfg_mod, cfg_method.get("method_kwargs"))
            save_split_predictions(timestamp, val_loader, "val", method_ft, cfg_mod, cfg_method.get("method_kwargs"))

        print(f"fin d'entrainement par {method_ft}")
        return run_id, return_method
    else:
        return precedent_run_id, precedent_method
