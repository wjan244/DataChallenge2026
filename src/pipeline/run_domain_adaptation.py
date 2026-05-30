import mlflow

from src.pipeline.evaluation import run_evaluation
from src.config import *

from src.data.data_loader import*
from src.pipeline.train import run_train




def run_domain_adaptation(cfg, file_name, timestamp, experiment_id, precedent_run_id=None, precedent_method=None):
    cfg_mod = cfg["model"]
    cfg_glob = cfg["globaux"]
    cfg_method = cfg["domain_adaptation_training"]

    if cfg_method["run_execution"]==True:

        print(f"début d'entrainement par {cfg_method['method_FT']}")
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"{timestamp}_{cfg_mod}_domain adaptation"):
            get_celeba_train_loader = globals()[cfg_method["loader_factory"]]
            get_celeba_val_loader = globals()[cfg_method["val_loader_factory"]]
            get_challenge_val_loader = globals()[cfg_method["val_loader_challenge"]]



            train_loader = get_celeba_train_loader(batch_size=cfg_glob["BATCH_SIZE"],
                                                   num_workers=NUM_WORKERS, model_name=cfg_mod,
                                                   augmentation=cfg_method.get("augmentation"))

            val_loader = get_celeba_val_loader(batch_size=cfg_glob["BATCH_SIZE"],
                                               num_workers=NUM_WORKERS, model_name=cfg_mod)

            # obtenir le DataLoader du dataset Challenge (ne pas unpacker un DataLoader)
            challenge_val_loader = get_challenge_val_loader(split=cfg_method["val_split"],
                                                            batch_size=cfg_glob["BATCH_SIZE"],
                                                            num_workers=NUM_WORKERS, model_name=cfg_mod)

            run_id, _, _, _, _ = run_train(timestamp, train_loader, val_loader, cfg_mod, cfg_glob, cfg_method, precedent_run_id, precedent_method, prefix=None)
            return_method = cfg_method["method_FT"]

            # évaluer sur CelebA puis sur le dataset Challenge
            run_evaluation(timestamp=timestamp, val_loader=val_loader, method_FT=return_method, cfg_glob=cfg_glob, cfg_mod=cfg_mod, prefix="score sur Celeba", method_kwargs=cfg_method.get("method_kwargs"),index="CeleBa_evaluation")
            run_evaluation(timestamp=timestamp, val_loader=challenge_val_loader, method_FT=return_method, cfg_glob=cfg_glob, cfg_mod=cfg_mod, prefix="score sur Dataset Challenge", method_kwargs=cfg_method.get("method_kwargs"),index="Challenge_evaluation")

            print(f"fin d'entrainement par {cfg_method['method_FT']}")
            return run_id, return_method
    else:
        return precedent_run_id, precedent_method