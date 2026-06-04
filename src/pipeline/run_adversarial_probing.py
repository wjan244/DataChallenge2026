import mlflow

from src.pipeline.evaluation import run_evaluation
from src.config import *
from src.data.data_loader import*
from src.pipeline.test import run_test, save_split_predictions
from src.pipeline.train import run_train


def run_adversarial_probing(cfg, timestamp, experiment_id, precedent_run_id=None, precedent_method=None):
    cfg_mod = cfg.get("model")
    cfg_glob = cfg.get("globaux", {})
    cfg_method = cfg["adversarial_probing_training"]
    
    method_ft = cfg_method["method_FT"]
    loss_name = cfg_method["loss_name_1"]  # on utilise nMSE comme perte principale de référence

    if cfg_method["run_execution"] == True:
        print(f"début d'entrainement par {method_ft}")
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"{timestamp}_{cfg_mod}_{method_ft}") as run:
            print(f"MLflow run: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}/runs/{run.info.run_id}")
            
            get_challenge_train_loader = globals()[cfg_method["loader_factory"]]
            get_challenge_val_loader = globals()[cfg_method["val_loader_factory"]]

            train_loader = get_challenge_train_loader(batch_size=cfg_glob["BATCH_SIZE"],
                                                      num_workers=NUM_WORKERS, model_name=cfg_mod,
                                                      augmentation=cfg_method["augmentation"])

            val_loader = get_challenge_val_loader(split="val_samp", batch_size=cfg_glob["BATCH_SIZE"],
                                                  num_workers=NUM_WORKERS, model_name=cfg_mod)

            # nettoyage des kwargs pour run_evaluation et le testing
            cfg_method_adv = cfg_method.copy()
            cfg_method_adv.pop("loader_factory", None)
            cfg_method_adv.pop("val_loader_factory", None)
            cfg_method_adv.pop("method_FT", None)
            cfg_method_adv.pop("val_split", None)

            # lancement de la boucle d'entraînement adaptée (gérant les dict d'outputs)
            run_id, _, _, df_test, _ = run_train(timestamp, train_loader, val_loader, cfg_mod, cfg_glob, cfg_method, precedent_run_id, precedent_method, prefix=None)

            # evaluation locale et MLFlow
            run_evaluation(timestamp=timestamp, cfg_glob=cfg_glob, val_loader=val_loader, loss_name=loss_name, method_FT=method_ft, cfg_mod=cfg_mod, prefix=None, method_kwargs=cfg_method_adv.get("method_kwargs"), index=None)
            
            return_method = method_ft
            
            # prédictions sur le split Test et sauvegarde des logs
            test_loader = get_challenge_test_loader(df_test, cfg_glob["BATCH_SIZE"], NUM_WORKERS, model_name=cfg_mod)
            run_test(timestamp, cfg_glob, test_loader, method_ft, cfg_mod, method_kwargs=cfg_method_adv.get("method_kwargs"))
            save_split_predictions(timestamp, train_loader, "train", method_ft, cfg_mod, cfg_method.get("method_kwargs"))

        print(f"fin d'entrainement par {method_ft}")
        return run_id, return_method
    else:
        return precedent_run_id, precedent_method