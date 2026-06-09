import mlflow

from src_amo.pipeline.evaluation import run_evaluation
from src_amo.config import *
from src_amo.data.data_loader import*
from src_amo.pipeline.test import run_test, save_split_predictions
from src_amo.pipeline.train import run_train
from src_amo.config import CHECKPOINT_DIR


def run_scratch(cfg, timestamp, experiment_id, precedent_method=None): #precedent_run_id=None,
    cfg_mod = cfg["model"]
    cfg_glob = cfg["globaux"]
    cfg_method = cfg["scratch_training"]

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

            ckpt_name = cfg_method.get("pretrained_checkpoint")
            ckpt_path = CHECKPOINT_DIR / ckpt_name if ckpt_name else None

            run_id, _, _, df_test, _ = run_train(timestamp, train_loader, val_loader, cfg_mod, cfg_glob, cfg_method, precedent_run_id=None, precedent_method=None, prefix=None, pretrained_checkpoint_path=ckpt_path)
            
            run_evaluation(
                    timestamp=timestamp,
                    val_loader=val_loader,
                    method_FT=cfg_method["method_FT"],
                    cfg_glob=cfg_glob,
                    loss_name=cfg_method["loss_name"],
                    cfg_mod=cfg_mod,
                    prefix="scratch",
                    method_kwargs=cfg_method.get("method_kwargs"),
                    index=None
                )
            return_method = cfg_method["method_FT"]
            test_loader = get_challenge_test_loader(df_test, cfg_glob["BATCH_SIZE"], NUM_WORKERS,model_name=cfg_mod)
            run_test(timestamp, test_loader, cfg_method["method_FT"],cfg_mod)
            save_split_predictions(timestamp, train_loader, "train", cfg_method["method_FT"], cfg_mod, cfg_method.get("method_kwargs"))

        print(f"fin d'entrainement par {cfg_method['method_FT']}")

        return run_id, return_method
    else:
        # return precedent_run_id,precedent_method
        return None, None