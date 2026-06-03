import mlflow
import torch

from src.config import *
from src.data.data_utils import get_challenge_split
from src.data.data_loader import *
from src.models.models import get_model
from src.models.loss import UniversalLossWrapper
from src.pipeline.evaluation import save_results
from src.pipeline.run_cnn_ft import _count_trainable_params, _train_phase
from src.models.loss import LOSS_MAPPING

def run_scratch(cfg, timestamp, experiment_id): #precedent_run_id=None,
    cfg_mod = cfg["model"]
    cfg_glob = cfg["globaux"]
    cfg_method = cfg["scratch_training"]

    if cfg_method["run_execution"] != True:
        return None, None

    if cfg_method["loss_name"] not in ("nMSE", "nLiteMSE", "PGWLoss", "PGWLossRegularized"):
        raise ValueError(f"Wrong Loss name {cfg_method['loss_name']}. Exiting")
    
    method_FT = cfg_method["method_FT"]
    learning_rate = cfg_method["learning_rate"]
    num_epoch = cfg_method["num_epoch"]
    mkwargs = cfg_method.get("method_kwargs") or {}
    
    print(f"début d'entrainement par {cfg_method['method_FT']}")
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{timestamp}_{cfg_mod}_{cfg_method['method_FT']}") as run:
        print(f"MLflow run: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}/runs/{run.info.run_id}")
        get_challenge_train_loader = globals()[cfg_method["loader_factory"]]
        get_challenge_val_loader = globals()[cfg_method["val_loader_factory"]]

        train_loader = get_challenge_train_loader(
            batch_size=cfg_glob["BATCH_SIZE"], num_workers=NUM_WORKERS,
            model_name=cfg_mod, augmentation=cfg_method["augmentation"]
        )
        val_loader = get_challenge_val_loader(
            split="val_samp", batch_size=cfg_glob["BATCH_SIZE"],
            num_workers=NUM_WORKERS, model_name=cfg_mod
        )
        _, _, _, df_test = get_challenge_split()
        test_loader = get_challenge_test_loader(df_test, cfg_glob["BATCH_SIZE"], NUM_WORKERS, model_name=cfg_mod)

        model = get_model(cfg_mod, num_classes=1, method=method_FT, **mkwargs)
        model = model.to(DEVICE)
        
        if cfg_glob.get("COMPILE",False) :
            print("Compiling model")
            # compile for faster run but first epoch is slower
            if DEVICE.type == 'mps':
                model = torch.compile(model, backend="aot_eager")
            else:   
                model = torch.compile(model)
            

        _, total_params = _count_trainable_params(model)
        mlflow.log_params({
            **cfg_glob,
            **{k: v for k, v in cfg_method.items() if k != "method_kwargs"},
            **mkwargs,
            "model": cfg_mod,
            "model_tag": f"{cfg_mod}_{method_FT}",
            "timestamp": timestamp,
            "total_params": total_params,
        })

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = CHECKPOINT_DIR / f"{timestamp}_{cfg_mod}_{method_FT}.pt"
        loss_fn = UniversalLossWrapper(LOSS_MAPPING[cfg_method["loss_name"]]())
        global_step = 0
        best_score = float("inf")

        trainable, _ = _count_trainable_params(model)
        mlflow.log_metric("trainable_params", trainable, step=0)
        
        global_step, best_score, _ = _train_phase(
            model, train_loader, val_loader, loss_fn, cfg_glob, cfg_method,
            lr=learning_rate, num_epoch=num_epoch,
            phase_idx=0, save_path=save_path, global_step=global_step, best_score=best_score
        )


        # load best checkpoint, log, evaluate, generate submission
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        mlflow.log_artifact(str(save_path))

        save_results(model, timestamp, train_loader, val_loader, test_loader,
                     loss_name=cfg_method["loss_name"], cfg_mod=cfg_mod, method_FT=method_FT)
    print(f"End of training by {method_FT}")
    return run.info.run_id, method_FT