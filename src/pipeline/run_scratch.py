import mlflow

from src.config import *
from src.models.models import get_model
from src.pipeline.evaluation import save_results
from src.pipeline.run_cnn_ft import (
    VALID_LOSS_NAMES,
    _build_loaders,
    _compile_model,
    _build_loss_fn,
    _load_best_checkpoint,
    _count_trainable_params,
    _train_phase,
)


def run_scratch(cfg, timestamp, experiment_id):
    cfg_mod = cfg["model"]
    cfg_glob = cfg["globaux"]
    cfg_method = cfg["scratch_training"]

    if cfg_method["run_execution"] != True:
        return None, None

    if cfg_method["loss_name"] not in VALID_LOSS_NAMES:
        raise ValueError(f"Wrong Loss name {cfg_method['loss_name']}. Exiting")

    method_FT = cfg_method["method_FT"]
    learning_rate = cfg_method["learning_rate"]
    num_epoch = cfg_method["num_epoch"]
    mkwargs = cfg_method.get("method_kwargs") or {}

    print(f"Start of training by {method_FT}")
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{timestamp}_{cfg_mod}_{method_FT}") as run:
        print(f"MLflow run: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}/runs/{run.info.run_id}")

        train_loader, val_loader, test_loader = _build_loaders(cfg_glob, cfg_method, cfg_mod)

        model = get_model(cfg_mod, num_classes=1, method=method_FT, **mkwargs)
        model = model.to(DEVICE)
        model = _compile_model(model, cfg_glob)

        if cfg_method.get("pretrained_checkpoint"):
            ckpt = CHECKPOINT_DIR / cfg_method["pretrained_checkpoint"]
            print(f"Resuming from checkpoint: {ckpt}")
            _load_best_checkpoint(model, ckpt)

        trainable, total_params = _count_trainable_params(model)
        mlflow.log_params({
            **cfg_glob,
            **{k: v for k, v in cfg_method.items() if k != "method_kwargs"},
            **mkwargs,
            "model": cfg_mod,
            "model_tag": f"{cfg_mod}_{method_FT}",
            "timestamp": timestamp,
            "total_params": total_params,
            "loss_alpha": cfg_method.get("loss_alpha", 1.0),
        })

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = CHECKPOINT_DIR / f"{timestamp}_{cfg_mod}_{method_FT}.pt"
        loss_fn = _build_loss_fn(cfg_method)

        global_step = 0
        best_score = float("inf")
        mlflow.log_metric("trainable_params", trainable, step=0)

        global_step, best_score, _ = _train_phase(
            model, train_loader, val_loader, loss_fn, cfg_glob, cfg_method,
            lr=learning_rate, num_epoch=num_epoch,
            phase_idx=0, save_path=save_path, global_step=global_step, best_score=best_score
        )

        _load_best_checkpoint(model, save_path)
        mlflow.log_artifact(str(save_path))

        save_results(model, timestamp, train_loader, val_loader, test_loader,
                     loss_name=cfg_method["loss_name"], cfg_mod=cfg_mod, method_FT=method_FT)
    print(f"End of training by {method_FT}")
    return run.info.run_id, method_FT
