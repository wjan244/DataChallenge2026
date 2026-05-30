import logging
import mlflow
import pandas as pd
import torch
import torch.nn as nn
import time

from pathlib import Path
from tqdm import tqdm
from src.config import*

from src.data.data_utils import get_challenge_split
from src.models.loss import WeightedMSELoss, WeightedLiteMSELoss, UniversalLossWrapper
from src.models.models import get_model

logger = logging.getLogger(__name__)


def run_train(timestamp: str, train_loader, val_loader, cfg_mod, cfg_glob, cfg_method,
              precedent_run_id, precedent_method, prefix: str | None = None,) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    """
    Pipe d'entrainement complet du modèle défini dans config.py:
    - extraire les poids du run_train précédent
    - instancier le modèle avec les poids du modèle base ou ceux du précédent entrainement
    - préparer les données
    - dataAugmentation définie dans src.transforms
    - backpropagation suivant les paramètres de configuration de config.py
    - optimisation du learning rate avec un cosine scheduler
    - sauvegarde des poids/métriques/paramètres en local et sur le Dashboard MLFlow
    """
    # Configure log level from config (default: WARNING) — set verbose: DEBUG in YAML to see all steps
    log_level = getattr(logging, str(cfg_glob.get("VERBOSE", "WARNING")).upper(), logging.WARNING)
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    logger.setLevel(log_level)

    # Loss mapping
    LOSS_MAPPING = {"MSE":nn.MSELoss,"BCE":nn.BCELoss, "nMSE":WeightedMSELoss, "nLiteMSE":WeightedLiteMSELoss}

    # création des dossiers locaux
    CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True)
    HISTORY_DIR.mkdir(parents=True,exist_ok=True)

    # charger les paramètres yaml
    learning_rate = cfg_method["learning_rate"]
    num_epoch = cfg_method["num_epoch"]
    loss_name = cfg_method["loss_name"]
    method_FT = cfg_method["method_FT"]
    patience = cfg_glob["PATIENCE"]
    model_tag = f"{cfg_mod}_{method_FT}"
    logger.debug("[1/7] Config loaded — model=%s  method=%s  lr=%s  epochs=%d  loss=%s",
                 cfg_mod, method_FT, learning_rate, num_epoch, loss_name)

    # # load dataframes
    logger.debug("[2/7] Loading dataset split…")
    _, df_val_raw, df_val_samp, df_test = get_challenge_split()
    logger.debug("[2/7] Split loaded — val_raw=%d  val_samp=%d  test=%d",
                 len(df_val_raw), len(df_val_samp), len(df_test))

    # extraction des poids précédents
    logger.debug("[3/7] Resolving previous weights (precedent_run_id=%s)…", precedent_run_id)
    if precedent_run_id:
        precedent_tag = f"{cfg_mod}_{precedent_method}"
        weights = mlflow.artifacts.download_artifacts(
            run_id=precedent_run_id,
            artifact_path=f"{timestamp}_{precedent_tag}.pt")
        logger.debug("[3/7] Weights downloaded from run %s", precedent_run_id)
    else:
        weights = None
        logger.debug("[3/7] No previous run — starting from pretrained backbone")

    # instancier le modèle
    cfg_method_kwargs = cfg_method.get("method_kwargs") or {}
    logger.debug("[4/7] Building model — method_kwargs=%s", cfg_method_kwargs)
    model = get_model(cfg_mod, num_classes=1, method=method_FT, weights=weights, **cfg_method_kwargs)
    # -> DEVICE
    model = model.to(DEVICE)
    logger.debug("[4/7] Model ready on %s", DEVICE)

    # GD
    base_loss = LOSS_MAPPING[loss_name]()
    loss_fn = UniversalLossWrapper(base_loss)
        # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=num_epoch,eta_min=0,last_epoch=-1)
    logger.debug("[5/7] Loss=%s  optimizer=Adam  scheduler=CosineAnnealingLR(T_max=%d)", loss_name, num_epoch)

    # paramétrisatio MLFlow
        # use get (not pop) so cfg_method stays intact for callers after run_train returns
    # kwargs = cfg_method.pop("method_kwargs") # coupe et colle

    hyper_params = {**cfg_glob, **cfg_method, **cfg_method_kwargs}
    hyper_params.update({
        "model": cfg_mod,
        "model_tag": model_tag,
        "time_stamp": timestamp,
        "prefix": prefix
    })
    mlflow.log_params(hyper_params)
    logger.debug("[5/7] MLflow params logged")

    # entrainement
    save_path = CHECKPOINT_DIR / f"{timestamp}_{model_tag}.pt"
    best_loss = float('inf')

    # initialisation early stopping
    best_loss = float('inf')
    patience_counter = 0
    logger.debug("[6/7] Starting training loop — %d epochs  patience=%d", num_epoch, patience)

    train_start = time.time()
    for n in range(num_epoch):
        print(f"Epoch {n+1}")
        epoch_start = time.time()
        logger.debug("[6/7] Epoch %d/%d — train phase", n + 1, num_epoch)
        model.train()
        running_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Entraînement")

        for batch_idx, batch in progress_bar:
            X = batch[0].to(DEVICE)
            # normalize y to shape [B,1]
            y = batch[1].to(DEVICE).float()
            y = y.squeeze()
            y = y.view(-1, 1)

            # fixer les coefficients par défaut
            iw = None
            pi = None

            # extraction des coefficients en fonction de la loss appelée
            if loss_name == "nLiteMSE":
                    iw = batch[4].to(DEVICE).unsqueeze(1).float()
            elif loss_name == "nMSE":
                iw = batch[4].to(DEVICE).unsqueeze(1).float()
                pi = batch[5].to(DEVICE).unsqueeze(1).float()

            y_pred = model(X)
            loss = loss_fn(y_pred, y, iw, pi)

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = running_loss/len(train_loader)
        logger.debug("[6/7] Epoch %d — train_loss=%.6f", n + 1, final_loss)

        # boucle d'évaluation
        logger.debug("[6/7] Epoch %d/%d — val phase", n + 1, num_epoch)
        model.eval()
        val_loss = 0
        with torch.inference_mode():
            for batch in val_loader:
                X_val = batch[0].to(DEVICE)
                # normalize y_val to shape [B,1]
                y_val = batch[1].to(DEVICE).float()
                y_val = y_val.squeeze()
                y_val = y_val.view(-1, 1)

                # validation: handle possible presence of iw/pi
                iw_val = None
                pi_val = None
                if loss_name == "nLiteMSE":
                    iw_val = batch[4].to(DEVICE).unsqueeze(1).float()
                elif loss_name == "nMSE":
                    iw_val = batch[4].to(DEVICE).unsqueeze(1).float()
                    pi_val = batch[5].to(DEVICE).unsqueeze(1).float()

                y_pred_val = model(X_val)
                loss_v = loss_fn(y_pred_val, y_val, iw_val, pi_val)
                val_loss += loss_v.item()

        final_val_loss = val_loss / len(val_loader)
        logger.debug("[6/7] Epoch %d — val_loss=%.6f", n + 1, final_val_loss)

        # enregistrement des métriques sur MLflow
        mlflow.log_metric(key="lr", value=scheduler.get_last_lr()[0], step=n)
        mlflow.log_metric(key="train_loss", value=final_loss, step=n)
        mlflow.log_metric(key="val_loss", value=final_val_loss, step=n)

        # log the time to run the epoch
        epoch_time = time.time() - epoch_start
        mlflow.log_metric("epoch_time_s", epoch_time, step=n)
        # update du scheduler
        scheduler.step()

        # sauvegarde du modèle en local et mlflow
        if final_val_loss < best_loss:
            best_loss = final_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info("Epoch %d — checkpoint saved  val_loss=%.6f", n + 1, final_val_loss)
            print(f"modèle sauvegardé à l'époque {n+1} - Val Loss: {final_val_loss:.4f}")
        else:
            patience_counter += 1
            logger.debug("Epoch %d — no improvement (%d/%d)", n + 1, patience_counter, patience)

        # sauvegarde loss en local
        log_path = HISTORY_DIR / f"{timestamp}_train_history_loss_{model_tag}.csv"
        row_dict = {**cfg_glob, **cfg_method}

        row_dict.update({
            "id_run": mlflow.active_run().info.run_id,
            "date": timestamp,
            "modèle": cfg_mod,
            "tag": model_tag,
            "epoch": n + 1,
            "final_train_loss": final_loss,
            "final_val_loss": final_val_loss
        })
        new_row = pd.DataFrame([row_dict])

        if log_path.exists():
            new_row.to_csv(log_path, mode='a', header=False, index=False)
        else:
            new_row.to_csv(log_path, index=False)

        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d (patience=%d)", n + 1, patience)
            print(f"stagnation de l'entraînement - arrêt à l'époque {n+1}")
            break

    # sauvegarde des poids sur MLFlow
    logger.debug("[7/7] Training done in %.1fs — saving final artifact", time.time() - train_start)
    model.load_state_dict(torch.load(save_path))
    mlflow.log_artifact(local_path=str(save_path))

    # log the total training time
    mlflow.log_metric("total_train_time_s", time.time() - train_start)
    logger.debug("[7/7] Artifact logged to MLflow")

    # récupérer l'id de run_train (à injeter sur le run_train suivant)
    run_id = mlflow.active_run().info.run_id
    logger.debug("[7/7] run_id=%s", run_id)

    return run_id, df_val_raw, df_val_samp, df_test,log_path

