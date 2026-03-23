import os
import pickle
from datetime import datetime

import hydra
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from utils.helper import get_scaler, setup
from utils.loss import get_loss_functions
from utils.lr_warmup import get_lr_scheduler_with_warmup
from utils.trainer import MGDA_PointPI_Trainer

setup()  # Load env and lock random seeds


def load_data(path: str | os.PathLike):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def get_run_string(cfg: DictConfig):
    model_choice = HydraConfig.get().runtime.choices.model
    comment = f"({cfg.comment})" if cfg.comment != "" else ""
    set_up = f"{model_choice}{comment}"
    run_name = set_up + "@" + datetime.now().strftime("%Y-%m-%d_%H:%M")
    return set_up, run_name


# I - target
# CI_CR, CI_CM - lag covariates
# HI, Iclr, Icams - future regressors


@hydra.main(config_path="config", config_name="conf", version_base=None)
def main(cfg: DictConfig):  # TODO: Add typing to DictConfig later
    data = load_data(cfg.data_path)

    set_up, run_name = get_run_string(cfg)  # For logging in WandB
    model_save_name = (
        set_up.replace(" ", "_")
        .replace("(", "[")
        .replace(")", "]")
        .replace("|", "_")
        .replace("=", "-")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_step_ahead = data.get("num_step_ahead", 16)
    target_col_name = data.get("target_col_name", "I")

    # Scalers
    y_scaler = get_scaler(
        data["df_train_nonan"], [target_col_name], save_path=cfg.model_save_path, save_name="target"
    )

    # Gradient Accumulation Calculation
    target_batch_size = 32_768
    current_batch_size = cfg.run.batch_size
    accumulation_steps = max(1, target_batch_size // current_batch_size)
    print(
        f"Gradient Accumulation enabled: {accumulation_steps} steps (Target Batch: {target_batch_size}, Physical: {current_batch_size})"
    )

    # Instantiate model, optimizer, and scheduler using Hydra configs
    model: nn.Module = instantiate(cfg.model)
    optimizer: Optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler: LRScheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    model.to(device)

    if cfg.run.lr_warmup:
        scheduler = get_lr_scheduler_with_warmup(
            lr_scheduler=scheduler, optimizer=optimizer, warmup_steps=cfg.run.lr_warmup_epochs
        )

    # Loss functions with IRR thresholding
    desired_picp, irr_threshold = 0.9, 10
    threshold = y_scaler.transform(np.full((1, num_step_ahead), irr_threshold))[0, 0] + 1e-8
    print(f"Using IRR threshold (normalized): {threshold:.6f}")
    pi_loss, reg_loss = get_loss_functions(threshold, desired_picp)

    if wandb_key := os.environ.get("WANDB_API_KEY"):
        run_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb.login(key=wandb_key)
        run = wandb.init(project="pi_chronos", name=run_name, config=run_cfg)
    else:
        run = None

    try:
        trainer: MGDA_PointPI_Trainer = instantiate(
            cfg.trainer,
            wandb_run=run,
            model_save_name=model_save_name,
            accumulation_steps=accumulation_steps,
        )

        model = trainer.training(
            datasets="cache_piraw",
            y_scaler=y_scaler,
            pi_loss=pi_loss,
            reg_loss=reg_loss,
            optimizer=optimizer,
            scheduler=scheduler,
            model=model,
            norm_mgda_gradients=cfg.run.norm_grad,
            hydra_config=cfg,
        ).to(device)

        # Evaluation Metrics for best model (lowest val loss)
        for split in ["train", "val"]:
            print(f"Evaluating on {split} set...")
            trainer.report_model(split)
    except Exception as e:
        if run:
            run.alert(title="PI Training Failed", text=f"Training failed with error: {str(e)}")
        raise e
    finally:
        if run:
            run.finish()


if __name__ == "__main__":
    main()
