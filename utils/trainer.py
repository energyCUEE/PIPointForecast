import gc
import os
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from wandb import Run as WandbRun

from utils.dataset import (
    CachedPIRawBatchDataset,
)
from utils.evalmetrics import EvaluationMetrics
from utils.formulations import PIloss_split, Regressionloss

Split = Literal["train", "val", "test"]

METRICS = [
    ("loss_total", "min"),
    ("loss_picp_avg", "min"),
    ("loss_pi_avg", "min"),
    ("loss_point_avg", "min"),
    ("pinaw_avg", "min"),
    ("picp_avg", "max"),
]


class MGDA_PointPI_Trainer:
    model: nn.Module
    loaders: Dict[Split, DataLoader]
    pi_loss: PIloss_split
    reg_loss: Regressionloss
    metrics: EvaluationMetrics
    device: torch.device

    def __init__(
        self,
        *,
        wandb_run: Optional[WandbRun] = None,
        dbg_mode=False,
        log_step=10,
        accumulation_steps=1,
        num_lag: int | None = None,
        set_patience: int = 200,
        model_save_path: str = "./save_model",
        model_save_name: str = "forecast-model",
        num_epochs: int = 2000,
        min_save_epoch: int = 10,
    ):

        self.wandb = wandb_run
        self.dbg_mode = dbg_mode
        self.step = 0
        self.epoch = 0
        self.log_step = log_step
        self.accumulation_steps = accumulation_steps
        self._num_lag = num_lag  # None = use all cached lag steps

        self.set_patience = set_patience
        self.model_save_path = model_save_path
        self.model_save_name = model_save_name
        self.num_epochs = num_epochs
        self.min_save_epoch = min_save_epoch

        self.save_dir = os.path.join(self.model_save_path, self.model_save_name)

        if self.wandb is not None:
            logger.debug("Wandb is enabled for Chronos MGDA PointPI Trainer.")
            for metric_name, goal in METRICS:
                self.wandb.define_metric(f"train/{metric_name}", step_metric="epoch", summary=goal)
                self.wandb.define_metric(f"val/{metric_name}", step_metric="epoch", summary=goal)

            self.wandb.define_metric("train_step/pi_loss", summary="min")
            self.wandb.define_metric("train_step/reg_loss", summary="min")
            self.wandb.define_metric("train_step/weighted_loss", summary="min")

            logger.debug(
                f"Logged every {self.log_step} steps with accumulation steps {self.accumulation_steps}."
            )

    def eval(self):
        self.model.eval()

    def _init_dataloaders(
        self,
        datasets: Dict[Split, Dataset] | Literal["cache_piraw"],
    ):
        if datasets == "cache_piraw":
            datasets = {
                split: CachedPIRawBatchDataset(
                    cache_dir=f"data/raw/{split}/b32768",
                    batch_size=self.batch_size,
                    num_lag=self._num_lag,
                    shuffle=(split == "train"),
                    chronos_future=self.cfg.run.chronos_future,
                )
                for split in ["train", "val"]
            }
        elif isinstance(datasets, dict):
            pass
        else:
            raise ValueError("datasets must be a dict or 'cache'.")

        self.loaders = {
            split: DataLoader(
                dataset=datasets[split],
                batch_size=None,
                pin_memory=False,
                num_workers=0,
                shuffle=(split == "train"),
            )
            for split in ["train", "val"]
        }
        return datasets

    def gamma_calculator(self, grads_alltask):  # For two task only
        """
        Compute the interpolation coefficient gamma between two gradient vectors.

        Parameters
        ----------
        grads_alltask : list of torch.Tensor
            A list containing two gradient tensors:
            - grads_alltask[0] = gradient from task 1
            - grads_alltask[1] = gradient from task 2

        Returns
        -------
        gamma : float
            Weight for task 1's gradient in the interpolation.
        1 - gamma : float
            Weight for task 2's gradient in the interpolation.
        """

        # Extract gradients from the list
        grads1 = grads_alltask[0]
        grads2 = grads_alltask[1]

        # --- Compute gamma ---
        # Formula: gamma = <(g2 - g1), g2> / ||g1 - g2||^2
        #   - Numerator: dot product between (g2 - g1) and g2
        #   - Denominator: squared L2 distance between g1 and g2
        #   - This ensures gamma is based on relative alignment of gradients
        gamma = torch.dot(grads2 - grads1, grads2) / (torch.norm(grads1 - grads2, p=2) ** 2 + 1e-12)

        # Clip gamma to the range [0, 1]
        gamma = torch.clip(gamma, 0.0, 1.0)

        # Return gamma (task 1 weight) and 1 - gamma (task 2 weight) as scalars
        return gamma.item(), (1 - gamma).item()

    @logger.catch(message="Error occurred during training.", level="ERROR", reraise=True)
    def training(
        self,
        datasets: Dict[Split, Dataset] | Literal["cache_piraw"],
        y_scaler: StandardScaler,
        pi_loss: PIloss_split,
        reg_loss: Regressionloss,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        norm_mgda_gradients: bool,
        hydra_config: Optional[DictConfig] = None,
    ):
        self.cfg = hydra_config
        logger.debug("Start training in Chronos MGDA PointPI Trainer.")

        self.device = next(model.parameters()).device
        logger.info(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.pi_loss = pi_loss
        self.reg_loss = reg_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.y_scaler = y_scaler
        self.norm_mgda_gradients = norm_mgda_gradients

        datasets = self._init_dataloaders(datasets)

        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model has {num_trainable_params:,} trainable parameters.")

        self.metrics = EvaluationMetrics()
        desired_picp = 1 - pi_loss.delta
        num_ahead = datasets["train"].num_ahead

        loss_config = {
            pi_loss.__class__: pi_loss.get_config(),
            reg_loss.__class__: reg_loss.get_config(),
        }

        best_model_checkpoint = {
            "loss_config": loss_config,
            "optimizer": None,
            "scheduler": None,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "set_patience": self.set_patience,
            "num_ahead": num_ahead,
        }

        best_val_loss = float("inf")

        dev_picp = np.full(num_ahead, None)

        if self.step != 0 or self.epoch != 0:
            logger.info(f"Resuming training from epoch {self.epoch} and step {self.step}.")

        logger.info("Beginning training loop...")
        logger.info(f"Total epochs to train: {self.num_epochs}.")
        logger.info(
            f"Train — total samples: {len(datasets['train']):,}. batches per epoch: {len(self.loaders['train']):,}."
        )
        logger.info(
            f"Validate — total samples: {len(datasets['val']):,}. batches per epoch: {len(self.loaders['val']):,}."
        )

        for epoch in range(1, self.num_epochs + 1):
            self.epoch = epoch
            t = self._train_epoch(dev_picp)
            _, val_loss, dev_picp = self._evaluation_epoch(desired_picp, t)
            self.scheduler.step()

            if epoch < self.min_save_epoch:
                continue

            if val_loss < best_val_loss:
                logger.success(f"New best model found at epoch {epoch} val_loss={val_loss:.6f}")
                best_val_loss = val_loss
                best_epoch = epoch

                best_model_checkpoint.update(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "best_val_loss": best_val_loss,
                        "best_epoch": best_epoch,
                    }
                )
                os.makedirs(self.save_dir, exist_ok=True)
                save_path = os.path.join(self.save_dir, "best_ckpt.pth")
                model_class_path = os.path.join(self.save_dir, "model_class.pkl")

                if hydra_config is not None:
                    OmegaConf.save(
                        hydra_config, os.path.join(self.save_dir, "config.yaml"), resolve=True
                    )

                model_class = self.model.__class__
                torch.save(model_class, model_class_path)

                torch.save(best_model_checkpoint, save_path)
                patience = self.set_patience  # reset patience
            else:
                patience -= 1

            if patience == 0:
                logger.info(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")

                if self.wandb is not None:
                    self.wandb.alert(
                        title="PI stop Training Stopped Early",
                        text=f"Training stopped early at epoch {epoch}. Best epoch was {best_epoch}.",
                    )

                print("-----Train----")
                self.report_model("train")
                print("-----Val----")
                self.report_model("val")
                break

        message = f"Training Finished. Best epoch was {best_epoch} with validation loss {best_val_loss:.6f}."
        logger.success(message)
        if self.wandb is not None:
            self.wandb.alert(title="PI Training Finished", text=message)

        model.load_state_dict(best_model_checkpoint["state_dict"])
        self.model = model.to(self.device)

        return self.model

    @logger.catch(
        message="Error occurred during a training step.",
        level="ERROR",
        reraise=True,
    )
    def _train_epoch(self, dev_picp: np.ndarray):
        self.model.train()

        self.pi_loss.dev_picp = dev_picp
        self.pi_loss.return_seploss = False

        t = self.pi_loss.update_t(dev_picp)

        cur_epoch_step = 0
        for X_batch, y_batch in tqdm(
            self.loaders["train"],
            leave=False,
            unit="batch",
            desc=f"Train {self.epoch}/{self.num_epochs}",
        ):
            self.step += 1
            cur_epoch_step += 1
            self._train_batch(X_batch, y_batch, cur_epoch_step)

        return t

    @logger.catch(message="Error occurred during a training batch.", level="ERROR", reraise=True)
    def _train_batch(
        self,
        X_batch: Tuple[torch.Tensor, torch.Tensor],
        y_batch: torch.Tensor,
        cur_epoch_step: int,
    ):
        # Auto-cast to match model dtype
        dtype = next(self.model.parameters()).dtype

        X_batch = (X_batch[0].to(self.device).to(dtype), X_batch[1].to(self.device).to(dtype))

        y_batch = y_batch.to(self.device).to(dtype)
        pi_batch, y_hat_batch = self.model(X_batch)
        pi_batch, y_hat_batch = pi_batch.to(self.device), y_hat_batch.to(self.device)

        # Compute per-task mean losses directly
        loss_pi_mean = self.pi_loss(y_batch, pi_batch).mean()
        loss_reg_mean = self.reg_loss(y_batch, y_hat_batch).mean()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        scaled_loss1 = loss_pi_mean
        scaled_loss2 = loss_reg_mean

        grads1 = torch.autograd.grad(
            scaled_loss1, trainable_params, retain_graph=True, allow_unused=True
        )
        grads2 = torch.autograd.grad(
            scaled_loss2, trainable_params, retain_graph=False, allow_unused=True
        )
        # set retain_graph=True for the first backward pass to allow computing multiple gradients from the same graph, then False for the second to free it immediately after

        # Extract scalar values for logging before deleting graph-connected tensors
        log_pi_loss = loss_pi_mean.item()
        log_reg_loss = loss_reg_mean.item()
        del loss_pi_mean, loss_reg_mean, scaled_loss1, scaled_loss2
        del pi_batch, y_hat_batch, y_batch  # free forward-pass activations

        with torch.no_grad():
            g1_flat = torch.cat([g.flatten() for g in grads1 if g is not None])
            g2_flat = torch.cat([g.flatten() for g in grads2 if g is not None])

            g1_norm_val = torch.linalg.vector_norm(g1_flat) + 1e-12
            g2_norm_val = torch.linalg.vector_norm(g2_flat) + 1e-12

            g1_normalized = g1_flat / g1_norm_val
            g2_normalized = g2_flat / g2_norm_val

            if self.norm_mgda_gradients:
                gamma1, gamma2 = self.gamma_calculator([g1_normalized, g2_normalized])
            else:
                gamma1, gamma2 = self.gamma_calculator([g1_flat, g2_flat])

            # Compute cosine similarity here while vectors are available
            cos_sim = torch.dot(g1_normalized, g2_normalized).item()
            weighted_loss_val = gamma1 * log_pi_loss + gamma2 * log_reg_loss
            g1_norm_scalar = g1_norm_val.item()
            g2_norm_scalar = g2_norm_val.item()

            # Free large flattened gradient vectors
            del g1_flat, g2_flat, g1_normalized, g2_normalized, g1_norm_val, g2_norm_val

        # Manually assign the combined gradients
        for param, g1, g2 in zip(trainable_params, grads1, grads2):
            combined_grad = torch.zeros_like(param)
            if g1 is not None:
                combined_grad += gamma1 * g1
            if g2 is not None:
                combined_grad += gamma2 * g2

            # Divide by accumulation steps
            combined_grad = combined_grad / self.accumulation_steps

            if param.grad is None:
                param.grad = combined_grad
            else:
                param.grad += combined_grad

        del grads1, grads2

        # Only step optimizer every accumulation_steps
        if self.step % self.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.wandb is not None:
            if self.step % (self.log_step * self.accumulation_steps) != 0:
                return
            self.wandb.log(
                {
                    "train_step/pi_loss": log_pi_loss,
                    "train_step/reg_loss": log_reg_loss,
                    "train_step/weighted_loss": weighted_loss_val,
                    "train_step/gamma1": gamma1,
                    "train_step/g1_norm": g1_norm_scalar,
                    "train_step/gamma2": gamma2,
                    "train_step/g2_norm": g2_norm_scalar,
                    "train_step/lr": self.optimizer.param_groups[0]["lr"],
                    "train_step/grad_cosim": cos_sim,
                },
                step=self.step // self.accumulation_steps,
            )

    @torch.no_grad()
    def inference_split(
        self, split: Split, device=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if device is None:
            device = self.device

        if self.loaders.get(split) is None:
            raise ValueError(
                f"DataLoader for split '{split}' not found. Please call `training` first."
            )

        pi_batches = []
        y_hat_batches = []
        y_batches = []

        self.model.eval()
        model_device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        loader = self.loaders[split]

        for X_batch, y_batch in tqdm(
            loader,
            leave=False,
            unit="batch",
            desc=f"Infer {split}, Epoch {self.epoch}/{self.num_epochs}",
        ):
            X_batch = (X_batch[0].to(model_device).to(dtype), X_batch[1].to(model_device).to(dtype))

            pi_batch, y_hat_batch = self.model(X_batch)

            pi_batches.append(pi_batch.detach().cpu())
            y_hat_batches.append(y_hat_batch.detach().cpu())
            y_batches.append(y_batch.detach().cpu())

        pi_all = torch.cat(pi_batches, dim=0)
        yhat_all = torch.cat(y_hat_batches, dim=0)
        y_all = torch.cat(y_batches, dim=0)

        del pi_batches, y_hat_batches, y_batches
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return pi_all, yhat_all, y_all

    @torch.no_grad()
    @logger.catch(message="Error occurred during evaluation.", level="ERROR", reraise=True)
    def _evaluation_epoch(
        self,
        desired_picp: float,
        t: torch.Tensor,
    ):
        deviceback = torch.device("cpu")
        model, pi_loss, reg_loss = self.model, self.pi_loss, self.reg_loss
        metrics = self.metrics

        model.eval()

        pi_train, yhat_train, y_train = self.inference_split("train", deviceback)

        loss_pi_train_tensor = pi_loss(y_train, pi_train)
        loss_point_train_tensor = reg_loss(y_train, yhat_train)

        # Compute scalars directly — avoid allocating intermediate eachstep array
        loss_pi_train_avg = loss_pi_train_tensor.mean().item()
        loss_point_train_avg = loss_point_train_tensor.mean().item()
        loss_train = loss_pi_train_avg + loss_point_train_avg

        pi_loss.return_seploss = True
        _, _, loss_picp_train_tensor = pi_loss(y_train, pi_train)
        loss_picp_train_avg = loss_picp_train_tensor.mean().item()
        pi_loss.return_seploss = False
        del loss_pi_train_tensor, loss_point_train_tensor, loss_picp_train_tensor

        upper_train = pi_train[:, 1::2]
        lower_train = pi_train[:, 0::2]

        picp_train = metrics.PICP(y_train, upper_train, lower_train)
        pinaw_train = metrics.PINAW(upper_train, lower_train).astype(np.float64)

        dev_picp = desired_picp - picp_train  # (1-delta) - PICP

        picp_train_avg = picp_train.mean().item()
        pinaw_train_avg = pinaw_train.mean().item()

        del pi_train, yhat_train, y_train, upper_train, lower_train
        del picp_train, pinaw_train
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # --- Validation ---
        pi_val, yhat_val, y_val = self.inference_split("val", deviceback)

        loss_pi_val_tensor = pi_loss(y_val, pi_val)
        loss_point_val_tensor = reg_loss(y_val, yhat_val)

        loss_pi_val_avg = loss_pi_val_tensor.mean().item()
        loss_point_val_avg = loss_point_val_tensor.mean().item()
        loss_val = loss_pi_val_avg + loss_point_val_avg

        pi_loss.return_seploss = True
        _, _, loss_picp_val_tensor = pi_loss(y_val, pi_val)
        loss_picp_val_avg = loss_picp_val_tensor.mean().item()
        pi_loss.return_seploss = False
        del loss_pi_val_tensor, loss_point_val_tensor, loss_picp_val_tensor

        upper_val = pi_val[:, 1::2]
        lower_val = pi_val[:, 0::2]

        picp_val = metrics.PICP(y_val, upper_val, lower_val)
        pinaw_val = metrics.PINAW(upper_val, lower_val).astype(np.float64)

        picp_val_avg = picp_val.mean().item()
        pinaw_val_avg = pinaw_val.mean().item()

        del pi_val, yhat_val, y_val, upper_val, lower_val
        del picp_val, pinaw_val

        if self.wandb is not None:
            self.wandb.log(
                {
                    # Train
                    "epoch": self.epoch,
                    "train/loss_total": loss_train,
                    "train/loss_picp_avg": loss_picp_train_avg,
                    "train/loss_pi_avg": loss_pi_train_avg,
                    "train/loss_point_avg": loss_point_train_avg,
                    "train/pinaw_avg": pinaw_train_avg,
                    "train/picp_avg": picp_train_avg,
                    # Validate
                    "val/loss_total": loss_val,
                    "val/loss_picp_avg": loss_picp_val_avg,
                    "val/loss_pi_avg": loss_pi_val_avg,
                    "val/loss_point_avg": loss_point_val_avg,
                    "val/pinaw_avg": pinaw_val_avg,
                    "val/picp_avg": picp_val_avg,
                },
                step=self.step // self.accumulation_steps,
            )

        return loss_train, loss_val, dev_picp

    @torch.no_grad()
    def report_model(self, split: Split):
        pi_all, yhat_all, y_all = self.inference_split(split)
        upper = pi_all[:, 1::2]
        lower = pi_all[:, 0::2]

        upper = upper.detach().cpu().numpy()
        lower = lower.detach().cpu().numpy()
        yhat_all = yhat_all.detach().cpu().numpy()
        y_all = y_all.detach().cpu().numpy()
        self.show_report(y_all, upper, lower, yhat_all, split)

    def show_report(self, y_all, upper, lower, yhat_all, split: Split):
        yhat_all = self.y_scaler.inverse_transform(yhat_all)
        y_all = self.y_scaler.inverse_transform(y_all)
        upper = self.y_scaler.inverse_transform(upper)
        lower = self.y_scaler.inverse_transform(lower)

        eval_all = self.metrics.evaluate_all(
            y_all,
            upper,
            lower,
            yhat=yhat_all,
            ytarget=y_all,
            normalize=False,
            delta=0.1,
            quantile=0.5,
        )

        self.metrics.report_performance(eval_all)
