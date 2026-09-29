from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer

from rp1.training.harness.checkpointing import save_pretrained
from rp1.utils.logging import logger


class NonFiniteGradientGuard(Callback):
    """Turn a non-finite optimizer update into a no-op and optionally abort."""

    def __init__(self, max_skipped: int | None) -> None:
        super().__init__()
        self.max_skipped = max_skipped
        self.skipped = 0

    def on_before_optimizer_step(self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer) -> None:
        del pl_module
        if not any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for group in optimizer.param_groups
            for parameter in group["params"]
        ):
            return
        optimizer.zero_grad(set_to_none=True)
        self.skipped += 1
        logger.warning(f"Non-finite gradients skipped ({self.skipped}) at step {trainer.global_step}")
        if self.max_skipped is not None and self.skipped > self.max_skipped:
            raise ValueError(f"more than {self.max_skipped} non-finite gradient steps")


class PortableCheckpointCallback(Callback):
    def __init__(
        self,
        run_name: str,
        config: DictConfig | dict[str, object],
        cache_dir: str | Path,
        *,
        epoch_interval: int,
        step_interval: int,
    ) -> None:
        super().__init__()
        self.run_name = run_name
        self.config = config
        self.cache_dir = cache_dir
        self.epoch_interval = epoch_interval
        self.step_interval = step_interval
        self._saved: set[str] = set()

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        del outputs, batch, batch_idx
        step = trainer.global_step
        if trainer.is_global_zero and self.step_interval > 0 and step > 0 and step % self.step_interval == 0:
            model = pl_module.model
            if not isinstance(model, nn.Module):
                raise TypeError("Lightning module model must be torch.nn.Module")
            self._save(model, f"step_{step}")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if (self.epoch_interval > 0 and epoch % self.epoch_interval == 0) or epoch == trainer.max_epochs:
            model = pl_module.model
            if not isinstance(model, nn.Module):
                raise TypeError("Lightning module model must be torch.nn.Module")
            self._save(model, f"epoch_{epoch}")

    def _save(self, model: nn.Module, suffix: str) -> None:
        if suffix in self._saved:
            return
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.config,
            filename=f"weights_{suffix}.pt",
            cache_dir=self.cache_dir,
        )
        self._saved.add(suffix)


__all__ = ["NonFiniteGradientGuard", "PortableCheckpointCallback"]
