"""Lightning experiment logging configured from the shared run context."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.loggers.logger import Logger
from omegaconf import DictConfig, OmegaConf


def make_logger(cfg: DictConfig) -> Logger:
    tracking = Path(cfg.run.tracking)
    mode = cfg.logging.wandb.mode
    if mode == "disabled":
        return CSVLogger(save_dir=str(tracking), name="csv")

    raw_kwargs = OmegaConf.to_container(cfg.logging.wandb.config, resolve=True)
    if not isinstance(raw_kwargs, Mapping):
        raise TypeError("logging.wandb.config must resolve to a mapping")
    kwargs: dict[str, Any] = {str(key): value for key, value in raw_kwargs.items() if value is not None}
    kwargs.setdefault("name", cfg.output.model_name)
    kwargs.setdefault("id", Path(cfg.run.directory).name)
    return WandbLogger(
        save_dir=str(tracking),
        offline=mode == "offline",
        **kwargs,
    )


__all__ = ["make_logger"]
