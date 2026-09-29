"""Checkpoint I/O for world models, values and planners, in Stable-WM's artifact layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import torch
from omegaconf import OmegaConf
from stable_worldmodel.wm.utils import load_pretrained as _load_pretrained
from stable_worldmodel.wm.utils import save_pretrained as _save_pretrained
from torch import nn

from rp1.core.agent.solver.base import PlannerCheckpoint


def load_pretrained(
    name: str | Path,
    cache_dir: str | None = None,
    extra_args: Any | None = None,
) -> nn.Module:
    """Load a local checkpoint before falling back to Stable-WM resolution.

    Stable-WM 0.1.1 resolves every relative name below its global checkpoint
    cache.  Repository configs intentionally use checkout-relative paths, so
    turn an existing local path into an absolute path before delegating.  A
    missing path is left untouched because it may be a Hugging Face repo ID.
    """
    path = Path(name).expanduser()
    resolved_name = str(path.resolve()) if path.exists() else str(name)
    kwargs: dict[str, Any] = {"extra_args": extra_args}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    return cast(nn.Module, _load_pretrained(resolved_name, **kwargs))


def save_pretrained(model: nn.Module, run_name: str, config: Any | None = None, **kwargs: Any) -> None:
    """Accept both plain mappings and OmegaConf configs.

    Stable-WM 0.1.1 unconditionally calls ``OmegaConf.to_container`` and so
    rejects plain dictionaries; they are converted to an OmegaConf object first.
    """
    if config is not None and not OmegaConf.is_config(config):
        config = OmegaConf.create(config)
    _save_pretrained(model, run_name=run_name, config=config, **kwargs)


def load_wm(name: str, cache_dir: str | None = None, device: str = "cpu") -> nn.Module:
    """Load a frozen world model for planning or caching."""
    wm = load_pretrained(name, cache_dir=cache_dir)
    wm = wm.to(device).eval()
    wm.requires_grad_(False)
    return wm


@runtime_checkable
class PretrainedMetric(Protocol):
    """Inference and serialization contract implemented by metric modules."""

    latent_dim: int

    def cost(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor: ...

    def pretrained_config(self) -> dict[str, object]: ...


def save_metric(module: nn.Module, *, run_name: str, cache_dir: str | Path) -> Path:
    """Save a metric using Stable-WM's weights plus Hydra config layout."""
    if not isinstance(module, PretrainedMetric):
        raise TypeError(f"{type(module).__name__} does not implement the pretrained metric contract")
    cache_dir = Path(cache_dir)
    save_pretrained(module, run_name=run_name, config=module.pretrained_config(), cache_dir=str(cache_dir))
    return cache_dir.resolve() / "checkpoints" / run_name


def load_metric(
    name: str | Path,
    device: str | torch.device = "cpu",
    cache_dir: str | Path | None = None,
) -> nn.Module:
    """Load and validate a Stable-WM metric artifact."""
    path = Path(name).expanduser()
    if path.is_file() and not (path.parent / "config.json").is_file():
        raise ValueError(f"Unsupported metric checkpoint {path}; expected a Stable-WM artifact with config.json")
    if cache_dir is None and path.exists():
        resolved = path.resolve()
        checkpoint_root = next(
            (parent for parent in (resolved, *resolved.parents) if parent.name == "checkpoints"), None
        )
        if checkpoint_root is not None:
            cache_dir = checkpoint_root.parent
    try:
        module = load_pretrained(str(name), cache_dir=None if cache_dir is None else str(cache_dir))
    except Exception as error:
        if path.is_file():
            raise ValueError(
                f"Unsupported metric checkpoint {path}; expected a Stable-WM artifact with config.json"
            ) from error
        raise
    if not isinstance(module, nn.Module) or not callable(getattr(module, "cost", None)):
        raise TypeError(f"checkpoint {name!s} does not contain a value metric")
    return module.to(device).eval()


def load_planner(path: str, value_path: str | None) -> PlannerCheckpoint:
    """Load a learned planner checkpoint and the value it plans against.

    The checkpoint records the value's path; a relative one is resolved against
    the checkpoint's directory. ``value_path`` replaces the recorded one.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"planner checkpoint {path} must contain a mapping")
    value = Path(value_path or str(payload["value"]))
    if not value.is_absolute():
        value = Path(path).resolve().parent / value
    return PlannerCheckpoint(payload=payload, value=load_metric(value))


__all__ = [
    "PretrainedMetric",
    "load_metric",
    "load_planner",
    "load_pretrained",
    "load_wm",
    "save_metric",
    "save_pretrained",
]
