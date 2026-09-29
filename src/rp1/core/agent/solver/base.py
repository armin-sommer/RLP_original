"""Contracts shared by the learned solvers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

from rp1.core.world_model.base import LatentWorldModel


@dataclass(frozen=True)
class PlannerCheckpoint:
    """A trained planner's saved payload and the value it was trained against."""

    payload: Mapping[str, Any]
    value: nn.Module


class EncoderWorldModel(LatentWorldModel, Protocol):
    wants_proprio: bool
    obs_key: str
    goal_key: str

    def encode(self, info: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]: ...


class EnvironmentCost(Protocol):
    def get_cost(self, info: dict[str, Any], actions: torch.Tensor) -> torch.Tensor: ...


def unwrap_encoder(model: Any) -> torch.nn.Module:
    """Peel planning-cost wrappers off ``model`` until an encoder world model appears.

    ``MetricCost`` holds its inner model at ``.base`` and ``LatentGoalCost`` at
    ``.model``; for LeWM/PLDM the two nest.
    """
    candidate: torch.nn.Module = model
    for _ in range(4):
        if callable(getattr(candidate, "encode", None)):
            return candidate
        inner = getattr(candidate, "base", None)
        if inner is None:
            inner = getattr(candidate, "model", None)
        if not isinstance(inner, torch.nn.Module):
            break
        candidate = inner
    if not callable(getattr(candidate, "encode", None)):
        raise TypeError(f"{type(model).__name__} does not wrap an encoder world model")
    return candidate


__all__ = ["EncoderWorldModel", "EnvironmentCost", "PlannerCheckpoint", "unwrap_encoder"]
