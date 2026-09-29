"""Value-metric and planner-cost contracts."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

import torch
from torch import nn

TensorInfo = dict[str, torch.Tensor]


class ValueMetric(Protocol):
    latent_dim: int

    def cost(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor: ...


class PlanningCost(Protocol):
    def parameters(self, *args: Any, **kwargs: Any) -> Iterator[nn.Parameter]: ...

    def criterion(self, info_dict: TensorInfo) -> torch.Tensor: ...

    def get_cost(self, info_dict: TensorInfo, action_candidates: torch.Tensor) -> torch.Tensor: ...


__all__ = ["PlanningCost", "TensorInfo", "ValueMetric"]
