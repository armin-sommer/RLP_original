"""The planning cost of Stable-WM 0.1.1's LeWM and PLDM, with its goal broadcasting fixed."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from rp1.core.agent.value.base import PlanningCost, TensorInfo


class LatentGoalCost(nn.Module):
    """Expose a corrected terminal latent-MSE cost around a Stable-WM model.

    The 0.1.1 LeWM/PLDM criterion treats ``goal_emb`` as if it already had a
    candidate dimension. In the actual policy path it has shape ``(B,T,D)``;
    imagined states have shape ``(B,S,T,D)``. This adapter owns that broadcast
    and caches the encoded goal across repeated CEM iterations.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def parameters(self, *args: Any, **kwargs: Any) -> Iterator[nn.Parameter]:
        return self.model.parameters(*args, **kwargs)

    def criterion(self, info_dict: TensorInfo) -> torch.Tensor:
        pred = info_dict["predicted_emb"]
        goal = info_dict["goal_emb"]
        if goal.ndim == pred.ndim:
            goal = goal[:, 0]
        if goal.ndim == 2:
            goal = goal[:, None, :]
        goal = goal[:, None, -1:, :].expand_as(pred)
        return F.mse_loss(
            pred[..., -1:, :],
            goal[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred.ndim)))

    def get_cost(self, info_dict: TensorInfo, action_candidates: torch.Tensor) -> torch.Tensor:
        if "goal_emb" not in info_dict:
            if "goal" not in info_dict:
                raise KeyError("planning info lacks 'goal'")
            goal = {
                key: value[:, 0]
                for key, value in info_dict.items()
                if torch.is_tensor(value) and not key.startswith("_")
            }
            goal["pixels"] = goal["goal"]
            for key in tuple(goal):
                if key.startswith("goal_"):
                    goal[key.removeprefix("goal_")] = goal.pop(key)
            goal.pop("action", None)
            info_dict["goal_emb"] = cast(Any, self.model).encode(goal)["emb"]
        cast(Any, self.model).rollout(info_dict, action_candidates)
        return self.criterion(info_dict)


def as_planning_cost(model: nn.Module) -> PlanningCost:
    """Wrap Stable-WM's LeWM and PLDM; return any other planning cost unchanged."""
    module = type(model).__module__
    name = type(model).__name__
    if module.startswith("stable_worldmodel.wm.") and name in {"LeWM", "PLDM"}:
        return cast(PlanningCost, LatentGoalCost(model))
    if not callable(getattr(model, "get_cost", None)) or not callable(getattr(model, "criterion", None)):
        raise TypeError(f"{type(model).__name__} does not implement the planning-cost contract")
    return cast(PlanningCost, model)


__all__ = ["LatentGoalCost", "as_planning_cost"]
