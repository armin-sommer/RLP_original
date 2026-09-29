"""The energy rp1 minimises: the value of a plan's imagined end state, shared by training and deployment."""

import torch

from rp1.core.agent.value.temporal import ValueFunction, window_pair

__all__ = ["plan_energy"]


def plan_energy(value: ValueFunction, trajectory: torch.Tensor, goal: torch.Tensor, frames: int) -> torch.Tensor:
    """``V(z_H, z_goal)`` for an imagined ``trajectory`` ``(B, H, D)``, one energy per row.

    A window value (``frames > 1``) scores the last ``frames`` imagined latents
    against the goal tiled as often.
    """
    return value(*window_pair(trajectory, goal, frames))
