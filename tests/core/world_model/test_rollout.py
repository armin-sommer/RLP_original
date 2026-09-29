from __future__ import annotations

import torch
from torch import nn

from rp1.core.world_model.rollout import rollout_terminal, rollout_traj


class _AdditiveWorldModel(nn.Module):
    """Next latent = last latent + the last action's embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.action_encoder = nn.Identity()
        self.dim = dim

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        return emb + act_emb[..., : self.dim]


def test_rollout_unrolls_the_plan_one_block_per_step() -> None:
    wm = _AdditiveWorldModel(dim=2)
    history = torch.zeros(1, 3, 2)
    actions = torch.zeros(1, 2, 2)
    plan = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]], requires_grad=True)
    trajectory = rollout_traj(wm, history, actions, plan)
    assert trajectory.shape == (1, 3, 2)
    assert torch.equal(trajectory[0], torch.tensor([[1.0, 0.0], [1.0, 2.0], [2.0, 3.0]]))
    assert torch.equal(rollout_terminal(wm, history, actions, plan), trajectory[:, -1])
    (gradient,) = torch.autograd.grad(trajectory[:, -1].sum(), plan)
    assert torch.equal(gradient, torch.ones_like(plan))
