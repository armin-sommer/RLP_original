"""The rp1 planner network: the learned update rule of the plan refinement."""

import torch
from torch import nn

__all__ = ["PlannerNet"]


class PlannerNet(nn.Module):
    """One refinement step: ``plan, dV/dplan, V -> clip(plan + MLP(...), ±action_limit)``.

    The network sees the whole plan, the value's gradient with respect to it and the
    value itself, and predicts a residual update of the plan.
    """

    def __init__(self, horizon: int, action_dim: int, hidden_dim: int, action_limit: float) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.action_limit = action_limit
        plan_size = horizon * action_dim
        self.net = nn.Sequential(
            nn.Linear(2 * plan_size + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, plan_size),
        )

    def forward(self, plan: torch.Tensor, gradient: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size = plan.shape[0]
        inputs = torch.cat(
            [plan.reshape(batch_size, -1), gradient.reshape(batch_size, -1), value.reshape(batch_size, 1)],
            dim=-1,
        )
        update = self.net(inputs).view(batch_size, self.horizon, self.action_dim)
        return (plan + update).clamp(-self.action_limit, self.action_limit)
