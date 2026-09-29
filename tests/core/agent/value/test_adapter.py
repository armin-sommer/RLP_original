from __future__ import annotations

import torch
from torch import nn

from rp1.core.agent.value import LatentGoalCost
from rp1.core.agent.value.base import TensorInfo


def test_latent_goal_cost_broadcasts_candidates_and_caches_goal() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encode_calls = 0

        def encode(self, goal: TensorInfo) -> dict[str, torch.Tensor]:
            self.encode_calls += 1
            return {"emb": goal["pixels"].float()}

        def rollout(self, info: TensorInfo, actions: torch.Tensor) -> None:
            batch, candidates = actions.shape[:2]
            info["predicted_emb"] = torch.zeros(batch, candidates, 1, 3)

    model = Model()
    cost = LatentGoalCost(model)
    info = {"goal": torch.ones(2, 1, 3)}
    actions = torch.zeros(2, 4, 5, 2)
    first = cost.get_cost(info, actions)
    second = cost.get_cost(info, actions)
    assert first.shape == (2, 4)
    assert torch.equal(first, torch.full((2, 4), 3.0))
    assert torch.equal(second, first)
    assert model.encode_calls == 1
