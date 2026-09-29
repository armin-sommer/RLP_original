from __future__ import annotations

import torch


def test_unwrap_encoder_peels_cost_wrappers() -> None:
    from torch import nn

    from rp1.core.agent.solver.base import unwrap_encoder
    from rp1.core.agent.value.adapter import LatentGoalCost

    class WM(nn.Module):
        def encode(self, info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return info

    wm = WM()
    assert unwrap_encoder(wm) is wm
    assert unwrap_encoder(LatentGoalCost(wm)) is wm

    class OuterCost(nn.Module):  # MetricCost-shaped: inner stack at .base
        def __init__(self, base: nn.Module) -> None:
            super().__init__()
            self.base = base

    assert unwrap_encoder(OuterCost(LatentGoalCost(wm))) is wm
