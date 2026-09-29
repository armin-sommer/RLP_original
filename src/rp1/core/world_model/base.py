"""Structural contract for the world-model operations the agent uses."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn


class LatentWorldModel(Protocol):
    @property
    def action_encoder(self) -> nn.Module: ...

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor: ...


__all__ = ["LatentWorldModel"]
