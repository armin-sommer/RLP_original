"""Stable-WM's gradient solver with device-safe warm starts and solve timing."""

from __future__ import annotations

import time
from typing import Any

import torch
from stable_worldmodel.solver import GradientSolver as _GradientSolver

from rp1.utils.logging import logger


class GradientSolver(_GradientSolver):
    """Move full-horizon warm starts to the solver device before noising."""

    def solve(self, *args: Any, **kwargs: Any) -> Any:
        """Time the upstream solve so every planner reports one latency line."""
        start_time = time.time()
        result = super().solve(*args, **kwargs)
        logger.info(f"Adam solve completed in {time.time() - start_time:.4f} seconds")
        return result

    def init_action(self, n_envs: int, actions: torch.Tensor | None = None) -> None:
        # Stable-WM 0.1.1 moves ``actions`` to the solver device only when it
        # appends a tail, so a full-horizon warm start stays on the caller's device.
        if actions is not None:
            actions = actions.to(self.device)
        super().init_action(n_envs, actions)


__all__ = ["GradientSolver"]
