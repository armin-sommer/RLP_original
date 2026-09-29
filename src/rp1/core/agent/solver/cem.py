"""Stable-WM's CEM solver with deadline forwarding and solve timing."""

import time
from collections.abc import Sequence
from typing import Any

from stable_worldmodel.solver.cem import CEMSolver as StableCEMSolver

from rp1.utils.logging import logger


class CEMSolver(StableCEMSolver):
    """CEM that forwards deadline metadata to the planning cost."""

    def solve(self, *args: Any, **kwargs: Any) -> Any:
        """Time the upstream solve so every planner reports one latency line."""
        start_time = time.time()
        result = super().solve(*args, **kwargs)
        logger.info(f"CEM solve completed in {time.time() - start_time:.4f} seconds")
        return result

    def set_align_remaining(self, remaining_chunks: Sequence[int] | None) -> None:
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "set_align_remaining"):
            model.set_align_remaining(remaining_chunks)


__all__ = ["CEMSolver"]
