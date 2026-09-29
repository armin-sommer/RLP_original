"""Learning-rate and coefficient schedules."""

from __future__ import annotations

import math


def cosine_interpolate(base: float, final: float | None, step: int, total: int) -> float:
    """Cosine interpolation from ``base`` to ``final`` over ``total`` steps."""
    if final is None or total <= 0:
        return base
    step = min(step, total)
    return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * step / total))


__all__ = ["cosine_interpolate"]
