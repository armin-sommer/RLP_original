"""Torch device selection."""

import torch


def pick_device(name: str) -> str:
    """``name`` itself, or the best available backend for ``"auto"``."""
    if name and name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


__all__ = ["pick_device"]
