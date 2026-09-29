"""Typed dataset boundary used by cache, training, and evaluation code."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]
RowBatch = dict[str, Array]


class Dataset(Protocol):
    """The Stable-WM dataset surface consumed here."""

    @property
    def column_names(self) -> Sequence[str]: ...

    def get_col_data(self, name: str) -> Array: ...

    def get_row_data(self, indices: list[int]) -> RowBatch: ...


def episode_index(dataset: Dataset) -> Array:
    """Per-row episode index, flattened.

    Lance serves ``episode_idx`` without listing it in ``column_names``; the
    evaluation h5 files list ``ep_idx`` explicitly.
    """
    names = ("ep_idx", "episode_idx") if "ep_idx" in dataset.column_names else ("episode_idx", "ep_idx")
    for name in names:
        try:
            return np.asarray(dataset.get_col_data(name)).reshape(-1)
        except (KeyError, ValueError, NotImplementedError):
            continue
    raise KeyError("dataset exposes neither 'episode_idx' nor 'ep_idx'")


class RowRange:
    """Rows ``[start, end)`` of a dataset, re-indexed from zero."""

    def __init__(self, dataset: Dataset, start: int, end: int) -> None:
        self.dataset = dataset
        self.start = start
        self.end = end
        self.column_names = dataset.column_names

    def get_col_data(self, name: str) -> Array:
        return np.asarray(self.dataset.get_col_data(name)[self.start : self.end])

    def get_row_data(self, indices: list[int]) -> RowBatch:
        rows = self.dataset.get_row_data([self.start + index for index in indices])
        return {str(key): np.asarray(value) for key, value in rows.items()}


def load_action_stats(path: str | Path) -> tuple[Array, Array]:
    """``(mean, std)`` from a JSON file with ``mean`` and ``std`` lists."""
    with Path(path).open() as file:
        statistics = json.load(file)
    mean = np.asarray(statistics["mean"], dtype=np.float64)
    std = np.asarray(statistics["std"], dtype=np.float64)
    if mean.shape != std.shape:
        raise ValueError(f"action statistics in {path} have mismatched shapes {mean.shape} and {std.shape}")
    if not (std > 1e-5).all():
        raise ValueError(f"action statistics in {path} have a degenerate std: {std.tolist()}")
    return mean, std


__all__ = ["Array", "Dataset", "RowBatch", "RowRange", "episode_index", "load_action_stats"]
