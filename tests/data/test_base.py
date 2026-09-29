from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rp1.data.base import Array, RowBatch, RowRange, episode_index, load_action_stats


class _Dataset:
    def __init__(self, columns: dict[str, Array], listed: list[str]) -> None:
        self.columns = columns
        self.column_names = listed

    def get_col_data(self, name: str) -> Array:
        if name not in self.columns:
            raise KeyError(name)
        return self.columns[name]

    def get_row_data(self, indices: list[int]) -> RowBatch:
        return {name: values[indices] for name, values in self.columns.items()}


def test_episode_index_reads_the_unlisted_lance_column() -> None:
    dataset = _Dataset({"episode_idx": np.array([[0], [0], [1]])}, listed=["pixels"])
    assert episode_index(dataset).tolist() == [0, 0, 1]


def test_episode_index_prefers_a_listed_ep_idx() -> None:
    dataset = _Dataset({"ep_idx": np.array([5, 6]), "episode_idx": np.array([0, 1])}, listed=["ep_idx"])
    assert episode_index(dataset).tolist() == [5, 6]


def test_row_range_reindexes_from_zero() -> None:
    dataset = _Dataset({"value": np.arange(10)}, listed=["value"])
    view = RowRange(dataset, 3, 7)
    assert view.get_col_data("value").tolist() == [3, 4, 5, 6]
    assert view.get_row_data([0, 2])["value"].tolist() == [3, 5]


def test_action_stats_are_validated(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"mean": [0.0, 1.0], "std": [1.0, 2.0]}))
    mean, std = load_action_stats(good)
    assert mean.tolist() == [0.0, 1.0] and std.tolist() == [1.0, 2.0]
    degenerate = tmp_path / "degenerate.json"
    degenerate.write_text(json.dumps({"mean": [0.0], "std": [0.0]}))
    with pytest.raises(ValueError, match="degenerate"):
        load_action_stats(degenerate)
