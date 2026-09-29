from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from rp1.core.agent.policy import WorldModelPolicy
from rp1.data.base import Array, RowBatch
from rp1.inference.benchmark import build_policy, sample_tasks
from rp1.utils.config import compose_config


class _Dataset:
    """Episodes of the given lengths, with a TwoRoom-like state and door observation."""

    column_names = ["state", "observation"]

    def __init__(self, lengths: list[int]) -> None:
        self.episode = np.concatenate([np.full(length, index) for index, length in enumerate(lengths)])
        self.step = np.concatenate([np.arange(length) for length in lengths])
        x = np.where(self.step % 2 == 0, 100.0, 124.0)
        self.state = np.stack([x, np.zeros_like(x)], axis=1)
        self.observation = np.tile([0, 0, 0, 0, 112.0, 50.0], (len(self.step), 1))

    def get_col_data(self, name: str) -> Array:
        columns = {"episode_idx": self.episode, "step_idx": self.step, "state": self.state}
        return columns.get(name, self.observation)

    def get_row_data(self, indices: list[int]) -> RowBatch:
        raise NotImplementedError


def _config(**benchmark: object) -> DictConfig:
    defaults = {"num_episodes": 3, "goal_offset_steps": 2, "episode_range": None, "cross_wall": False}
    config = OmegaConf.create({"runtime": {"seed": 0}, "benchmark": {**defaults, **benchmark}})
    assert isinstance(config, DictConfig)
    return config


def test_tasks_leave_room_for_the_goal_inside_the_episode() -> None:
    dataset = _Dataset([4, 6])
    episodes, starts = sample_tasks(_config(num_episodes=5), dataset)
    lengths = {0: 4, 1: 6}
    assert all(start + 2 < lengths[int(episode)] for episode, start in zip(episodes, starts, strict=True))
    assert len(set(zip(episodes.tolist(), starts.tolist(), strict=True))) == 5


def test_tasks_are_seeded_and_restricted_to_the_episode_range() -> None:
    dataset = _Dataset([10, 10, 10])
    first = sample_tasks(_config(num_episodes=2, episode_range="1:3"), dataset)
    second = sample_tasks(_config(num_episodes=2, episode_range="1:3"), dataset)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    assert set(first[0].tolist()) <= {1, 2}


def test_cross_wall_keeps_goals_across_the_wall() -> None:
    dataset = _Dataset([10])
    _, starts = sample_tasks(_config(goal_offset_steps=1, cross_wall=True, wall_center=112.0), dataset)
    assert len(starts) == 3


def test_too_few_starts_is_an_error() -> None:
    with pytest.raises(ValueError, match="valid evaluation starts"):
        sample_tasks(_config(num_episodes=10), _Dataset([4]))


def test_a_planning_policy_builds_from_the_evaluate_config(cube_world_model: Path, cube_planner: Path) -> None:
    cfg = compose_config(
        Path("inference"),
        "evaluate",
        [
            "benchmark=cube_lewm",
            "core/agent/solver=rp1",
            f"core.world_model.checkpoint={cube_world_model}",
            f"core.agent.solver.checkpoint.path={cube_planner}",
        ],
    )
    policy = build_policy(cfg, "cpu", process={}, transform={})
    assert isinstance(policy, WorldModelPolicy)
    assert policy.cfg.plan_len == cfg.planning.horizon * cfg.planning.action_block
