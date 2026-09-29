from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hydra.errors import InstantiationException
from omegaconf import OmegaConf, open_dict

from rp1.core.agent.value import build_metric
from rp1.data import LatentCache
from rp1.methods.rp1.train import ActionBlocks, PlanningTasks
from rp1.training.harness.checkpointing import load_planner, save_metric
from rp1.utils.config import compose_config, dispatch


def test_training_writes_a_deployable_planner(agent_data: Any, cube_world_model: Path, tmp_path: Path) -> None:
    data = agent_data
    cfg = compose_config(
        Path("methods/rp1"),
        "train",
        [
            f"training.cache={data.blocks}",
            f"training.cache_td={data.dense}",
            f"training.h5={data.actions}",
            f"training.wm={cube_world_model}",
            "training.steps=2",
            "training.batch=4",
            "training.td_batch=8",
            "training.pretrain=1",
            "training.max_delta=3",
            "training.expand_weight=1.0",
            "training.replay_prob=0.5",
            "runtime.device=cpu",
        ],
    )
    with open_dict(cfg):
        cfg.run = OmegaConf.create({"directory": str(tmp_path), "checkpoints": str(tmp_path / "checkpoints")})
    (tmp_path / "checkpoints").mkdir()
    dispatch(cfg)

    checkpoint = load_planner(str(tmp_path / "checkpoints" / "planner.pt"), None)
    assert checkpoint.payload["iterations"] == cfg.core.agent.planner.iterations
    state = torch.randn(2, 192)
    assert checkpoint.value(state, state).shape == (2,)


def test_action_blocks_of_a_phase_multiplexed_cache_start_at_their_phase(agent_data: Any) -> None:
    blocks = ActionBlocks(str(agent_data.actions), 5, None, phases=5)
    # phase 3 of source episode 1 (episode 1 * 5 + 3), block 2: step 3 + 5 * 2 of that episode
    assert blocks.row(8, 2) == 60 + 13
    assert np.array_equal(blocks(8, 2), blocks.normalized[73:78].reshape(-1))


def test_band_mix_draws_goals_within_the_drawn_band(agent_data: Any) -> None:
    blocks = ActionBlocks(str(agent_data.actions), 5, None, phases=1)
    cache = LatentCache.load(str(agent_data.blocks), mmap=False)
    tasks = PlanningTasks(
        cache, blocks, max_delta=6, band_mix=[1, 6], p_cross=0.0, rng=np.random.default_rng(0), device="cpu"
    )
    offsets = [tasks._offset(remaining=6) for _ in range(2000)]
    assert set(offsets) == set(range(1, 7))
    assert offsets.count(1) > 2000 / 2  # the one-block band alone takes half the draws


def test_a_parameter_free_value_trains_the_planner_frozen(
    agent_data: Any, cube_world_model: Path, tmp_path: Path
) -> None:
    value = save_metric(build_metric("l2", 192, {}), run_name="l2", cache_dir=tmp_path)
    overrides = [
        f"training.cache={agent_data.blocks}",
        f"training.cache_td={agent_data.dense}",
        f"training.h5={agent_data.actions}",
        f"training.wm={cube_world_model}",
        f"training.init_value={value}",
        "training.steps=2",
        "training.batch=4",
        "training.max_delta=3",
        "runtime.device=cpu",
    ]
    cfg = compose_config(Path("methods/rp1"), "train", [*overrides, "training.freeze_critic_frac=0"])
    with open_dict(cfg):
        cfg.run = OmegaConf.create({"directory": str(tmp_path), "checkpoints": str(tmp_path / "checkpoints")})
    dispatch(cfg)
    assert (tmp_path / "checkpoints" / "planner.pt").is_file()

    cfg = compose_config(Path("methods/rp1"), "train", overrides)
    with open_dict(cfg):
        cfg.run = OmegaConf.create({"directory": str(tmp_path), "checkpoints": str(tmp_path / "checkpoints")})
    with pytest.raises(InstantiationException, match="freeze_critic_frac=0"):
        dispatch(cfg)
