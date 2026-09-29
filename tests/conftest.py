from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from rp1.data import LatentCache

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "core"


def pytest_configure(config: pytest.Config) -> None:
    # each xdist worker would otherwise start a thread per core, and the workers'
    # spinning thread pools starve each other by an order of magnitude
    workers = os.environ.get("PYTEST_XDIST_WORKER_COUNT")
    if workers:
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // int(workers)))


@pytest.fixture
def cube_world_model() -> Path:
    return ASSETS / "world_model" / "cube_lewm"


@pytest.fixture
def cube_planner() -> Path:
    return ASSETS / "agent" / "cube" / "lewm" / "s0" / "planner.pt"


@dataclass(frozen=True)
class AgentData:
    """Latent caches at one and five primitive steps per row, and their action h5."""

    dense: Path
    blocks: Path
    actions: Path


@pytest.fixture
def agent_data(tmp_path: Path) -> AgentData:
    episodes, steps, dim, frameskip = 3, 60, 192, 5
    generator = torch.Generator().manual_seed(0)
    z = torch.randn(episodes * steps, dim, generator=generator)
    episode = torch.arange(episodes * steps) // steps
    step = torch.arange(episodes * steps) % steps
    LatentCache(z=z, episode_idx=episode, step_idx=step).save(tmp_path / "dense.pt")
    LatentCache(z=z[::frameskip], episode_idx=episode[::frameskip], step_idx=step[::frameskip] // frameskip).save(
        tmp_path / "blocks.pt"
    )
    with h5py.File(tmp_path / "actions.h5", "w") as file:
        file["action"] = np.random.default_rng(0).normal(size=(episodes * steps, 5)).astype(np.float32)
        file["ep_offset"] = np.arange(episodes) * steps
        file["ep_len"] = np.full(episodes, steps)
    return AgentData(tmp_path / "dense.pt", tmp_path / "blocks.pt", tmp_path / "actions.h5")
