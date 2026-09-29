from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra.errors import InstantiationException
from omegaconf import OmegaConf, open_dict

from rp1.data import LatentCache
from rp1.utils.config import compose_config, dispatch


def _subsample(tmp_path: Path, phases: int) -> LatentCache:
    LatentCache(
        z=torch.arange(20, dtype=torch.float32)[:, None],
        episode_idx=torch.arange(20) // 10,
        step_idx=torch.arange(20) % 10,
    ).save(tmp_path / "fs1.pt")
    cfg = compose_config(
        Path("training/data"),
        "job/subsample_cache",
        [
            f"preparation.inp={tmp_path / 'fs1.pt'}",
            f"preparation.out={tmp_path / 'fs5.pt'}",
            "preparation.frameskip=5",
            f"preparation.phases={phases}",
        ],
    )
    with open_dict(cfg):
        cfg.run = OmegaConf.create({"directory": str(tmp_path)})
    dispatch(cfg)
    return LatentCache.load(str(tmp_path / "fs5.pt"), mmap=False)


def test_every_phase_of_the_stride_becomes_an_episode(tmp_path: Path) -> None:
    cache = _subsample(tmp_path, phases=5)
    assert cache.phase_multiplex == 5
    episodes = cache.episodes()
    assert len(episodes) == 10
    # phase 3 of source episode 1 holds steps 3 and 8 of that episode
    assert cache.z[episodes[8], 0].tolist() == [13.0, 18.0]


def test_phases_beyond_the_stride_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(InstantiationException, match="phases must be in"):
        _subsample(tmp_path, phases=6)
