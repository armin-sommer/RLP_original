"""Offline TD value learning."""

from __future__ import annotations

import torch

from rp1.data import LatentCache
from rp1.training.phases.agent.learners.td import TDConfig, fit


def _cache() -> LatentCache:
    torch.manual_seed(0)
    return LatentCache(
        z=torch.randn(600, 16),
        episode_idx=torch.arange(600, dtype=torch.int64) // 40,
        step_idx=torch.arange(600, dtype=torch.int64) % 40,
    )


def _config(save_every: int = 0) -> TDConfig:
    return TDConfig(
        head="quasimetric",
        symmetric=False,
        hidden_dim=32,
        depth=2,
        embed_dim=16,
        n_step=1,
        gamma=0.98,
        expectile=0.03,
        p_cross=0.3,
        batch_size=64,
        steps=150,
        save_every=save_every,
        seed=0,
        max_delta=6,
        lr=1e-3,
        weight_decay=1e-4,
        tau=0.005,
        balanced=True,
        n_buckets=10,
        huber_beta=1.0,
        num_components=8,
        softplus=True,
        sym_frac=0.5,
        alpha_init=0.75,
        near_frac=0.0,
        near_max=3,
    )


def test_training_is_deterministic_at_a_fixed_seed() -> None:
    cache = _cache()
    a = fit(cache, _config(), "cpu")
    b = fit(cache, _config(), "cpu")
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb), "TD fit is not deterministic at a fixed seed"


def test_snapshots_arrive_every_save_every_steps_before_the_last() -> None:
    steps: list[int] = []
    fit(_cache(), _config(save_every=50), "cpu", lambda head, step: steps.append(step))
    assert steps == [50, 100]
