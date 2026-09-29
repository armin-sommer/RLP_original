"""Horizon-matched temporal regression.

The pairwise head learns the temporal separation of two states on the same
logged trajectory::

    min_phi  E_{(i,j)}  Huber( m_phi(z_i, z_j),  |t_i - t_j| / s )

with balanced full-horizon pairs from
:class:`~rp1.training.phases.agent.samplers.BalancedHorizonPairSampler`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rp1.core.agent.value.head import PairwiseMetricHead
from rp1.data import LatentCache
from rp1.training.phases.agent.samplers import BalancedHorizonPairSampler
from rp1.utils.logging import logger


@dataclass
class RegressionConfig:
    hidden_dim: int
    depth: int
    softplus: bool
    symmetric: bool
    scale: float
    lr: float
    weight_decay: float
    batch_size: int
    steps: int
    n_buckets: int
    max_delta: int | None
    shuffle_labels: bool
    seed: int
    huber_beta: float


def fit(cache: LatentCache, cfg: RegressionConfig, device: str) -> PairwiseMetricHead:
    torch.manual_seed(cfg.seed)
    head = PairwiseMetricHead(
        cache.latent_dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        softplus=cfg.softplus,
        symmetric=cfg.symmetric,
        scale=cfg.scale,
    ).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sampler = BalancedHorizonPairSampler(
        cache,
        n_buckets=cfg.n_buckets,
        max_delta=cfg.max_delta,
        random_order=True,
        seed=cfg.seed,
    )
    rng = torch.Generator().manual_seed(cfg.seed + 1)

    head.train()
    log_interval = max(1, cfg.steps // 20)
    for step in range(cfg.steps):
        batch = sampler.sample(cfg.batch_size)
        z_i = batch["z_i"].to(device)
        z_j = batch["z_j"].to(device)
        y = (batch["label"] / cfg.scale).to(device)
        if cfg.shuffle_labels:  # the negative control
            perm = torch.randperm(y.shape[0], generator=rng)
            y = y[perm]
        pred = head(z_i, z_j)
        loss = F.smooth_l1_loss(pred, y, beta=cfg.huber_beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch 2.7 Tensor.backward lacks a typed signature.
        opt.step()
        if step == 0 or (step + 1) % log_interval == 0 or step + 1 == cfg.steps:
            logger.info(f"Regression training step={step + 1}/{cfg.steps} loss={loss.item():.6f}")
    logger.success(f"Regression head trained ({cfg.steps} steps), final loss={loss.item():.4f}")
    head.eval()
    return head


__all__ = ["RegressionConfig", "fit"]
