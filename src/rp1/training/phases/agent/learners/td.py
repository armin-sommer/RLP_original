"""Offline TD learning of a goal-conditioned reachability (quasi)metric.

A goal-conditioned temporal-distance value ``d(z, z_g)``, learned by n-step
distance TD with hindsight goals (balanced full-horizon and cross-episode),
optionally on a quasimetric head so long cross-room distances stitch:

    n-step target (distance):
        reached within n_eff steps  ->  target = δ            (Monte-Carlo, exact)
        else                        ->  target = c(n_eff) + gamma^n_eff * d_target(z_{t+n}, z_g)
        with c(n_eff) = n_eff (gamma=1) or (1-gamma^n_eff)/(1-gamma)

    min  expectile_Huber( d(z_t, z_g) - stop_grad(target) )

n→∞ recovers Monte-Carlo (= the paper's regression target); n=1 is pure bootstrap.
gamma approaching 1 learns true undiscounted steps-to-go; gamma < 1 discounts long range.
The planner terminal cost is ``d(z_pred, z_goal)`` (lower == closer).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch

from rp1.core.agent.value.head import IQEHead, PairwiseMetricHead, QuasimetricHead
from rp1.data import LatentCache
from rp1.training.phases.agent.samplers import NStepGoalSampler
from rp1.utils.logging import logger


@dataclass
class TDConfig:
    head: str
    symmetric: bool
    hidden_dim: int
    depth: int
    embed_dim: int
    n_step: int
    gamma: float
    lr: float
    weight_decay: float
    batch_size: int
    steps: int
    save_every: int
    tau: float
    expectile: float
    p_cross: float
    balanced: bool
    max_delta: int | None
    n_buckets: int
    seed: int
    huber_beta: float
    num_components: int
    softplus: bool
    sym_frac: float
    alpha_init: float
    near_frac: float
    near_max: int


MetricHead = IQEHead | PairwiseMetricHead | QuasimetricHead

# the sampler outputs a TD step reads
TD_KEYS = ("z_t", "z_tn", "z_g", "n_eff", "reached", "dist")


def expectile_loss(diff: torch.Tensor, expectile: float, beta: float) -> torch.Tensor:
    """Expectile-weighted Huber loss."""
    huber = torch.nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff), beta=beta, reduction="none")
    weight = torch.where(diff > 0, 1.0 - expectile, expectile)  # diff=pred-target
    return (weight * huber).mean()


def n_step_target(sample: dict[str, torch.Tensor], bootstrap: torch.Tensor, gamma: float) -> torch.Tensor:
    """The n-step TD target of a sampled batch (the module docstring's formula).

    ``sample`` holds the sampler's ``n_eff``, ``reached`` and ``dist``; ``bootstrap``
    is the target network's ``d(z_{t+n}, z_g)``.
    """
    steps, reached, distance = sample["n_eff"], sample["reached"], sample["dist"]
    if gamma >= 1.0:
        cost, discount = steps, torch.ones_like(steps)
    else:
        discount = gamma**steps
        cost = (1.0 - discount) / (1.0 - gamma)
    return reached * distance + (1.0 - reached) * (cost + discount * bootstrap)


def _make_head(cfg: TDConfig, latent_dim: int) -> MetricHead:
    if cfg.head == "iqe":
        return IQEHead(
            latent_dim,
            hidden_dim=cfg.hidden_dim,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_components=cfg.num_components,
            alpha_init=cfg.alpha_init,
        )
    if cfg.head == "quasimetric":
        return QuasimetricHead(
            latent_dim,
            hidden_dim=cfg.hidden_dim,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            sym_frac=cfg.sym_frac,
        )
    return PairwiseMetricHead(
        latent_dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        softplus=cfg.softplus,
        symmetric=cfg.symmetric,
        scale=1.0,
    )


def fit(
    cache: LatentCache,
    cfg: TDConfig,
    device: str,
    snapshot: Callable[[MetricHead, int], None] | None = None,
) -> MetricHead:
    """Train a temporal-distance head, handing ``snapshot`` a CPU copy every ``save_every`` steps."""
    torch.manual_seed(cfg.seed)
    value = _make_head(cfg, cache.latent_dim).to(device)
    target = copy.deepcopy(value).to(device)
    for p in target.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(value.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sampler = NStepGoalSampler(
        cache,
        n_step=cfg.n_step,
        p_cross=cfg.p_cross,
        n_buckets=cfg.n_buckets,
        balanced=cfg.balanced,
        seed=cfg.seed,
        max_delta=cfg.max_delta,
        near_frac=cfg.near_frac,
        near_max=cfg.near_max,
    )
    if cfg.near_frac > 0:
        logger.info(f"TD near-goal oversampling: frac={cfg.near_frac} max={cfg.near_max} steps")
    value.train()
    log_interval = max(1, cfg.steps // 20)
    for step in range(cfg.steps):
        batch = cast(dict[str, torch.Tensor], sampler.sample(cfg.batch_size))
        sample = {key: batch[key].to(device) for key in TD_KEYS}
        with torch.no_grad():
            target_value = n_step_target(sample, target(sample["z_tn"], sample["z_g"]), cfg.gamma)
        pred = value(sample["z_t"], sample["z_g"])
        loss = expectile_loss(pred - target_value, cfg.expectile, cfg.huber_beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch 2.7 Tensor.backward lacks a typed signature.
        opt.step()
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), value.parameters(), strict=True):
                tp.mul_(1.0 - cfg.tau).add_(cfg.tau * sp)
        if step == 0 or (step + 1) % log_interval == 0 or step + 1 == cfg.steps:
            logger.info(
                f"TD training step={step + 1}/{cfg.steps} loss={loss.item():.6f} "
                f"prediction_mean={pred.mean().item():.6f}"
            )
        if snapshot is not None and cfg.save_every > 0 and (step + 1) % cfg.save_every == 0 and step + 1 < cfg.steps:
            snapshot(copy.deepcopy(value).cpu().eval(), step + 1)
    logger.success(f"TD trained (n={cfg.n_step} gamma={cfg.gamma} head={cfg.head}), final loss={loss.item():.4f}")
    value.eval()
    return value


__all__ = ["TDConfig", "expectile_loss", "fit", "n_step_target"]
