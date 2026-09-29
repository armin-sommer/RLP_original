"""Contrastive value learning (contrastive RL / InfoNCE).

A critic ``f(z_s, z_g) = phi(z_s) . psi(z_g)``, monotone in discounted
reachability, following contrastive RL (Eysenbach et al.). For a
batch of anchors ``S`` and geometric-future goals ``G`` (positives on the
diagonal), the symmetric InfoNCE objective is::

    logits = phi(S) @ psi(G) ^ T / temperature  # (B, B)
    L = 0.5 * (CE(logits, arange(B)) + CE(logits ^ T, arange(B)))

A high critic value means "goal is reachable from state", so the terminal cost
handed to the planner is the negative critic: ``cost = -f(z_pred, z_goal)``
(lower == more reachable).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rp1.core.agent.value.head import ContrastiveCritic
from rp1.data import LatentCache
from rp1.training.phases.agent.samplers import GeometricFutureSampler
from rp1.utils.logging import logger


@dataclass
class ContrastiveConfig:
    hidden_dim: int
    rep_dim: int
    depth: int
    gamma: float
    temperature: float
    lr: float
    weight_decay: float
    batch_size: int
    steps: int
    seed: int


def fit(cache: LatentCache, cfg: ContrastiveConfig, device: str) -> ContrastiveCritic:
    torch.manual_seed(cfg.seed)
    critic = ContrastiveCritic(
        cache.latent_dim,
        hidden_dim=cfg.hidden_dim,
        rep_dim=cfg.rep_dim,
        depth=cfg.depth,
    ).to(device)
    opt = torch.optim.AdamW(critic.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sampler = GeometricFutureSampler(cache, gamma=cfg.gamma, seed=cfg.seed)

    critic.train()
    log_interval = max(1, cfg.steps // 20)
    for step in range(cfg.steps):
        batch = sampler.sample(cfg.batch_size)
        z_s = batch["z_s"].to(device)
        z_g = batch["z_g"].to(device)
        s_rep = critic.phi(z_s)  # (B, R)
        g_rep = critic.psi(z_g)  # (B, R)
        logits = (s_rep @ g_rep.t()) / cfg.temperature  # (B, B)
        labels = torch.arange(logits.shape[0], device=device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch 2.7 Tensor.backward lacks a typed signature.
        opt.step()
        if step == 0 or (step + 1) % log_interval == 0 or step + 1 == cfg.steps:
            acc = (logits.argmax(dim=1) == labels).float().mean()
            logger.info(
                f"Contrastive training step={step + 1}/{cfg.steps} loss={loss.item():.6f} accuracy={acc.item():.6f}"
            )
    logger.success(f"Contrastive critic trained ({cfg.steps} steps), final loss={loss.item():.4f}")
    critic.eval()
    return critic


__all__ = ["ContrastiveConfig", "fit"]
