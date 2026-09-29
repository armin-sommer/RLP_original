"""Differentiable unroll of a frozen world model over an action plan.

The functions own no weights and call only ``wm.predict`` and ``wm.action_encoder``.
"""

import torch

from rp1.core.world_model.base import LatentWorldModel

__all__ = ["rollout_terminal", "rollout_traj"]


def rollout_terminal(
    wm: LatentWorldModel, z_hist: torch.Tensor, a_hist: torch.Tensor, plan: torch.Tensor
) -> torch.Tensor:
    """Autoregressive unroll of ``plan``; the terminal latent.

    z_hist: (B, 3, D) latent history, a_hist: (B, 2, a) action-block history,
    plan: (B, H, a). Differentiable w.r.t. ``plan``.
    """
    return rollout_traj(wm, z_hist, a_hist, plan)[:, -1]


def rollout_traj(wm: LatentWorldModel, z_hist: torch.Tensor, a_hist: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
    """Like :func:`rollout_terminal` but returns all H imagined latents (B, H, D)."""
    embs = list(z_hist.unbind(dim=1))
    acts = list(a_hist.unbind(dim=1))
    outs: list[torch.Tensor] = []
    for t in range(plan.shape[1]):
        acts.append(plan[:, t])
        win_e = torch.stack(embs[-3:], dim=1)
        win_a = torch.stack(acts[-3:], dim=1)
        nxt = wm.predict(win_e, torch.as_tensor(wm.action_encoder(win_a)))[:, -1]
        embs.append(nxt)
        outs.append(nxt)
    return torch.stack(outs, dim=1)
