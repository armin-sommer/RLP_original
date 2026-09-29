"""MetricCost: a planning cost that scores a frozen world model's rollouts with a learned value.

It implements Stable-WM's cost protocol, so any sampling solver plans with it in
place of the world model. The world model's own cost computes the predicted
terminal latent and the goal latent; the value then scores them.

Modes:

* ``latent``      -- passthrough baseline ``c_lat = ||z_hat_T - z_g||^2`` (the
                     mismatched Euclidean cost the paper repairs).
* ``replacement`` -- ``m_phi(z_hat_T, z_g)`` (TRM replaces the cost).
* ``hybrid``      -- ``std(c_lat) + lambda * std(m_phi)`` with per-row (per-env,
                     over candidates) standardisation to prevent scale dominance.
* ``shuffled``    -- like ``replacement`` but with a head trained on shuffled
                     temporal labels (negative control).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, cast

import torch
from torch import nn

from rp1.core.agent.value.adapter import as_planning_cost
from rp1.core.agent.value.base import TensorInfo, ValueMetric


class MetricCost(nn.Module):
    """Wrap a frozen world model and a value behind the cost interface.

    Args:
        base_wm: a frozen world model whose ``get_cost`` also fills
            ``info_dict['predicted_emb']`` (rollout) and ``info_dict['goal_emb']``
            (encoded goal).
        metric: a module with ``cost(z_pred, z_goal) -> Tensor`` (regression head,
            TD value, or contrastive critic). ``None`` only for ``latent`` mode.
        mode: one of ``latent | replacement | hybrid | shuffled``.
        lam: hybrid weighting ``lambda`` on the standardised metric term.
    """

    def __init__(
        self,
        base_wm: nn.Module,
        metric: nn.Module | None,
        mode: str,
        lam: float,
        metrics: Sequence[nn.Module] | None,
        deadline_mode: str,
    ) -> None:
        super().__init__()
        assert mode in {"latent", "replacement", "hybrid", "shuffled"}, mode
        if mode != "latent":
            assert metric is not None, f"mode={mode} requires a metric module"
        self.base = as_planning_cost(base_wm)
        self.metric = metric
        self.metrics = nn.ModuleList(metrics or ([] if metric is None else [metric]))
        self.mode = mode
        self.lam = lam
        if deadline_mode not in {"terminal", "deadline"}:
            raise ValueError(f"unsupported deadline mode: {deadline_mode}")
        self.deadline_mode = deadline_mode
        self._align_remaining: tuple[int, ...] | None = None

    def set_align_remaining(self, remaining_chunks: Sequence[int] | None) -> None:
        """Publish per-environment chunks remaining until the graded step."""
        self._align_remaining = None if remaining_chunks is None else tuple(map(int, remaining_chunks))

    def parameters(self, *args: Any, **kwargs: Any) -> Iterator[nn.Parameter]:
        return self.base.parameters(*args, **kwargs)

    @staticmethod
    def _standardize(x: torch.Tensor) -> torch.Tensor:
        """Per-row (over candidate dim) standardisation."""
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mu) / sd

    def _metric_inputs(
        self,
        info_dict: TensorInfo,
        action_candidates: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build terminal/history inputs matching the metric checkpoint width."""
        if self.metric is None:
            raise RuntimeError("metric input requested in latent-only mode")
        predicted = info_dict["predicted_emb"]
        goal_raw = info_dict["goal_emb"]
        base_dim = predicted.shape[-1]
        # a metric without ``latent_dim`` takes one frame; a wider one takes a window
        metric_dim = int(getattr(self.metric, "latent_dim", base_dim))
        if metric_dim % base_dim:
            raise ValueError(f"metric latent_dim={metric_dim} is not a multiple of world-model latent dim={base_dim}")
        context = metric_dim // base_dim
        if context < 1:
            raise ValueError(f"invalid metric context width: {context}")

        goal = goal_raw[..., -1, :]
        timeline = predicted.shape[-2]
        end = torch.full((predicted.shape[0],), timeline - 1, device=predicted.device, dtype=torch.long)
        remaining_raw = info_dict.get("_align_remaining")
        if (
            self.deadline_mode == "deadline"
            and action_candidates is not None
            and (remaining_raw is not None or self._align_remaining is not None)
        ):
            if remaining_raw is None:
                remaining = torch.as_tensor(self._align_remaining, device=predicted.device)
            else:
                remaining = remaining_raw.to(device=predicted.device)
                if remaining.ndim > 1:
                    remaining = remaining[:, 0]
            if remaining.numel() != predicted.shape[0]:
                raise ValueError(
                    f"deadline metadata does not match planning batch: {remaining.numel()} != {predicted.shape[0]}"
                )
            plan_chunks = int(action_candidates.shape[-2])
            offset = timeline - plan_chunks
            end = offset + remaining.clamp(1, plan_chunks) - 1

        offsets = torch.arange(context, device=predicted.device) - context + 1
        indices = (end[:, None] + offsets[None]).clamp_min(0)
        view = indices[:, None, :, None].expand(predicted.shape[0], predicted.shape[1], context, predicted.shape[-1])
        window = predicted.gather(-2, view)
        pred = window.flatten(start_dim=-2)
        if context == 2:
            current, previous = window[..., -1, :], window[..., -2, :]
            pred = torch.cat([current, current - previous], dim=-1)
            goal = torch.cat([goal, torch.zeros_like(goal)], dim=-1)
        elif context > 2:
            goal = torch.cat([goal] * context, dim=-1)

        if goal.ndim < pred.ndim:
            goal = goal.unsqueeze(1)
        return pred, goal.expand_as(pred)

    def _metric_terminal_cost(
        self,
        info_dict: TensorInfo,
        action_candidates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the learned metric to the cached terminal/goal latents.

        Expects ``predicted_emb`` (B, S, T, D) and ``goal_emb`` in either
        ``(B, T, D)`` or Stable-WM's ``(B, 1, T, D)`` form.
        """
        pred, goal = self._metric_inputs(info_dict, action_candidates)
        costs = [cast(ValueMetric, metric).cost(pred.float(), goal.float()) for metric in self.metrics]
        return torch.stack(costs).amax(dim=0)

    def get_cost(self, info_dict: TensorInfo, action_candidates: torch.Tensor) -> torch.Tensor:
        # also fills predicted_emb and goal_emb
        c_lat = self.base.get_cost(info_dict, action_candidates)
        if self.mode == "latent":
            return c_lat
        m = self._metric_terminal_cost(info_dict, action_candidates)
        if self.mode in ("replacement", "shuffled"):
            return m
        return self._standardize(c_lat) + self.lam * self._standardize(m)

    def criterion(self, info_dict: TensorInfo) -> torch.Tensor:
        if self.mode == "latent":
            return self.base.criterion(info_dict)
        m = self._metric_terminal_cost(info_dict)
        if self.mode in ("replacement", "shuffled"):
            return m
        return self._standardize(self.base.criterion(info_dict)) + self.lam * self._standardize(m)


__all__ = ["MetricCost"]
