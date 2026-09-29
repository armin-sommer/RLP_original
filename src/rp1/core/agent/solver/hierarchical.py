"""Hierarchical CEM: a macro-action search picks a waypoint, a primitive-action CEM plans toward it.

The high level searches ``hl_horizon`` macro-actions and scores the terminal
latent with a macro-level value, whose distances are counted in macro steps
(``hl_cost="value"``), or with the squared latent distance (``"latent"``). Its
dynamics are either the high-level world model
(:class:`rp1.core.world_model.hierarchical.HWM`, ``hl_dynamics="f2"``) or the
low-level world model composed over a bank of real action chunks
(``"compose"``), which removes the macro-action bottleneck and executes only
actions the data contains.

At stride 25 a 200-step OGBench cube goal is a distance-8 macro query present
in every episode, whereas a primitive-step value reaches it only by chaining
n-step backups that saturate.

Both levels refit on hard top-k elites, the rule of the flat CEM baseline, so
the optimizer stays out of the hierarchy comparison. ``oracle_subgoal``
replaces the high level with the dataset's true latent one macro ahead, which
separates subgoal generation from low-level execution.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
from torch import nn

from rp1.core.agent.solver.base import EncoderWorldModel, unwrap_encoder
from rp1.core.agent.solver.cem import CEMSolver
from rp1.core.agent.value.temporal import ValueFunction
from rp1.core.world_model.base import LatentWorldModel
from rp1.core.world_model.hierarchical import load_hwm
from rp1.core.world_model.rollout import rollout_traj
from rp1.data import LatentCache
from rp1.utils.logging import logger

__all__ = ["HierarchicalCEMSolver"]


def _refit(
    mean: torch.Tensor,
    iterations: int,
    samples: int,
    topk: int,
    cost: Callable[[torch.Tensor], torch.Tensor],
    limit: float | None,
) -> torch.Tensor:
    """CEM on hard top-k elites from ``mean``, candidates clamped to ``limit`` unless it is None."""
    batch, horizon, width = mean.shape
    std = torch.ones_like(mean)
    elites = min(topk, samples)
    with torch.no_grad():
        for _ in range(iterations):
            candidates = torch.randn(batch, samples, horizon, width, device=mean.device) * std.unsqueeze(1)
            candidates = candidates + mean.unsqueeze(1)
            candidates[:, 0] = mean  # keep the incumbent
            if limit is not None:
                candidates = candidates.clamp(-limit, limit)
            costs = cost(candidates.reshape(batch * samples, horizon, width)).view(batch, samples)
            best = torch.topk(costs, elites, dim=1, largest=False).indices
            rows = torch.arange(batch, device=mean.device).unsqueeze(1).expand(-1, elites)
            elite = candidates[rows, best]
            mean, std = elite.mean(dim=1), elite.std(dim=1).clamp(min=0.05)
    return mean


class HierarchicalCEMSolver(CEMSolver):
    """CEM at both levels of a two-level plan; ``n_steps`` and ``num_samples`` configure the low level."""

    def __init__(
        self,
        *args: Any,
        hl_dynamics: str,
        hwm_path: str | None,
        bank_path: str | None,
        hl_value: nn.Module | None,
        hl_cost: str,
        hl_opt: str,
        hl_lr: float,
        hl_adam_steps: int,
        hl_horizon: int,
        hl_samples: int,
        hl_iters: int,
        hl_topk: int,
        hl_amax: float,
        ll_samples: int | None,
        ll_iters: int | None,
        ll_topk: int,
        ll_amax: float,
        ll_clamp: bool,
        subgoal_index: int,
        oracle_subgoal: bool,
        cache_path: str | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if hl_cost not in ("value", "latent"):
            raise ValueError(f"hl_cost must be 'value' or 'latent', got {hl_cost!r}")
        if hl_opt not in ("cem", "adam"):
            raise ValueError(f"hl_opt must be 'cem' or 'adam', got {hl_opt!r}")
        if hl_dynamics not in ("f2", "compose"):
            raise ValueError(f"hl_dynamics must be 'f2' or 'compose', got {hl_dynamics!r}")
        if hl_cost == "value" and hl_value is None:
            raise ValueError("hl_cost='value' needs hl_value")
        self.hl_cost, self.hl_opt, self.hl_dynamics = hl_cost, hl_opt, hl_dynamics
        self.hl_lr, self.hl_adam_steps = hl_lr, hl_adam_steps
        self.hl_horizon, self.hl_samples, self.hl_iters, self.hl_amax = hl_horizon, hl_samples, hl_iters, hl_amax
        self.hl_topk = hl_topk or max(2, round(0.1 * hl_samples))
        self.ll_samples = ll_samples or self.num_samples
        self.ll_iters = ll_iters or self.n_steps
        self.ll_topk = ll_topk or max(2, round(0.1 * self.ll_samples))
        self.ll_amax, self.ll_clamp = ll_amax, ll_clamp
        self.subgoal_index = subgoal_index
        self.hl_value = None if hl_value is None else cast(ValueFunction, hl_value.to(self.device).eval())

        self.hwm: Any = None
        self.bank: torch.Tensor | None = None
        if hl_dynamics == "f2":
            if not hwm_path:
                raise ValueError("hl_dynamics='f2' needs hwm_path")
            self.hwm, saved = load_hwm(hwm_path, device=self.device)
            self.macro_dim = self.hwm.macro_dim
            self.hl_stride = int(saved["cfg"]["stride"])
        else:
            if not bank_path:
                raise ValueError("hl_dynamics='compose' needs bank_path")
            saved = torch.load(bank_path, map_location=self.device, weights_only=False)
            self.bank = saved["chunks"].to(self.device).float()
            self.hl_stride = int(saved["stride"])
            self.macro_dim = 0

        self.oracle_subgoal = oracle_subgoal
        self._oracle_latents: torch.Tensor | None = None
        self._oracle_rows: dict[int, Any] = {}
        self._episodes: list[int] | None = None
        self._starts: list[int] = []
        self._decision = 0
        self._previous_subgoal: torch.Tensor | None = None
        if oracle_subgoal:
            if not cache_path:
                raise ValueError("oracle_subgoal needs cache_path, the dense latent cache of the evaluation data")
            cache = LatentCache.load(cache_path, mmap=True)
            self._oracle_latents = cache.z.to(self.device).float()
            self._oracle_rows = cache.episodes()
            logger.info(f"Oracle subgoals from {cache_path} ({len(self._oracle_rows)} episodes)")
        logger.info(
            f"Hierarchical CEM: stride {self.hl_stride}, macro_dim {self.macro_dim}, hl_horizon {self.hl_horizon} "
            f"(= {self.hl_stride * self.hl_horizon} steps), hl_cost {self.hl_cost}"
        )

    def set_task_context(self, episodes: Sequence[int], start_steps: Sequence[int]) -> None:
        """The dataset episode and start step each environment runs; the oracle subgoal reads them."""
        self._episodes = [int(episode) for episode in episodes]
        self._starts = [int(start) for start in start_steps]
        self._decision = 0
        self._previous_subgoal = None

    def _terminal_cost(self, terminal: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if self.hl_cost == "value":
            assert self.hl_value is not None
            return self.hl_value(terminal, goal)
        return (terminal - goal).pow(2).sum(-1)

    def _oracle_waypoint(self, batch: int) -> torch.Tensor:
        """The true latent one macro ahead of this decision, per environment."""
        if self._episodes is None or self._oracle_latents is None:
            raise RuntimeError("oracle_subgoal needs set_task_context from the benchmark")
        ahead = (self._decision + 1) * self.hl_stride
        rows = []
        for index in range(batch):
            episode = self._oracle_rows[self._episodes[index]]
            rows.append(int(episode[min(self._starts[index] + ahead, len(episode) - 1)]))
        return self._oracle_latents[torch.as_tensor(rows, dtype=torch.long, device=self.device)]

    def _encode(self, info_dict: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``(B, 3, D)`` latent history and the ``(B, D)`` goal latent."""
        base = cast(EncoderWorldModel, unwrap_encoder(self.model))
        if getattr(base, "wants_proprio", False):
            raise NotImplementedError("the hierarchical solver does not support proprio world models")
        with torch.no_grad():
            pixels = info_dict.get("pixels_hist", info_dict["pixels"]).to(self.device, dtype=self.dtype)
            z_hist = base.encode({"pixels": pixels})["emb"][:, -3:].float()
            if z_hist.shape[1] < 3:  # pad a short history at the episode start
                z_hist = torch.cat([z_hist[:, :1].expand(-1, 3 - z_hist.shape[1], -1), z_hist], dim=1)
            goal = info_dict["goal"].to(self.device, dtype=self.dtype)
            z_goal = base.encode({"pixels": goal})["emb"][:, -1].float()
        return z_hist, z_goal

    def _adam_macros(self, z0: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """Macro-actions by Adam through the high-level world model and the terminal cost.

        The macro space is ``hl_horizon x macro_dim``, too wide for a few hundred
        CEM samples; the clamp sits inside the graph so the optimizer sees the bound.
        """
        macros = torch.zeros(z0.shape[0], self.hl_horizon, self.macro_dim, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([macros], lr=self.hl_lr)
        with torch.enable_grad():  # type: ignore[no-untyped-call]  # PyTorch stub is untyped.
            for _ in range(self.hl_adam_steps):
                trajectory = self.hwm.rollout_from(z0, macros.clamp(-self.hl_amax, self.hl_amax))
                cost = self._terminal_cost(trajectory[:, -1], z_goal).sum()
                optimizer.zero_grad(set_to_none=True)
                cost.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
        return macros.detach().clamp(-self.hl_amax, self.hl_amax)

    def solve(self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None) -> dict[str, Any]:
        del init_action
        start = time.time()
        z_hist, z_goal = self._encode(info_dict)
        batch = z_hist.shape[0]
        z0 = z_hist[:, -1]

        if self.oracle_subgoal:
            subgoal = self._oracle_waypoint(batch)
            if self.hl_value is not None:
                message = (
                    f"Oracle decision {self._decision}: V(z0, goal) {self.hl_value(z0, z_goal).mean():.2f}, "
                    f"V(z0, subgoal) {self.hl_value(z0, subgoal).mean():.2f}"
                )
                if self._previous_subgoal is not None and self._previous_subgoal.shape[0] == batch:
                    # what is left of the waypoint the low level aimed at over the last macro
                    message += f", residual {self.hl_value(z0, self._previous_subgoal).mean():.2f}"
                logger.info(message + " macro steps")
            self._previous_subgoal = subgoal
            self._decision += 1
            return self._solve_low(z_hist, subgoal, start)

        if self.hl_dynamics == "compose":
            return self._solve_compose(z_hist, z_goal, start)

        z0_samples = z0.repeat_interleave(self.hl_samples, dim=0)
        goal_samples = z_goal.repeat_interleave(self.hl_samples, dim=0)

        def high_cost(macros: torch.Tensor) -> torch.Tensor:
            return self._terminal_cost(self.hwm.rollout_from(z0_samples, macros)[:, -1], goal_samples)

        if self.hl_opt == "adam":
            macros = self._adam_macros(z0, z_goal)
        else:
            macros = _refit(
                torch.zeros(batch, self.hl_horizon, self.macro_dim, device=self.device),
                self.hl_iters,
                self.hl_samples,
                self.hl_topk,
                high_cost,
                self.hl_amax,
            )
        with torch.no_grad():
            waypoints = self.hwm.rollout_from(z0, macros)
            subgoal = waypoints[:, min(self.subgoal_index, waypoints.shape[1] - 1)]
            if self.hl_value is not None:
                logger.info(
                    f"High level: V(z0, goal) {self.hl_value(z0, z_goal).mean():.2f}, "
                    f"V(z0, subgoal) {self.hl_value(z0, subgoal).mean():.2f} macro steps"
                )
        return self._solve_low(z_hist, subgoal, start)

    def _solve_low(self, z_hist: torch.Tensor, subgoal: torch.Tensor, start: float) -> dict[str, Any]:
        """CEM over primitive action blocks toward ``subgoal``."""
        batch = z_hist.shape[0]
        base = cast(LatentWorldModel, unwrap_encoder(self.model))
        histories = z_hist.repeat_interleave(self.ll_samples, dim=0)
        actions = torch.zeros(batch * self.ll_samples, 2, self.action_dim, device=self.device)
        targets = subgoal.repeat_interleave(self.ll_samples, dim=0)

        def low_cost(plan: torch.Tensor) -> torch.Tensor:
            return (rollout_traj(base, histories, actions, plan)[:, -1] - targets).pow(2).sum(-1)

        plan = _refit(
            torch.zeros(batch, self.horizon, self.action_dim, device=self.device),
            self.ll_iters,
            self.ll_samples,
            self.ll_topk,
            low_cost,
            self.ll_amax if self.ll_clamp else None,
        )
        logger.info(f"Hierarchical solve completed in {time.time() - start:.4f} seconds")
        return {"actions": plan, "costs": [], "mean": [], "var": []}

    def _as_blocks(self, chunks: torch.Tensor) -> torch.Tensor:
        """``(B, S, K, act_dim)`` chunks as a ``(B, S*K/block, block*act_dim)`` plan."""
        batch, slots, length, width = chunks.shape
        per_block = self.action_dim // width
        return chunks.reshape(batch, slots * length // per_block, per_block * width)

    def _solve_compose(self, z_hist: torch.Tensor, z_goal: torch.Tensor, start: float) -> dict[str, Any]:
        """A categorical CEM over sequences of real action chunks, rolled through the low-level world model.

        The first chunk of the best sequence is executed as is, so every deployed
        action is one the data contains.
        """
        assert self.bank is not None
        base = cast(LatentWorldModel, unwrap_encoder(self.model))
        batch, samples, slots, bank = z_hist.shape[0], self.hl_samples, self.hl_horizon, self.bank.shape[0]
        histories = z_hist.repeat_interleave(samples, dim=0)
        actions = torch.zeros(batch * samples, 2, self.action_dim, device=self.device)
        goals = z_goal.repeat_interleave(samples, dim=0)
        logits = torch.zeros(batch, slots, bank, device=self.device)
        best: torch.Tensor | None = None
        with torch.no_grad():
            for _ in range(self.hl_iters):
                drawn = torch.multinomial(torch.softmax(logits, dim=-1).reshape(-1, bank), samples, replacement=True)
                drawn = drawn.reshape(batch, slots, samples).permute(0, 2, 1)  # (batch, samples, slots)
                plan = self._as_blocks(self.bank[drawn.reshape(-1, slots)])
                costs = self._terminal_cost(rollout_traj(base, histories, actions, plan)[:, -1], goals)
                elite = costs.view(batch, samples).topk(max(1, samples // 10), dim=1, largest=False).indices
                # each slot's next distribution: the log frequency of every chunk among the elites
                logits = torch.full_like(logits, -1e4)
                for row in range(batch):
                    chosen = drawn[row, elite[row]]
                    for slot in range(slots):
                        values, counts = torch.unique(chosen[:, slot], return_counts=True)
                        logits[row, slot, values] = counts.float().log()
                best = drawn[torch.arange(batch), elite[:, 0]]
            assert best is not None
            chunks = self.bank[best]
            first = self._as_blocks(chunks[:, :1])[:, : self.horizon]
            if first.shape[1] < self.horizon:  # repeat the last block of a short chunk
                first = torch.cat([first, first[:, -1:].expand(-1, self.horizon - first.shape[1], -1)], dim=1)
        logger.info(f"Hierarchical (compose) solve completed in {time.time() - start:.4f} seconds")
        return {"actions": first, "costs": [], "mean": [], "var": []}
