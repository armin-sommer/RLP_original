"""L2OSolver: plan with a trained L2O-MPC checkpoint.

*Learning to Optimize in Model Predictive Control* (Sacks & Boots, ICRA 2022)
keeps MPC's sample-rollout-reduce loop and learns the whole reduction;
:class:`L2ONet` holds the learned update and this solver supplies the rollouts
and the cost.

A decision costs ``num_samples * iters`` forward world-model unrolls and no
backward pass. The network reads the ``N`` costs positionally, so the sample
count and the horizon are fixed by the checkpoint; the iteration count is not
(``iters``). Costs come from the checkpoint's value through the frozen world
model.

L2O-MPC has no learned warm start: the unexecuted tail of the previous plan is
shifted forward. With ``receding_horizon == horizon`` nothing is left of it.
"""

import time
from collections.abc import Callable, Sequence
from typing import Any, NotRequired, TypedDict, cast

import numpy as np
import torch
from stable_worldmodel.solver.cem import CEMSolver

from rp1.core.agent.planner.l2o import L2ONet
from rp1.core.agent.solver.base import EncoderWorldModel, PlannerCheckpoint, unwrap_encoder
from rp1.core.agent.value.temporal import ValueFunction, trajectory_value, windowed_terminal_value
from rp1.core.world_model.base import LatentWorldModel
from rp1.core.world_model.rollout import rollout_traj
from rp1.utils.logging import logger


class L2OCheckpoint(TypedDict):
    kind: str
    z_dim: int
    horizon: int
    a_dim: int
    iters: int
    num_samples: int
    value: str
    sd: dict[str, torch.Tensor]
    amax: NotRequired[float]
    init_std: NotRequired[float]
    hidden: NotRequired[int]
    learn_std: NotRequired[bool]
    gate_bias: NotRequired[float]
    halton: NotRequired[bool]
    seed_val: NotRequired[int]
    temporal_objective: NotRequired[str]
    value_context: NotRequired[int]


__all__ = ["L2OSolver"]


class L2OSolver(CEMSolver):
    """Sample-and-reduce planning with L2O-MPC's learned update rule."""

    def __init__(
        self,
        *args: Any,
        checkpoint: PlannerCheckpoint,
        iters: int | None,
        cost_chunk: int,
        graphed: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cost_chunk = int(cost_chunk)
        self.graphed = graphed
        self._graph: Any = None

        ck = cast(L2OCheckpoint, checkpoint.payload)
        if ck.get("kind") != "l2o":
            raise ValueError(f"L2OSolver: unsupported checkpoint kind {ck.get('kind')!r}")
        self.net = L2ONet(
            horizon=ck["horizon"],
            a_dim=ck["a_dim"],
            num_samples=ck["num_samples"],
            hidden=ck.get("hidden", 1024),
            amax=ck.get("amax", 2.5),
            init_std=ck.get("init_std", 1.0),
            learn_std=ck.get("learn_std", False),
            gate_bias=ck.get("gate_bias", 0.0),
            halton=ck.get("halton", True),
            seed_val=ck.get("seed_val", 0),
        ).to(self.device)
        self.net.load_state_dict(ck["sd"])
        self.net.eval()
        self.net.requires_grad_(False)
        self._actor_horizon = int(ck["horizon"])
        self.iters = int(ck["iters"] if iters is None else iters)
        requested_samples: int = self.num_samples
        if requested_samples != self.net.num_samples:
            logger.info(
                f"L2O sample count is fixed by the checkpoint: using {self.net.num_samples} "
                f"(config asked for {requested_samples})"
            )
        self.temporal_objective = ck.get("temporal_objective", "terminal")

        value_module = checkpoint.value.to(self.device)
        value_module.eval()
        self.value = cast(ValueFunction, value_module)
        latent_dim = int(getattr(value_module, "latent_dim", ck["z_dim"]))
        self.value_context = int(ck.get("value_context", max(latent_dim // int(ck["z_dim"]), 1)))
        if self.value_context > 1 and self.temporal_objective != "terminal":
            raise ValueError("window critics support the terminal objective only")

    @property
    def horizon(self) -> int:
        return int(self._actor_horizon)

    def _base(self) -> torch.nn.Module:
        return unwrap_encoder(self.model)

    # ------------------------------------------------------------- encoding
    def _encode(self, info_dict: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Latent history ``(B, 3, D)`` and goal latent ``(B, D)``."""
        wm = cast(EncoderWorldModel, self._base())
        with torch.no_grad():
            px = info_dict["pixels"].to(self.device, dtype=self.dtype)
            enc_in = {"pixels": px}
            if getattr(wm, "wants_proprio", False):
                pro = info_dict.get("proprio")
                if pro is None:
                    raise KeyError("proprio-variant WM: info_dict lacks 'proprio'")
                pro = torch.as_tensor(np.asarray(pro), dtype=torch.float32, device=self.device)
                enc_in["proprio"] = pro.reshape(px.shape[0], px.shape[1], -1)
            z_hist = wm.encode(enc_in)["emb"][:, -3:].float()
            if z_hist.shape[1] < 3:
                pad = z_hist[:, :1].expand(-1, 3 - z_hist.shape[1], -1)
                z_hist = torch.cat([pad, z_hist], dim=1)
            gx = info_dict["goal"].to(self.device, dtype=self.dtype)
            genc_in = {"pixels": gx}
            if getattr(wm, "wants_proprio", False):
                gpro = info_dict.get("goal_state")
                if gpro is None:
                    raise KeyError("proprio-variant WM: info_dict lacks 'goal_state'")
                gpro = torch.as_tensor(np.asarray(gpro), dtype=torch.float32, device=self.device)
                gpro = gpro.reshape(gx.shape[0], -1)[:, -2:]
                genc_in["proprio"] = gpro.unsqueeze(1).expand(-1, gx.shape[1], -1)
            z_goal = wm.encode(genc_in)["emb"][:, -1].float()
        return z_hist, z_goal

    def _cost_fn(
        self,
        z_hist: torch.Tensor,
        a_hist: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        wm = cast(LatentWorldModel, self._base())
        chunk = self.cost_chunk

        def score_rows(zh: torch.Tensor, ah: torch.Tensor, zg: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
            """Cost of one flattened batch of plans — the whole inner-loop hot path."""
            traj = rollout_traj(wm, zh, ah, flat)
            if self.value_context > 1:
                return windowed_terminal_value(self.value, traj, zg, self.value_context)
            return trajectory_value(self.value, traj, zg, zh[:, -1], self.temporal_objective)

        if self.graphed and self._graph is None:
            from rp1.core.agent.solver.graphed import GraphedCost

            self._graph = GraphedCost(
                score_rows,
                horizon=self.horizon,
                action_dim=self.action_dim,
                latent_dim=z_hist.shape[-1],
                device=self.device,
            )

        def cost(plans: torch.Tensor) -> torch.Tensor:
            batch, samples = plans.shape[0], plans.shape[1]
            flat = plans.reshape(batch * samples, plans.shape[2], plans.shape[3])
            zh = z_hist.repeat_interleave(samples, dim=0)
            ah = a_hist.repeat_interleave(samples, dim=0)
            zg = z_goal.repeat_interleave(samples, dim=0)
            if self.graphed and chunk <= 0:
                # the captured graph owns the whole row count; chunking would
                # change the shape per call and defeat the capture
                return cast(torch.Tensor, self._graph.costs(zh, ah, zg, flat)).view(batch, samples)
            size = flat.shape[0] if chunk <= 0 else chunk
            scored: list[torch.Tensor] = [
                score_rows(
                    zh[start : start + size],
                    ah[start : start + size],
                    zg[start : start + size],
                    flat[start : start + size],
                )
                for start in range(0, flat.shape[0], size)
            ]
            return torch.cat(scored).view(batch, samples)

        return cost

    # ---------------------------------------------------------------- solve
    def solve(self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None) -> dict[str, Any]:
        start_time = time.time()
        z_hist, z_goal = self._encode(info_dict)
        encode_seconds = time.time() - start_time
        batch = z_hist.shape[0]
        a_hist = torch.zeros(batch, 2, self.action_dim, device=self.device)
        mean, std = self.net.initial(batch, self.device)

        with torch.no_grad():
            if init_action is not None and init_action.shape[1] > 0:
                # previous decision's unexecuted tail, right-aligned in the plan
                previous = torch.zeros_like(mean)
                tail = init_action.to(device=self.device, dtype=mean.dtype)[:, : self.horizon]
                previous[:, self.horizon - tail.shape[1] :] = tail
                executed = self.horizon - tail.shape[1]
                mean, std = self.net.warm_start(previous, std, executed)

            cost_fn = self._cost_fn(z_hist, a_hist, z_goal)
            mean, std, _ = self.net.plan(cost_fn, mean, std, self.iters)
            final = cost_fn(mean.unsqueeze(1)).squeeze(1)

        plan = mean.detach().to(self.dtype).cpu()
        total_seconds = time.time() - start_time
        logger.info(
            f"L2O solve completed in {total_seconds:.4f} seconds "
            f"(encode {encode_seconds:.4f}, plan {total_seconds - encode_seconds:.4f})"
        )
        return {
            "actions": plan,
            "mean": [plan],
            "var": [std.detach().to(self.dtype).cpu()],
            "costs": final.detach().float().cpu().tolist(),
        }

    def set_align_remaining(self, remaining_chunks: Sequence[int] | None) -> None:
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "set_align_remaining"):
            model.set_align_remaining(remaining_chunks)
