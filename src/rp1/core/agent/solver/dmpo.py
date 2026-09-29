"""DMPOSolver: plan with a trained DMPO checkpoint.

*Deep Model Predictive Optimization* (Sacks et al., ICRA 2024) keeps MPC's
sample-rollout-reduce loop and learns the reduction; :class:`DMPONet` holds the
learned pieces and this solver supplies the rollouts and the cost.

A decision costs ``num_samples * iters`` forward world-model unrolls and no
backward pass. The network reads the ``N`` costs positionally, so the sample
count and the horizon are fixed by the checkpoint; the iteration count is not
(``iters``). Costs come from the checkpoint's value through the frozen world
model, the value the rp1 solver plans against too.

The learned warm start consumes the unexecuted plan tail that
:class:`~rp1.core.agent.policy.WorldModelPolicy` passes as ``init_action``.
With ``receding_horizon == horizon`` nothing is left of it; the closed-loop
regime DMPO was published in is ``planning.receding_horizon=1``.
"""

import time
from collections.abc import Callable, Sequence
from typing import Any, NotRequired, TypedDict, cast

import numpy as np
import torch
from stable_worldmodel.solver.cem import CEMSolver

from rp1.core.agent.planner.dmpo import DMPONet
from rp1.core.agent.solver.base import EncoderWorldModel, PlannerCheckpoint, unwrap_encoder
from rp1.core.agent.value.temporal import ValueFunction, trajectory_value, windowed_terminal_value
from rp1.core.world_model.base import LatentWorldModel
from rp1.core.world_model.rollout import rollout_traj
from rp1.utils.logging import logger


class DMPOCheckpoint(TypedDict):
    kind: str
    z_dim: int
    horizon: int
    a_dim: int
    iters: int
    num_samples: int
    value: str
    sd: dict[str, torch.Tensor]
    amax: NotRequired[float]
    action_range: NotRequired[float | None]
    init_std: NotRequired[float]
    hidden: NotRequired[int]
    temperature: NotRequired[float]
    step_size: NotRequired[float]
    scale_costs: NotRequired[bool]
    gated: NotRequired[bool]
    residual: NotRequired[bool]
    learn_std: NotRequired[bool]
    use_shift: NotRequired[bool]
    gate_activation: NotRequired[str]
    halton: NotRequired[bool]
    seed_val: NotRequired[int]
    learn_search_std: NotRequired[bool]
    mean_search_std: NotRequired[float]
    std_search_std: NotRequired[float]
    objective: NotRequired[str]
    temporal_objective: NotRequired[str]
    value_context: NotRequired[int]


__all__ = ["DMPOSolver"]


class DMPOSolver(CEMSolver):
    """Sample-and-reduce planning with DMPO's learned update rule."""

    def __init__(
        self,
        *args: Any,
        checkpoint: PlannerCheckpoint,
        iters: int | None,
        mppi_mode: bool,
        cost_chunk: int,
        report_cost: bool,
        graphed: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # mppi_mode bypasses the learned heads: the hand-written MPPI update DMPO
        # learns a residual on (the reference's ``is_mppi``)
        self.mppi_mode = bool(mppi_mode)
        self.cost_chunk = int(cost_chunk)
        self.report_cost = bool(report_cost)
        self.graphed = graphed
        self._graphed_cost: Any = None

        ck = cast(DMPOCheckpoint, checkpoint.payload)
        if ck.get("kind") != "dmpo":
            raise ValueError(f"DMPOSolver: unsupported checkpoint kind {ck.get('kind')!r}")
        self.net = DMPONet(
            horizon=ck["horizon"],
            a_dim=ck["a_dim"],
            num_samples=ck["num_samples"],
            hidden=ck.get("hidden", 256),
            amax=ck.get("amax", 2.5),
            init_std=ck.get("init_std", 1.0),
            temperature=ck.get("temperature", 0.05),
            step_size=ck.get("step_size", 0.8),
            scale_costs=ck.get("scale_costs", True),
            gated=ck.get("gated", True),
            residual=ck.get("residual", True),
            learn_std=ck.get("learn_std", True),
            use_shift=ck.get("use_shift", True),
            gate_activation=ck.get("gate_activation", "tanh"),
            halton=ck.get("halton", True),
            seed_val=ck.get("seed_val", 0),
            # present in on-policy checkpoints; deployment uses the locations,
            # but the heads must exist for the state dict to load
            learn_search_std=ck.get("learn_search_std", False),
            mean_search_std=ck.get("mean_search_std", 0.1),
            std_search_std=ck.get("std_search_std", 0.01),
        ).to(self.device)
        # a checkpoint without per-dimension bounds clips symmetrically at amax;
        # filling the buffers keeps the load strict
        state = dict(ck["sd"])
        for key, fill in (("a_low", -float(ck.get("amax", 2.5))), ("a_high", float(ck.get("amax", 2.5)))):
            if key not in state:
                state[key] = torch.full((int(ck["a_dim"]),), fill)
        self.net.load_state_dict(state)
        low, high = self.net.bounds
        logger.info(f"DMPO action bounds: [{float(low.min()):.2f}, {float(high.max()):.2f}]")
        self.net.eval()
        self.net.requires_grad_(False)
        self._actor_horizon = int(ck["horizon"])
        self.iters = int(ck["iters"] if iters is None else iters)
        requested_samples: int = self.num_samples
        if requested_samples != self.net.num_samples:
            logger.info(
                f"DMPO sample count is fixed by the checkpoint: using {self.net.num_samples} "
                f"(config asked for {requested_samples})"
            )
        self.temporal_objective = ck.get("temporal_objective", "terminal")

        value_module = checkpoint.value.to(self.device)
        value_module.eval()
        self.value = cast(ValueFunction, value_module)
        # window values score the last `context` imagined frames, as MetricCost does
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

        def cost(plans: torch.Tensor) -> torch.Tensor:
            batch, samples = plans.shape[0], plans.shape[1]
            flat = plans.reshape(batch * samples, plans.shape[2], plans.shape[3])
            zh = z_hist.repeat_interleave(samples, dim=0)
            ah = a_hist.repeat_interleave(samples, dim=0)
            zg = z_goal.repeat_interleave(samples, dim=0)
            size = flat.shape[0] if chunk <= 0 else chunk
            scored: list[torch.Tensor] = []
            for start in range(0, flat.shape[0], size):
                stop = start + size
                traj = rollout_traj(wm, zh[start:stop], ah[start:stop], flat[start:stop])
                if self.value_context > 1:
                    scored.append(windowed_terminal_value(self.value, traj, zg[start:stop], self.value_context))
                else:
                    scored.append(
                        trajectory_value(
                            self.value,
                            traj,
                            zg[start:stop],
                            zh[start:stop, -1],
                            self.temporal_objective,
                        )
                    )
            return torch.cat(scored).view(batch, samples)

        return cost

    def _graphed_cost_fn(
        self,
        z_hist: torch.Tensor,
        a_hist: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """The captured graph of the cost evaluation, bound to this decision."""
        if self._graphed_cost is None:
            from rp1.core.agent.solver.graphed_dmpo import GraphedSampledCost

            self._graphed_cost = GraphedSampledCost(
                cast(LatentWorldModel, self._base()),
                self.value,
                horizon=self.horizon,
                action_dim=self.action_dim,
                latent_dim=z_hist.shape[-1],
                num_samples=self.net.num_samples,
                device=self.device,
                temporal_objective=self.temporal_objective,
                value_context=self.value_context,
            )
        self._graphed_cost.bind(z_hist, a_hist, z_goal)
        return cast(Callable[[torch.Tensor], torch.Tensor], self._graphed_cost)

    # ---------------------------------------------------------------- solve
    def solve(self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None) -> dict[str, Any]:
        start_time = time.time()
        z_hist, z_goal = self._encode(info_dict)
        batch = z_hist.shape[0]
        a_hist = torch.zeros(batch, 2, self.action_dim, device=self.device)
        mean, std = self.net.initial(batch, self.device)

        with torch.no_grad():
            if init_action is not None and init_action.shape[1] > 0 and not self.mppi_mode:
                # previous decision's unexecuted tail, right-aligned in the plan
                previous = torch.zeros_like(mean)
                tail = init_action.to(device=self.device, dtype=mean.dtype)[:, : self.horizon]
                previous[:, self.horizon - tail.shape[1] :] = tail
                executed = self.horizon - tail.shape[1]
                mean, std = self.net.warm_start(previous, std, executed)

            cost_fn = self._cost_fn(z_hist, a_hist, z_goal)
            if self.graphed and not self.mppi_mode:
                cost_fn = self._graphed_cost_fn(z_hist, a_hist, z_goal)
            if self.mppi_mode:
                for _ in range(self.iters):
                    plans = self.net.plans(mean, std)
                    mean = self.net.clip(self.net.mppi_mean(mean, plans, cost_fn(plans)))
            else:
                mean, std, _ = self.net.plan(cost_fn, mean, std, self.iters)
            # the policy only consumes `actions`; scoring the final mean costs an extra rollout
            final = (
                cost_fn(mean.unsqueeze(1)).squeeze(1)
                if self.report_cost
                else torch.full((batch,), float("nan"), device=self.device)
            )

        plan = mean.detach().to(self.dtype).cpu()
        logger.info(f"DMPO solve completed in {time.time() - start_time:.4f} seconds")
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
