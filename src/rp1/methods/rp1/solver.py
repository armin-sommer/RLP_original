"""The rp1 solver: a trained planner network refines a plan along the value gradient."""

import time
from typing import Any, cast

import numpy as np
import torch
from stable_worldmodel.solver.cem import CEMSolver

from rp1.core.agent.solver.base import EncoderWorldModel, PlannerCheckpoint, unwrap_encoder
from rp1.core.agent.value.temporal import ValueFunction
from rp1.core.world_model.base import LatentWorldModel
from rp1.core.world_model.rollout import rollout_traj
from rp1.methods.rp1.energy import plan_energy
from rp1.methods.rp1.graphed import GraphedRefinement
from rp1.methods.rp1.net import PlannerNet
from rp1.utils.logging import logger

__all__ = ["RP1Solver"]


class RP1Solver(CEMSolver):
    """Plan with a trained rp1 checkpoint.

    Every decision encodes the observation and the goal, starts from the zero plan
    and applies the planner network K times, each time feeding it the plan's energy
    and energy gradient through the frozen world model. The result is deterministic.
    The horizon and K come from the checkpoint.

    ``CEMSolver`` provides only the environment bookkeeping (``configure``,
    ``action_dim``, device and dtype); no sampling takes place.
    """

    def __init__(
        self, *args: Any, checkpoint: PlannerCheckpoint, graphed: bool, graph_warmup_iters: int, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        payload = checkpoint.payload
        if payload.get("temporal_objective", "terminal") != "terminal" or payload.get("grounding"):
            raise ValueError("the planner was trained on an energy other than the terminal value")
        self.planner = PlannerNet(
            horizon=int(payload["horizon"]),
            action_dim=int(payload["action_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            action_limit=float(payload["action_limit"]),
        ).to(self.device)
        self.planner.load_state_dict(payload["state_dict"])
        self.planner.eval()
        self.iterations = int(payload["iterations"])
        self.value = cast(ValueFunction, checkpoint.value.to(self.device).eval())
        self.frames = int(payload["window_frames"])
        if self.frames > 1:
            logger.info(f"Window value over {self.frames} frames")
            if graphed:
                raise ValueError("graphed refinement does not support windowed values")
        self.graphed = graphed
        self.graph_warmup_iters = graph_warmup_iters
        self._graphed_refinement: GraphedRefinement | None = None

    @property
    def horizon(self) -> int:
        return self.planner.horizon

    def solve(self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None) -> dict[str, Any]:
        del init_action  # every decision refines from the zero plan
        start = time.time()
        wm = cast(EncoderWorldModel, unwrap_encoder(self.model))
        z_history, z_goal = self._encode(wm, info_dict)
        encoded = time.time()
        plan = self._refine(wm, z_history, z_goal)
        done = time.time()
        logger.info(
            f"RP1 solve completed in {done - start:.4f} seconds "
            f"(encode {encoded - start:.4f}, plan {done - encoded:.4f})"
        )
        return {"actions": plan.to(self.dtype).cpu()}

    @torch.no_grad()
    def _encode(self, wm: EncoderWorldModel, info_dict: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """The last three observation latents ``(B, 3, D)`` and the goal latent ``(B, D)``."""
        # with planning.history_len > 1 the policy publishes lagged frames one action block apart
        pixels = info_dict.get("pixels_hist", info_dict["pixels"]).to(self.device, dtype=self.dtype)
        observation = {"pixels": pixels}
        if getattr(wm, "wants_proprio", False):
            proprio = info_dict.get("proprio")
            if proprio is None:
                raise KeyError("a proprio-variant world model needs 'proprio' in the observation")
            proprio = torch.as_tensor(np.asarray(proprio), dtype=torch.float32, device=self.device)
            observation["proprio"] = proprio.reshape(pixels.shape[0], pixels.shape[1], -1)
        z_history = wm.encode(observation)["emb"][:, -3:].float()
        if z_history.shape[1] < 3:  # episode start, or planning.history_len < 3: repeat the oldest frame
            padding = z_history[:, :1].expand(-1, 3 - z_history.shape[1], -1)
            z_history = torch.cat([padding, z_history], dim=1)

        goal_pixels = info_dict["goal"].to(self.device, dtype=self.dtype)
        goal = {"pixels": goal_pixels}
        if getattr(wm, "wants_proprio", False):
            goal_state = info_dict.get("goal_state")
            if goal_state is None:
                raise KeyError("a proprio-variant world model needs 'goal_state' in the observation")
            goal_state = torch.as_tensor(np.asarray(goal_state), dtype=torch.float32, device=self.device)
            position = goal_state.reshape(goal_pixels.shape[0], -1)[:, -2:]  # the last frame's (x, y)
            goal["proprio"] = position.unsqueeze(1).expand(-1, goal_pixels.shape[1], -1)
        z_goal = wm.encode(goal)["emb"][:, -1].float()
        return z_history, z_goal

    def _refine(self, wm: LatentWorldModel, z_history: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """K learned refinement steps from the zero plan; the refined plan ``(B, H, action_dim)``."""
        batch = z_history.shape[0]
        # the executed actions are not passed on: planning starts from a zero action history,
        # as it does for the trainer's replayed starts
        a_history = torch.zeros(batch, 2, self.action_dim, device=self.device)
        graphed = self._graphed(wm, z_history, a_history, z_goal) if self.graphed else None
        plan = torch.zeros(batch, self.horizon, self.action_dim, device=self.device)
        for _ in range(self.iterations):
            if graphed is not None:
                energy, gradient = graphed.step(plan)
            else:
                energy, gradient = self._energy_and_gradient(wm, z_history, a_history, z_goal, plan)
            with torch.no_grad():
                plan = self.planner(plan, gradient, energy)
        return plan.detach()

    def _energy_and_gradient(
        self,
        wm: LatentWorldModel,
        z_history: torch.Tensor,
        a_history: torch.Tensor,
        z_goal: torch.Tensor,
        plan: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.enable_grad():  # type: ignore[no-untyped-call]  # PyTorch 2.7 context-manager stub is untyped.
            plan = plan.detach().requires_grad_(True)
            energy = plan_energy(self.value, rollout_traj(wm, z_history, a_history, plan), z_goal, self.frames)
            (gradient,) = torch.autograd.grad(energy.sum(), plan)
        return energy.detach(), gradient

    def _graphed(
        self, wm: LatentWorldModel, z_history: torch.Tensor, a_history: torch.Tensor, z_goal: torch.Tensor
    ) -> GraphedRefinement:
        """The CUDA-graphed refinement step, built on the first decision, bound to this one."""
        if self._graphed_refinement is None:
            self._graphed_refinement = GraphedRefinement(
                wm,
                self.value,
                self.horizon,
                self.action_dim,
                z_history.shape[-1],
                self.device,
                warmup_iters=self.graph_warmup_iters,
            )
        self._graphed_refinement.bind(z_history, a_history, z_goal)
        return self._graphed_refinement
