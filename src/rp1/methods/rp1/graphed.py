"""CUDA-graph capture of one rp1 refinement step: the rollout's energy and its gradient.

A planning decision on a small batch is bound by kernel launches rather than
compute, so the rollout and its backward are recorded once per batch size and
replayed every refinement iteration. CUDA graphs fix shapes and addresses: inputs
are copied into static buffers, and the world model's and value's weights must
not be reallocated after capture.
"""

from collections.abc import Callable
from typing import Any

import torch

from rp1.core.agent.value.temporal import ValueFunction
from rp1.core.world_model.rollout import rollout_traj
from rp1.methods.rp1.energy import plan_energy
from rp1.utils.logging import logger

__all__ = ["GraphedRefinement"]


class GraphedRefinement:
    """The energy and energy gradient of a plan, replayed from a CUDA graph.

    ``bind()`` stages one decision's latent history, action history and goal,
    capturing the graph the first time a batch size is seen; ``step()`` then
    serves one refinement iteration.
    """

    def __init__(
        self,
        wm: Any,
        value: ValueFunction,
        horizon: int,
        action_dim: int,
        latent_dim: int,
        device: torch.device | str,
        warmup_iters: int,
    ) -> None:
        if not torch.cuda.is_available() or torch.device(device).type != "cuda":
            raise ValueError("graphed refinement requires a CUDA device")
        self._wm = wm
        self._value = value
        self._horizon = horizon
        self._action_dim = action_dim
        self._latent_dim = latent_dim
        self._device = torch.device(device)
        self._warmup_iters = warmup_iters
        # batch size -> (graphed energy, latent history, action history, goal buffers)
        self._captured: dict[int, tuple[Callable[..., Any], torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._bound: int | None = None

    def _capture(self, batch: int) -> None:
        logger.info(f"GraphedRefinement: capturing CUDA graphs for batch {batch} (one-time, takes seconds)")
        z_history = torch.zeros(batch, 3, self._latent_dim, device=self._device)
        a_history = torch.zeros(batch, 2, self._action_dim, device=self._device)
        z_goal = torch.zeros(batch, self._latent_dim, device=self._device)
        wm, value = self._wm, self._value

        def energy(plan: torch.Tensor) -> torch.Tensor:
            return plan_energy(value, rollout_traj(wm, z_history, a_history, plan), z_goal, frames=1)

        sample = torch.zeros(batch, self._horizon, self._action_dim, device=self._device, requires_grad=True)
        graphed = torch.cuda.make_graphed_callables(  # type: ignore[no-untyped-call]  # PyTorch 2.7 stub is untyped.
            energy, (sample,), num_warmup_iters=self._warmup_iters
        )
        self._captured[batch] = (graphed, z_history, a_history, z_goal)

    def bind(self, z_history: torch.Tensor, a_history: torch.Tensor, z_goal: torch.Tensor) -> None:
        """Stage one decision's context; captures the graph for a new batch size."""
        batch = z_history.shape[0]
        if batch not in self._captured:
            self._capture(batch)
        _, z_history_buffer, a_history_buffer, z_goal_buffer = self._captured[batch]
        z_history_buffer.copy_(z_history)
        a_history_buffer.copy_(a_history)
        z_goal_buffer.copy_(z_goal)
        self._bound = batch

    def step(self, plan: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The energy of ``plan`` and its gradient with respect to the plan."""
        if self._bound is None or plan.shape[0] != self._bound:
            raise RuntimeError("GraphedRefinement.step called without a matching bind()")
        graphed, *_ = self._captured[self._bound]
        plan = plan.detach().requires_grad_(True)
        energy = graphed(plan)
        (gradient,) = torch.autograd.grad(energy.sum(), plan)
        return energy.detach(), gradient
