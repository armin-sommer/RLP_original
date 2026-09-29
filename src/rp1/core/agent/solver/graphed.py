"""CUDA-graph capture of the L2O-MPC solver's scoring: the forward-only cost of
``N`` sampled plans per problem, replayed every inner iteration.

A planning decision on a small batch is bound by kernel launches rather than
compute, so the scoring is recorded into a CUDA graph once and replayed.

CUDA graphs fix shapes and addresses: each batch size is captured once (which
takes seconds), inputs are copied into static buffers, and the world model's
and value's weights must not be reallocated after capture.
"""

from collections.abc import Callable

import torch

from rp1.utils.logging import logger

__all__ = ["GraphedCost"]


class GraphedCost:
    """Per-row-count cache of CUDA-graphed sampled-plan cost evaluations.

    The learned-optimizer solvers spend their whole decision in one shape:
    ``rows = batch * num_samples`` plans, each unrolled ``horizon`` blocks
    through the frozen world model and scored by the frozen critic, with no
    gradient anywhere. That is a fixed-shape, side-effect-free forward pass —
    the ideal CUDA-graph candidate, and it is where a launch-bound solver
    spends its wall-clock.

    ``costs()`` stages the decision's context and plans into static buffers and
    replays the graph, capturing on first use of a row count (seconds, once).
    The returned tensor is cloned: the graph's output buffer is overwritten by
    the next replay.
    """

    def __init__(
        self,
        score: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        horizon: int,
        action_dim: int,
        latent_dim: int,
        device: torch.device | str,
        warmup_iters: int = 3,
    ) -> None:
        if not torch.cuda.is_available() or torch.device(device).type != "cuda":
            raise ValueError("graphed cost evaluation requires a CUDA device")
        self._score = score
        self._horizon = int(horizon)
        self._action_dim = int(action_dim)
        self._latent_dim = int(latent_dim)
        self._device = torch.device(device)
        self._warmup_iters = int(warmup_iters)
        # rows -> (graph, z_hist, a_hist, z_goal, plans, costs) static tensors
        self._captured: dict[
            int, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

    def _capture(self, rows: int) -> None:
        logger.info(f"GraphedCost: capturing a CUDA graph for {rows} rows (one-time, takes seconds)")
        z_hist = torch.zeros(rows, 3, self._latent_dim, device=self._device)
        a_hist = torch.zeros(rows, 2, self._action_dim, device=self._device)
        z_goal = torch.zeros(rows, self._latent_dim, device=self._device)
        plans = torch.zeros(rows, self._horizon, self._action_dim, device=self._device)

        # Warm up on a side stream first: capture records whatever allocations
        # and cuBLAS handles already exist, so the first calls must not be
        # inside the capture.
        stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]  # PyTorch 2.7 stub is untyped.
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self._warmup_iters):
                self._score(z_hist, a_hist, z_goal, plans)
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()  # type: ignore[no-untyped-call]  # PyTorch 2.7 stub is untyped.
        with torch.cuda.graph(graph):
            costs = self._score(z_hist, a_hist, z_goal, plans)
        self._captured[rows] = (graph, z_hist, a_hist, z_goal, plans, costs)

    def costs(
        self,
        z_hist: torch.Tensor,
        a_hist: torch.Tensor,
        z_goal: torch.Tensor,
        plans: torch.Tensor,
    ) -> torch.Tensor:
        """Cost of every row of ``plans``; inputs already expanded to rows."""
        rows = plans.shape[0]
        if rows not in self._captured:
            self._capture(rows)
        graph, z_hist_s, a_hist_s, z_goal_s, plans_s, costs = self._captured[rows]
        z_hist_s.copy_(z_hist)
        a_hist_s.copy_(a_hist)
        z_goal_s.copy_(z_goal)
        plans_s.copy_(plans)
        graph.replay()  # type: ignore[no-untyped-call]  # PyTorch 2.7 stub is untyped.
        return costs.clone()
