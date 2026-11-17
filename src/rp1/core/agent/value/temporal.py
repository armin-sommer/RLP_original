"""Temporal objectives that score an imagined trajectory, shared by training and deployment."""

from typing import Protocol

import torch


class ValueFunction(Protocol):
    def __call__(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor: ...


def trajectory_value(
    value: ValueFunction,
    trajectory: torch.Tensor,
    goal: torch.Tensor,
    start: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Score an imagined trajectory under a supported temporal objective."""
    terminal = value(trajectory[:, -1], goal)
    if mode == "terminal":
        return terminal
    initial = value(start, goal)
    if mode == "tel-exact":
        return terminal - initial
    if mode != "tel-stopprev":
        raise ValueError(f"unsupported temporal objective: {mode}")
    horizon = trajectory.shape[1]
    values = value(
        trajectory.reshape(-1, trajectory.shape[-1]),
        goal.repeat_interleave(horizon, dim=0),
    ).view(trajectory.shape[0], horizon)
    previous = torch.cat([initial[:, None], values[:, :-1]], dim=1).detach()
    return (values - previous).sum(dim=1)


def window_pair(
    trajectory: torch.Tensor,
    goal: torch.Tensor,
    frames: int,
    start: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(stacked_state, tiled_goal)`` for an m-frame window value.

    ``trajectory`` is ``(B, H, D)`` latents, newest last. Fewer than ``frames``
    available repeats the oldest (the ``LatentCache.windowed`` clamp at episode
    starts). The goal is a single observed frame, so it is duplicated
    ``frames`` times — principled because the goal is static: duplicated goal
    frames encode zero velocity at the goal. ``frames == 1`` returns the
    terminal frame unchanged, so single-frame values are untouched.
    """
    if frames <= 1:
        return trajectory[:, -1], goal
    columns = []
    for k in range(frames - 1, -1, -1):
        index = trajectory.shape[1] - 1 - k
        if index >= 0:
            columns.append(trajectory[:, index])
        elif trajectory.shape[1]:
            columns.append(trajectory[:, 0])
        else:
            if start is None:
                raise ValueError("empty trajectory requires an explicit start frame")
            columns.append(start)
    return torch.cat(columns, dim=-1), goal.repeat(*([1] * (goal.dim() - 1)), frames)


def windowed_trajectory_value(
    value: ValueFunction,
    trajectory: torch.Tensor,
    goal: torch.Tensor,
    history: torch.Tensor,
    frames: int,
    mode: str,
) -> torch.Tensor:
    """Score an imagined trajectory with an m-frame window value.

    The terminal score stacks the last ``frames`` imagined latents and tiles
    the goal (:func:`window_pair`). The telescoped objectives window the real
    ``history`` (``(B, C, D)``, one action block apart) for the starting value,
    and — for ``tel-stopprev`` — every per-step window fills from
    ``[history, imagined[: t + 1]]``, exactly as the window fills at deploy.
    ``frames == 1`` callers should use :func:`trajectory_value` instead.
    """
    state, tiled_goal = window_pair(trajectory, goal, frames)
    terminal = value(state, tiled_goal)
    if mode == "terminal":
        return terminal
    initial = value(window_pair(history, goal, frames)[0], tiled_goal)
    if mode == "tel-exact":
        return terminal - initial
    if mode != "tel-stopprev":
        raise ValueError(f"unsupported temporal objective: {mode}")
    merged = torch.cat([history, trajectory], dim=1)
    context, horizon = history.shape[1], trajectory.shape[1]
    values = torch.stack(
        [value(*window_pair(merged[:, : context + t + 1], goal, frames)) for t in range(horizon)],
        dim=1,
    )
    previous = torch.cat([initial[:, None], values[:, :-1]], dim=1).detach()
    return (values - previous).sum(dim=1)


def windowed_terminal_value(
    value: ValueFunction,
    trajectory: torch.Tensor,
    goal: torch.Tensor,
    context: int,
) -> torch.Tensor:
    """Terminal value of a ``context``-frame window, as ``MetricCost`` forms it.

    Window values take ``context * D`` inputs: the last ``context`` imagined
    frames concatenated oldest first and the goal tiled to match (:func:`window_pair`).
    """
    if context < 1:
        raise ValueError(f"invalid value context width: {context}")
    return value(*window_pair(trajectory, goal, context))


__all__ = [
    "ValueFunction",
    "trajectory_value",
    "window_pair",
    "windowed_terminal_value",
    "windowed_trajectory_value",
]
