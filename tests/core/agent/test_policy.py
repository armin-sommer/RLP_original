from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rp1.core.agent.policy import NoMovePolicy, PlanConfig, WorldModelPolicy


class _Space:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _Env:
    num_envs = 1
    action_space = _Space((1, 1))
    single_action_space = _Space((1,))


class _Solver:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.remaining: list[list[int]] = []

    def configure(self, **kwargs: Any) -> None:
        self.configuration = kwargs

    def set_align_remaining(self, remaining: list[int]) -> None:
        self.remaining.append(remaining)

    def __call__(self, info: dict[str, Any], *, init_action: torch.Tensor | None) -> dict[str, torch.Tensor]:
        self.calls.append({"info": info, "init_action": init_action})
        offset = 10 * len(self.calls)
        actions = torch.tensor([[[offset + 1.0, offset + 2.0], [offset + 3.0, offset + 4.0], [0.0, 0.0]]])
        return {"actions": actions}


def _info(frame: float) -> dict[str, np.ndarray]:
    return {
        "pixels": np.full((1, 1, 2), frame, dtype=np.float32),
        "proprio": np.full((1, 1, 1), frame, dtype=np.float32),
        "terminated": np.array([False]),
    }


def test_policy_keeps_real_history_and_updates_deadline_on_replan() -> None:
    solver = _Solver()
    config = PlanConfig(horizon=3, receding_horizon=1, history_len=3, action_block=2, warm_start=True, deadline=5)
    policy = WorldModelPolicy(solver=solver, config=config)
    policy.set_env(_Env())

    first = policy.get_action(_info(1.0))
    buffered = policy.get_action(_info(2.0))
    replanned = policy.get_action(_info(3.0))

    assert first.item() == 11.0
    assert buffered.item() == 12.0
    assert replanned.item() == 21.0
    assert solver.remaining == [[3], [2]]
    assert len(solver.calls) == 2

    first_history = solver.calls[0]["info"]["pixels_hist"]
    second_history = solver.calls[1]["info"]["pixels_hist"]
    assert torch.equal(first_history[0, :, 0], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(second_history[0, :, 0], torch.tensor([1.0, 1.0, 3.0]))
    assert torch.equal(solver.calls[0]["info"]["_align_remaining"], torch.tensor([3]))
    assert torch.equal(solver.calls[1]["info"]["_align_remaining"], torch.tensor([2]))
    assert solver.calls[1]["init_action"] is not None


def test_no_move_policy_zeroes_the_action() -> None:
    class ActionSpace:
        def sample(self) -> np.ndarray:
            return np.array([1.0, -2.0], dtype=np.float32)

    class Env:
        def __init__(self) -> None:
            self.action_space = ActionSpace()

    policy = NoMovePolicy()
    policy.set_env(Env())
    assert np.array_equal(policy.get_action({}), np.zeros(2, dtype=np.float32))
