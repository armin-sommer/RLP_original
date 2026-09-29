"""Policies that act in the evaluation environments."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from stable_worldmodel.policy import BasePolicy
from stable_worldmodel.protocols import Transformable


@dataclass(frozen=True)
class PlanConfig:
    """Planning-loop configuration, including history length and deadline."""

    horizon: int
    receding_horizon: int
    history_len: int
    action_block: int
    warm_start: bool
    deadline: int | None

    @property
    def plan_len(self) -> int:
        return self.horizon * self.action_block


class NoMovePolicy(BasePolicy):
    """Emit a zero action: the success floor without planning."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.type = "nomove"

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        return np.zeros_like(self.env.action_space.sample())


class WorldModelPolicy(BasePolicy):
    """World-model policy with real history and deadline-aware replanning.

    Stable-WM 0.1.1 pads a single current observation when a three-frame model
    is evaluated. This policy keeps frames and executed action blocks at their
    training cadence, and publishes the number of reachable plan chunks to
    solvers that score against a deadline.
    """

    def __init__(
        self,
        solver: Any,
        config: PlanConfig,
        process: dict[str, Transformable] | None = None,
        transform: dict[str, Callable[[object], torch.Tensor]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.type = "world_model"
        self.cfg = config
        self.solver = solver
        self.process = process or {}
        self.transform = transform or {}
        self._action_buffer: list[deque[torch.Tensor]] = []
        self._next_init: torch.Tensor | None = None
        self._history_announced = False

    @property
    def flatten_receding_horizon(self) -> int:
        return self.cfg.receding_horizon * self.cfg.action_block

    def set_env(self, env: Any) -> None:
        self.env = env
        n_envs = int(getattr(env, "num_envs", 1))
        self.solver.configure(action_space=env.action_space, n_envs=n_envs, config=self.cfg)
        self._action_buffer = [deque(maxlen=self.flatten_receding_horizon) for _ in range(n_envs)]
        history = max(int(self.cfg.history_len), 1)
        self._frame_history: list[deque[torch.Tensor]] = [deque(maxlen=history) for _ in range(n_envs)]
        self._proprio_history: list[deque[torch.Tensor]] = [deque(maxlen=history) for _ in range(n_envs)]
        primitive_history = max(history - 1, 1) * self.cfg.action_block
        self._primitive_history: list[deque[torch.Tensor]] = [deque(maxlen=primitive_history) for _ in range(n_envs)]
        self._step_count = np.zeros(n_envs, dtype=np.int64)
        self._next_init = None

    @staticmethod
    def _stack_history(items: deque[torch.Tensor], length: int) -> torch.Tensor | None:
        values = list(items)
        if not values:
            return None
        while len(values) < length:
            values.insert(0, values[0])
        return torch.stack(values)

    def _flush(self, needs_flush: Any, n_envs: int) -> None:
        if needs_flush is None:
            return
        flags = np.asarray(needs_flush, dtype=bool)
        for index in range(n_envs):
            if not flags[index]:
                continue
            self._action_buffer[index].clear()
            self._frame_history[index].clear()
            self._proprio_history[index].clear()
            self._primitive_history[index].clear()
            self._step_count[index] = 0
            if self._next_init is not None:
                self._next_init[index] = 0

    def _attach_history(self, sliced: dict[str, Any], replan: list[int]) -> None:
        history = max(int(self.cfg.history_len), 1)
        if history <= 1:
            return
        frames = [self._stack_history(self._frame_history[index], history) for index in replan]
        if all(frame is not None for frame in frames):
            sliced["pixels_hist"] = torch.stack([frame for frame in frames if frame is not None])
        proprio = [self._stack_history(self._proprio_history[index], history) for index in replan]
        if all(item is not None for item in proprio):
            sliced["proprio_hist"] = torch.stack([item for item in proprio if item is not None])

        blocks = history - 1
        action_history: list[torch.Tensor] = []
        for index in replan:
            values = list(self._primitive_history[index])
            wanted = blocks * self.cfg.action_block
            if not values:
                continue
            while len(values) < wanted:
                values.insert(0, torch.zeros_like(values[0]))
            action_history.append(torch.stack(values[-wanted:]).reshape(blocks, -1))
        if len(action_history) == len(replan):
            sliced["action_hist"] = torch.stack(action_history)
        if "pixels_hist" in sliced and not self._history_announced:
            self._history_announced = True

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        del kwargs
        prepared = self._prepare_info(obs)
        n_envs = int(self.env.num_envs)
        self._flush(prepared.pop("_needs_flush", None), n_envs)

        history = max(int(self.cfg.history_len), 1)
        if history > 1:
            pixels = prepared.get("pixels")
            proprio = prepared.get("proprio")
            for index in range(n_envs):
                if self._step_count[index] % self.cfg.action_block:
                    continue
                if torch.is_tensor(pixels):
                    self._frame_history[index].append(pixels[index, -1])
                if torch.is_tensor(proprio):
                    self._proprio_history[index].append(proprio[index, -1])

        terminated = prepared.get("terminated")
        dead = np.asarray(terminated, dtype=bool) if terminated is not None else np.zeros(n_envs, dtype=bool)
        replan = [index for index in range(n_envs) if not self._action_buffer[index] and not dead[index]]

        if replan:
            index_tensor = torch.as_tensor(replan, dtype=torch.long)
            sliced: dict[str, Any] = {}
            for key, value in prepared.items():
                if torch.is_tensor(value):
                    sliced[key] = value[index_tensor]
                elif isinstance(value, np.ndarray):
                    sliced[key] = value[replan]
                elif isinstance(value, list):
                    sliced[key] = [value[index] for index in replan]
                else:
                    sliced[key] = value
            self._attach_history(sliced, replan)

            if self.cfg.deadline is not None and hasattr(self.solver, "set_align_remaining"):
                block = self.cfg.action_block
                remaining = [
                    max(1, (self.cfg.deadline - int(self._step_count[index]) + block - 1) // block) for index in replan
                ]
                sliced["_align_remaining"] = torch.as_tensor(remaining, dtype=torch.long)
                self.solver.set_align_remaining(remaining)

            initial = self._next_init[index_tensor] if self._next_init is not None else None
            outputs = self.solver(sliced, init_action=initial)
            actions = torch.as_tensor(outputs["actions"])
            keep = self.cfg.receding_horizon
            plan, rest = actions[:, :keep], actions[:, keep:]
            if self.cfg.warm_start and rest.shape[1] > 0:
                if self._next_init is None:
                    self._next_init = torch.zeros(n_envs, rest.shape[1], rest.shape[2], dtype=rest.dtype)
                self._next_init[index_tensor] = rest
            elif not self.cfg.warm_start:
                self._next_init = None
            plan = plan.reshape(len(replan), self.flatten_receding_horizon, -1)
            for row, env_index in enumerate(replan):
                self._action_buffer[env_index].extend(plan[row])

        single_shape = self.env.single_action_space.shape
        discrete = "Discrete" in type(self.env.single_action_space).__name__
        action = torch.full(
            (n_envs, *single_shape),
            fill_value=0 if discrete else float("nan"),
            dtype=torch.long if discrete else torch.float32,
        )
        for index in range(n_envs):
            if dead[index]:
                continue
            action[index] = self._action_buffer[index].popleft()
            if history > 1:
                self._primitive_history[index].append(action[index].clone())
            self._step_count[index] += 1
        result = action.reshape(*self.env.action_space.shape).numpy()
        if "action" in self.process:
            result = self.process["action"].inverse_transform(result)
        return result


__all__ = ["NoMovePolicy", "PlanConfig", "WorldModelPolicy"]
