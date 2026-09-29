from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from gymnasium.spaces import Box

from rp1.methods.rp1.solver import RP1Solver
from rp1.training.harness.checkpointing import load_planner, load_wm


def _solver(world_model: Path, planner: Path) -> RP1Solver:
    solver = RP1Solver(
        model=load_wm(str(world_model)),
        checkpoint=load_planner(str(planner), None),
        graphed=False,
        graph_warmup_iters=5,
        device="cpu",
        seed=0,
    )
    solver.configure(
        action_space=Box(-1, 1, shape=(2, 5)),
        n_envs=2,
        config=SimpleNamespace(horizon=5, action_block=5, receding_horizon=5),
    )
    return solver


def _observation() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "pixels": torch.randn(2, 3, 3, 224, 224, generator=generator),
        "goal": torch.randn(2, 1, 3, 224, 224, generator=generator),
    }


def test_plans_are_deterministic_and_inside_the_action_limit(cube_world_model: Path, cube_planner: Path) -> None:
    solver = _solver(cube_world_model, cube_planner)
    first = solver.solve(_observation())["actions"]
    second = solver.solve(_observation())["actions"]
    assert first.shape == (2, 5, 25)
    assert torch.equal(first, second)
    assert first.abs().max() <= solver.planner.action_limit


def test_zero_iterations_emit_the_zero_plan(cube_world_model: Path, cube_planner: Path) -> None:
    solver = _solver(cube_world_model, cube_planner)
    solver.iterations = 0
    assert torch.count_nonzero(solver.solve(_observation())["actions"]) == 0
