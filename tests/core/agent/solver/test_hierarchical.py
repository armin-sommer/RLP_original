from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from gymnasium.spaces import Box

from rp1.core.agent.solver import HierarchicalCEMSolver
from rp1.core.world_model.hierarchical import HWM, save_hwm
from rp1.training.harness.checkpointing import load_wm


def _solver(world_model: Path, hwm: Path, **overrides: Any) -> HierarchicalCEMSolver:
    settings: dict[str, Any] = {
        "hl_dynamics": "f2",
        "hwm_path": str(hwm),
        "bank_path": None,
        "hl_value": None,
        "hl_cost": "latent",
        "hl_opt": "cem",
        "hl_lr": 0.1,
        "hl_adam_steps": 2,
        "hl_horizon": 2,
        "hl_samples": 8,
        "hl_iters": 2,
        "hl_topk": 0,
        "hl_amax": 3.0,
        "ll_samples": None,
        "ll_iters": None,
        "ll_topk": 0,
        "ll_amax": 3.5,
        "ll_clamp": False,
        "subgoal_index": 0,
        "oracle_subgoal": False,
        "cache_path": None,
    }
    solver = HierarchicalCEMSolver(
        model=load_wm(str(world_model)),
        batch_size=2,
        num_samples=8,
        var_scale=1.0,
        n_steps=2,
        topk=2,
        device="cpu",
        seed=0,
        **{**settings, **overrides},
    )
    solver.configure(
        action_space=Box(-1, 1, shape=(2, 5)),
        n_envs=2,
        config=SimpleNamespace(horizon=5, action_block=5, receding_horizon=5),
    )
    return solver


def _hwm(tmp_path: Path) -> Path:
    model = HWM(macro_dim=4, depth=1, heads=2, mlp_dim=32, dim_head=8, chunk_len=25, act_dim=5)
    config = {"macro_dim": 4, "depth": 1, "heads": 2, "mlp_dim": 32, "dim_head": 8, "stride": 25, "act_dim": 5}
    return save_hwm(model, tmp_path / "hwm.pt", config, {})


def _observation() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "pixels": torch.randn(2, 1, 3, 224, 224, generator=generator),
        "goal": torch.randn(2, 1, 3, 224, 224, generator=generator),
    }


def test_both_high_level_optimizers_emit_a_primitive_plan(cube_world_model: Path, tmp_path: Path) -> None:
    hwm = _hwm(tmp_path)
    for optimizer in ("cem", "adam"):
        actions = _solver(cube_world_model, hwm, hl_opt=optimizer).solve(_observation())["actions"]
        assert actions.shape == (2, 5, 25)
        assert torch.isfinite(actions).all()
