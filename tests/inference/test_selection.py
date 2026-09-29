from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from rp1.inference.selection import Evaluation, collect, report, select, teacher_budget


def _evaluation(root: Path, *, seed: int, teacher: str | None, step: int, draw: int, success: float) -> None:
    run = root / f"s{seed}_{Path(teacher or 'none').name}_p{step}_d{draw}"
    (run / "metrics").mkdir(parents=True)
    checkpoint = run / "planner.pt"
    torch.save({"seed": seed, "step": step, "teacher": teacher}, checkpoint)
    OmegaConf.save(
        OmegaConf.create(
            {"runtime": {"seed": draw}, "core": {"agent": {"solver": {"checkpoint": {"path": str(checkpoint)}}}}}
        ),
        run / "config.yaml",
    )
    (run / "metrics" / "metrics.json").write_text(json.dumps({"success_rate": success}))


def test_a_teacher_budget_is_read_off_the_snapshot_name() -> None:
    assert teacher_budget("/runs/checkpoints/value_td_step3000") == 3000
    assert math.isinf(teacher_budget("/runs/checkpoints/value_td"))


def test_evaluation_runs_are_collected_with_their_grid_position(tmp_path: Path) -> None:
    _evaluation(tmp_path, seed=1, teacher="value_td_step6000", step=4000, draw=48, success=80.0)
    (evaluation,) = collect([tmp_path])
    assert evaluation == Evaluation(seed=1, teacher=6000, step=4000, draw=48, success=80.0)


def test_the_best_pair_wins_and_ties_go_to_the_smaller_budget() -> None:
    evaluations = [
        Evaluation(seed, teacher, step, draw, success)
        for seed in (0, 1)
        for draw in (48, 49)
        for teacher, step, success in ((3000, 4000, 70.0), (6000, 2000, 90.0), (9000, 2000, 90.0), (6000, 4000, 90.0))
    ]
    best, scores = select(evaluations, [48, 49])
    assert best == (6000, 2000)
    assert scores[(3000, 4000)] == 70.0


def test_a_pair_missing_a_seed_is_left_out() -> None:
    evaluations = [Evaluation(0, 3000, 2000, 48, 50.0), Evaluation(1, 3000, 2000, 48, 50.0)]
    evaluations += [Evaluation(0, 6000, 2000, 48, 100.0)]
    best, scores = select(evaluations, [48])
    assert best == (3000, 2000)
    assert (6000, 2000) not in scores


def test_the_report_is_each_seeds_mean_over_the_report_draws() -> None:
    evaluations = [
        Evaluation(seed, 3000, 2000, draw, 10.0 * seed + draw - 42) for seed in (0, 1, 2) for draw in (42, 43, 44)
    ]
    assert report(evaluations, (3000, 2000), [42, 43, 44]) == {0: 1.0, 1: 11.0, 2: 21.0}
    with pytest.raises(ValueError, match="lacks report evaluations"):
        report(evaluations[:-1], (3000, 2000), [42, 43, 44])
