"""Checkpoint selection for one experimental cell: ``pixi run select selection.runs=[<dir>, ...]``.

A cell trains one planner per (training seed, teacher snapshot) and keeps a snapshot
every few thousand planner steps, so every (teacher, planner step) pair is a candidate.
Selection reads every evaluation run under ``selection.runs``: the run's resolved config
names the planner checkpoint and the evaluation seed (the draw), ``metrics/metrics.json``
holds the success rate, and the checkpoint records its training seed, step and teacher.

A pair scores its mean success over every training seed and selection draw; ties go to
the smaller teacher budget, then to the earlier planner step. The chosen pair is reported
as the median over training seeds of each seed's mean over the report draws.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from rp1.utils.config import dispatch, run_hydra
from rp1.utils.logging import logger

# (teacher training steps, planner training step)
type Pair = tuple[float, int]


@dataclass(frozen=True)
class Evaluation:
    seed: int
    teacher: float
    step: int
    draw: int
    success: float

    @property
    def pair(self) -> Pair:
        return self.teacher, self.step


def teacher_budget(teacher: str | None) -> float:
    """Training steps of a teacher snapshot (``value_td_step<N>``); the final value, at the step cap, sorts last."""
    match = re.search(r"_step(\d+)$", Path(teacher).name) if teacher else None
    return float(match.group(1)) if match else math.inf


def collect(roots: Iterable[Path]) -> list[Evaluation]:
    """Every finished planner evaluation under ``roots``."""
    evaluations = []
    for root in roots:
        for metrics_path in sorted(Path(root).rglob("metrics/metrics.json")):
            run = metrics_path.parent.parent
            cfg = OmegaConf.load(run / "config.yaml")
            checkpoint = OmegaConf.select(cfg, "core.agent.solver.checkpoint.path")
            if checkpoint is None:
                continue
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if "step" not in payload:
                logger.warning(f"{checkpoint} records no training step; skipped")
                continue
            evaluations.append(
                Evaluation(
                    seed=int(payload["seed"]),
                    teacher=teacher_budget(payload["teacher"]),
                    step=int(payload["step"]),
                    draw=int(cfg.runtime.seed),
                    success=float(json.loads(metrics_path.read_text())["success_rate"]),
                )
            )
    return evaluations


def select(evaluations: Sequence[Evaluation], draws: Sequence[int]) -> tuple[Pair, dict[Pair, float]]:
    """The best pair on ``draws`` and every complete pair's score.

    A pair is complete when every training seed has an evaluation on every draw; the
    others are left out, since their means are not comparable.
    """
    seeds = {evaluation.seed for evaluation in evaluations}
    expected = {(seed, draw) for seed in seeds for draw in draws}
    runs: dict[Pair, dict[tuple[int, int], float]] = defaultdict(dict)
    for evaluation in evaluations:
        if evaluation.draw in draws:
            runs[evaluation.pair][(evaluation.seed, evaluation.draw)] = evaluation.success
    scores = {}
    for pair, results in runs.items():
        if set(results) != expected:
            logger.warning(f"pair {pair} has {len(results)} of {len(expected)} selection evaluations; left out")
            continue
        scores[pair] = statistics.fmean(results.values())
    if not scores:
        raise ValueError(f"no pair has evaluations for every seed {sorted(seeds)} on draws {list(draws)}")
    best = max(scores, key=lambda pair: (scores[pair], -pair[0], -pair[1]))
    return best, scores


def report(evaluations: Sequence[Evaluation], pair: Pair, draws: Sequence[int]) -> dict[int, float]:
    """Each training seed's mean success of ``pair`` over ``draws``."""
    by_seed: dict[int, dict[int, float]] = defaultdict(dict)
    for evaluation in evaluations:
        if evaluation.pair == pair and evaluation.draw in draws:
            by_seed[evaluation.seed][evaluation.draw] = evaluation.success
    missing = {
        seed: sorted(set(draws) - set(results)) for seed, results in by_seed.items() if set(results) != set(draws)
    }
    if not by_seed or missing:
        raise ValueError(f"pair {pair} lacks report evaluations: {missing or 'none found'}")
    return {seed: statistics.fmean(results.values()) for seed, results in sorted(by_seed.items())}


def _label(pair: Pair) -> str:
    teacher = "final" if math.isinf(pair[0]) else str(int(pair[0]))
    return f"teacher {teacher} / planner {pair[1]}"


def run(cfg: DictConfig) -> dict[str, object]:
    evaluations = collect(Path(str(root)).expanduser() for root in cfg.selection.runs)
    logger.info(f"{len(evaluations)} planner evaluations")
    best, scores = select(evaluations, list(cfg.selection.selection_draws))
    for pair in sorted(scores, key=lambda pair: -scores[pair]):
        logger.info(f"{_label(pair):<32} {scores[pair]:6.2f}{'  <- selected' if pair == best else ''}")
    per_seed = report(evaluations, best, list(cfg.selection.report_draws))
    summary: dict[str, object] = {
        "teacher_step": None if math.isinf(best[0]) else int(best[0]),
        "planner_step": best[1],
        "selection_score": scores[best],
        "report_per_seed": per_seed,
        "report_median": statistics.median(per_seed.values()),
        "report_mean": statistics.fmean(per_seed.values()),
    }
    (Path(cfg.run.metrics) / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.success(
        f"Selected {_label(best)}: report median {summary['report_median']:.1f} "
        f"over seeds {sorted(per_seed)} (mean {summary['report_mean']:.1f})"
    )
    return summary


def main() -> object:
    return run_hydra(dispatch, config_dir="inference", config_name="selection")


if __name__ == "__main__":
    main()


__all__ = ["Evaluation", "collect", "report", "run", "select", "teacher_budget"]
