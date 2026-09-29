"""Goal-reaching benchmark: draw start/goal tasks from a dataset and roll out a policy on them."""

from __future__ import annotations

import json
import os
import sys

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from stable_worldmodel.policy import BasePolicy
from torch import nn

from rp1.core.agent.policy import NoMovePolicy, PlanConfig, WorldModelPolicy
from rp1.core.agent.value import MetricCost, as_planning_cost
from rp1.core.world_model.featurize import image_transform
from rp1.data.base import Array, Dataset, episode_index
from rp1.environment import World
from rp1.training.harness.checkpointing import load_metric, load_pretrained
from rp1.utils.device import pick_device
from rp1.utils.logging import logger

type Transforms = dict[str, Callable[[object], torch.Tensor]]


def load_dataset(cfg: DictConfig, name: str) -> Dataset:
    dataset = swm.data.load_dataset(name, cache_dir=cfg.data.cache_dir, keys_to_cache=list(cfg.data.keys_to_cache))
    if not callable(getattr(dataset, "get_col_data", None)) or not hasattr(dataset, "column_names"):
        raise TypeError(f"dataset {name!r} does not provide the expected column interface")
    return cast(Dataset, dataset)


def fit_normalizers(cfg: DictConfig, dataset: Dataset) -> dict[str, Any]:
    """Z-score scalers for every cached non-pixel column, shared with its ``goal_`` twin.

    ``data.stats`` names the dataset the statistics come from; a checkpoint
    trained on another dataset than the evaluation pool is evaluated with its
    training statistics.
    """
    stats_name = cfg.data.stats or cfg.data.path
    stats_dataset = dataset if str(stats_name) == str(cfg.data.path) else load_dataset(cfg, stats_name)
    if stats_dataset is not dataset:
        logger.info(f"Using normalization stats from {stats_name}; evaluation tasks and goals from {cfg.data.path}")
    process: dict[str, Any] = {}
    for column in cfg.data.keys_to_cache:
        if column == "pixels":
            continue
        values = stats_dataset.get_col_data(column)
        process[column] = preprocessing.StandardScaler().fit(values[~np.isnan(values).any(axis=1)])
        if column != "action":
            process[f"goal_{column}"] = process[column]
    return process


def sample_tasks(cfg: DictConfig, dataset: Dataset) -> tuple[Array, Array]:
    """Episode ids and start steps of ``benchmark.num_episodes`` evaluation tasks.

    A start is valid when its goal, ``goal_offset_steps`` later, is still in
    the same episode; starts are drawn without replacement from the valid ones.
    """
    benchmark = cfg.benchmark
    row_episodes = episode_index(dataset)
    row_steps = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    episodes = np.unique(row_episodes)
    if benchmark.episode_range:
        low, high = map(int, str(benchmark.episode_range).split(":"))
        episodes = episodes[(episodes >= low) & (episodes < high)]
        if len(episodes) < benchmark.num_episodes:
            raise ValueError(
                f"episode range {benchmark.episode_range} has {len(episodes)} episodes; need {benchmark.num_episodes}"
            )
        logger.info(f"Restricted evaluation pool to episodes [{low}, {high}): {len(episodes)} eligible")

    last_start = np.full(row_episodes.shape, -1, dtype=np.int64)
    for episode in episodes:
        rows = row_episodes == episode
        last_start[rows] = np.max(row_steps[rows]) - benchmark.goal_offset_steps
    valid = np.nonzero((last_start >= 0) & (row_steps <= last_start))[0]
    logger.info(f"Found {len(valid)} valid evaluation starting points")

    if benchmark.cross_wall:
        valid = _cross_wall(dataset, valid, benchmark.goal_offset_steps, float(benchmark.wall_center))
    if len(valid) < benchmark.num_episodes:
        raise ValueError(f"Need {benchmark.num_episodes} valid evaluation starts; found {len(valid)}")
    # sorted, because the HDF5 reader requires increasing row indices
    rows = np.sort(np.random.default_rng(cfg.runtime.seed).choice(valid, size=benchmark.num_episodes, replace=False))
    logger.info(f"Selected evaluation row indices: {rows.tolist()}")
    return row_episodes[rows].astype(np.int64), row_steps[rows].astype(np.int64)


def _cross_wall(dataset: Dataset, starts: Array, offset: int, wall_center: float) -> Array:
    """TwoRoom starts whose goal lies on the other side of the dividing wall.

    The wall's axis is the one the first door does not sit on.
    """
    states = np.asarray(dataset.get_col_data("state"))
    first_door = np.asarray(dataset.get_col_data("observation"))[0, 4:6]
    axis = 0 if abs(float(first_door[0]) - wall_center) < 1e-3 else 1
    side_start = np.sign(states[starts, axis] - wall_center)
    side_goal = np.sign(states[starts + offset, axis] - wall_center)
    crossing = starts[side_start != side_goal]
    logger.info(f"Found {len(crossing)} cross-wall starting points on wall axis {axis}")
    return crossing


def load_world_model(cfg: DictConfig, device: str) -> nn.Module:
    model = load_pretrained(cfg.core.agent.policy.checkpoint)
    if cfg.runtime.bfloat16:
        model = model.to(torch.bfloat16)
    model = model.to(device).eval()
    model.requires_grad_(False)
    dynamic_model = cast(Any, model)  # Stable-WM exposes architecture-specific runtime attributes.
    dynamic_model.interpolate_pos_encoding = True
    if cfg.runtime.compile:
        encoder_name = "backbone" if hasattr(model, "backbone") else "encoder"
        setattr(model, encoder_name, torch.compile(getattr(model, encoder_name)))
        dynamic_model.predictor = torch.compile(dynamic_model.predictor)
    return model


def planning_cost(cfg: DictConfig, model: nn.Module, device: str) -> nn.Module:
    """The world model's latent goal cost, or a learned value on top of it."""
    cost = as_planning_cost(model)
    if not isinstance(cost, nn.Module):
        raise TypeError("planning cost must also be a torch module")
    value = cfg.core.agent.value
    if value.kind != "metric":
        return cost
    metrics = [load_metric(path, device=device) for path in value.checkpoints]
    logger.info(f"Plan-score metrics={list(value.checkpoints)} mode={value.mode}")
    return MetricCost(
        cost,
        metrics[0],
        value.mode,
        lam=float(value.blend_weight),
        metrics=metrics,
        deadline_mode=str(value.deadline_mode),
    )


def build_policy(cfg: DictConfig, device: str, process: dict[str, Any], transform: Transforms) -> BasePolicy:
    kind = cfg.core.agent.policy.kind
    if kind == "no_move":
        return NoMovePolicy()
    if kind == "random":
        return swm.policy.RandomPolicy()
    cost = planning_cost(cfg, load_world_model(cfg, device), device)
    solver = hydra.utils.instantiate(cfg.core.agent.solver, model=cost, device=device, seed=cfg.runtime.seed)
    # planning.budget bounds the episode, which the benchmark runs; the policy needs the rest
    fields = cast(dict[str, Any], OmegaConf.to_container(cfg.planning, resolve=True))
    planning = PlanConfig(**{name: value for name, value in fields.items() if name != "budget"})
    return WorldModelPolicy(solver=solver, config=planning, process=process, transform=transform)


def run(cfg: DictConfig) -> None:
    if cfg.planning.horizon * cfg.planning.action_block > cfg.planning.budget:
        raise ValueError("planning horizon x action block must not exceed the evaluation budget")
    device = pick_device(cfg.runtime.device)

    environment = cast(dict[str, Any], OmegaConf.to_container(cfg.environment, resolve=True))
    environment["max_episode_steps"] = 2 * cfg.planning.budget
    world = World(**environment, image_shape=(cfg.benchmark.image_size, cfg.benchmark.image_size))

    image_dtype = torch.bfloat16 if cfg.runtime.bfloat16 else torch.float32
    image = image_transform(cfg.benchmark.image_size, cfg.benchmark.train_resolution, image_dtype)
    dataset = load_dataset(cfg, cfg.data.path)
    process = fit_normalizers(cfg, dataset)
    policy = build_policy(cfg, device, process, {"pixels": image, "goal": image})
    world.set_policy(policy)
    episodes, starts = sample_tasks(cfg, dataset)
    # solvers that read the dataset task behind each environment, such as the oracle subgoal
    solver = getattr(policy, "solver", None)
    if hasattr(solver, "set_task_context"):
        solver.set_task_context(episodes.tolist(), starts.tolist())

    video_directory = Path(cfg.run.videos)
    logger.info(f"Saving evaluation videos to {video_directory}")
    callables = cast(list[dict[Any, Any]] | None, OmegaConf.to_container(cfg.benchmark.callables, resolve=True))

    def evaluate(count: int) -> dict[str, Any]:
        with torch.autocast(
            device_type=device if device != "mps" else "cpu", dtype=torch.bfloat16, enabled=cfg.runtime.bfloat16
        ):
            return world.evaluate(
                dataset=dataset,
                start_steps=starts.tolist()[:count],
                goal_offset=cfg.benchmark.goal_offset_steps,
                eval_budget=cfg.planning.budget,
                episodes_idx=episodes.tolist()[:count],
                callables=callables,
                video=video_directory,
            )

    if cfg.runtime.compile:
        logger.info("Warming up compiled model")
        evaluate(world.num_envs)
    start_time = time.time()
    metrics = evaluate(len(starts))
    elapsed = time.time() - start_time
    logger.info(f"Evaluation metrics: {metrics}")

    results_path = Path(cfg.run.metrics) / cfg.output.filename
    results_path.write_text(f"metrics: {metrics}\nevaluation_time_seconds: {elapsed}\n")
    summary = {
        "success_rate": float(metrics["success_rate"]),
        "episode_successes": np.asarray(metrics["episode_successes"]).astype(bool).tolist(),
        "evaluation_time_seconds": elapsed,
    }
    (Path(cfg.run.metrics) / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info(f"Evaluation results saved to {results_path}")


__all__ = ["build_policy", "fit_normalizers", "load_dataset", "planning_cost", "run", "sample_tasks"]
