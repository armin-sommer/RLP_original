from __future__ import annotations

import os
import sys
import sysconfig
from collections.abc import Callable
from pathlib import Path
from typing import cast

from hydra import compose, initialize_config_dir
from hydra.utils import call
from omegaconf import DictConfig, OmegaConf

from rp1.utils.logging import configured_logging, logger
from rp1.utils.run import RunMetadata, RunPaths, save_config


def get_config_root() -> Path:
    """The config tree: ``RP1_CONFIG_DIR``, the checkout's ``configs/``, or the installed copy."""
    candidates: list[Path] = []
    if override := os.environ.get("RP1_CONFIG_DIR"):
        candidates.append(Path(override))
    candidates.extend(
        [
            Path(__file__).resolve().parents[3] / "configs",
            Path(sysconfig.get_path("data")) / "share" / "rp1" / "configs",
        ]
    )
    for root in candidates:
        if root.is_dir():
            return root
    searched = ", ".join(map(str, candidates))
    raise FileNotFoundError(f"rp1 Hydra config directory not found; searched: {searched}")


def validate_config(cfg: DictConfig) -> None:
    missing = sorted(OmegaConf.missing_keys(cfg))
    if missing:
        raise ValueError(f"Missing required configuration values: {', '.join(missing)}")

    runtime = cfg.get("runtime")
    if runtime is not None and "seed" in runtime and int(runtime.seed) < 0:
        raise ValueError("runtime.seed must be non-negative")

    data = cfg.get("data")
    if data is not None and "train_split" in data and not 0.0 < float(data.train_split) < 1.0:
        raise ValueError("data.train_split must be between zero and one")

    logging = cfg.get("logging")
    if logging is not None and "wandb" in logging:
        mode = str(logging.wandb.mode)
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError(f"Unsupported logging.wandb.mode: {mode}")

    agent = cfg.get("core", {}).get("agent")
    if agent is not None and "policy" in agent:
        if agent.policy.kind not in {"random", "no_move", "world_model"}:
            raise ValueError(f"Unsupported core.agent.policy.kind: {agent.policy.kind}")
        if agent.policy.kind == "world_model" and not agent.policy.checkpoint:
            raise ValueError("core.agent.policy.checkpoint is required for a world-model policy")
    if agent is not None and "value" in agent and "kind" in agent.value:
        if agent.value.kind not in {"latent", "metric"}:
            raise ValueError(f"Unsupported core.agent.value.kind: {agent.value.kind}")
        if agent.value.kind == "metric" and not agent.value.checkpoints:
            raise ValueError("core.agent.value.checkpoints must not be empty for a metric value")

    benchmark = cfg.get("benchmark")
    if benchmark is not None:
        if int(benchmark.num_episodes) <= 0:
            raise ValueError("benchmark.num_episodes must be positive")
        episode_range = benchmark.get("episode_range")
        if episode_range:
            try:
                low, high = map(int, str(episode_range).split(":"))
            except ValueError as error:
                raise ValueError("benchmark.episode_range must use LO:HI syntax") from error
            if low < 0 or high <= low:
                raise ValueError("benchmark.episode_range must satisfy 0 <= LO < HI")

    planning = cfg.get("planning")
    if planning is not None:
        for key in ("horizon", "receding_horizon", "action_block"):
            if int(planning[key]) <= 0:
                raise ValueError(f"planning.{key} must be positive")
        if int(planning.budget) <= 0:
            raise ValueError("planning.budget must be positive")
        if int(planning.budget) < int(planning.receding_horizon):
            raise ValueError("planning.budget must be at least planning.receding_horizon")
        deadline = planning.get("deadline")
        if deadline is not None:
            if int(deadline) <= 0:
                raise ValueError("planning.deadline must be positive")
            if int(deadline) > int(planning.budget):
                raise ValueError("planning.deadline must fall within the planning budget")


def phase_config(cfg: DictConfig, phase: str, *components: DictConfig) -> DictConfig:
    """One flat mapping of the ``phase`` section, runtime, planning and ``components``.

    ``data``, ``environment``, ``output`` and ``run`` stay nested under their own keys.
    """
    sections: list[object] = [OmegaConf.to_container(cfg[phase], resolve=False)]
    for name in ("runtime", "planning"):
        if name in cfg:
            sections.append(OmegaConf.to_container(cfg[name], resolve=False))
    sections.extend(OmegaConf.to_container(component, resolve=False) for component in components)
    args = OmegaConf.merge(*sections)
    if not isinstance(args, DictConfig):
        raise TypeError(f"{phase} configuration must be a mapping")
    for name in ("data", "environment", "output", "run"):
        if name in cfg:
            args[name] = OmegaConf.to_container(cfg[name], resolve=False)
    return args


def dispatch(cfg: DictConfig) -> object:
    validate_config(cfg)
    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    return cast(object, call(cfg.entrypoint, cfg=cfg, _recursive_=False))


def _split_config_name(arguments: list[str], default: str) -> tuple[str, list[str]]:
    """Take Hydra's ``--config-name``/``-cn`` flag out of ``arguments``."""
    name, overrides = default, []
    iterator = iter(arguments)
    for argument in iterator:
        if argument in ("--config-name", "-cn"):
            name = next(iterator, "")
        elif argument.startswith("--config-name="):
            name = argument.split("=", 1)[1]
        else:
            overrides.append(argument)
    if not name:
        raise SystemExit("--config-name needs a value")
    return name, overrides


def compose_config(config_dir: Path, config_name: str, overrides: list[str]) -> DictConfig:
    """Compose ``config_name`` from ``config_dir`` with the whole config tree on the search path.

    Groups under ``config_dir`` are selected by their short name (``benchmark=``,
    ``job=``); everything else is reachable by its absolute path (``/core/...``).
    """
    root = get_config_root()
    with initialize_config_dir(config_dir=str(root / config_dir), version_base=None):
        return compose(config_name=config_name, overrides=[*overrides, f"hydra.searchpath=[file://{root}]"])


def run_hydra[ResultT](task: Callable[[DictConfig], ResultT], *, config_dir: str, config_name: str) -> ResultT:
    """Compose the command-line config, then run ``task`` inside a fresh run directory."""
    config_name, overrides = _split_config_name(list(sys.argv[1:]), config_name)
    if config_name.startswith("/"):  # a config outside config_dir, such as /methods/rp1/train
        config_dir, config_name = str(Path(config_name).parent).lstrip("/"), Path(config_name).name
    cfg = compose_config(Path(config_dir), config_name, overrides)
    paths = RunPaths.create(cfg.logging.run_root)
    paths.attach(cfg)
    level = str(cfg.logging.level)
    with configured_logging(paths.log, level):
        logger.info(f"Run directory: {paths.directory}")
        metadata = None
        try:
            metadata = RunMetadata(paths, cfg)
            save_config(cfg, paths.config)
            result = task(cfg)
        except KeyboardInterrupt:
            logger.warning("Run interrupted")
            if metadata is not None:
                metadata.finish("interrupted", "KeyboardInterrupt")
            raise SystemExit(130) from None
        except BaseException as error:  # noqa: BLE001 - command boundary records every failed lifecycle
            logger.opt(exception=error).error("Run failed")
            if metadata is not None:
                metadata.finish("failed", f"{type(error).__name__}: {error}")
            raise SystemExit(1) from None
        assert metadata is not None
        metadata.finish("succeeded")
        logger.info("Run completed")
        return result


__all__ = ["compose_config", "dispatch", "get_config_root", "phase_config", "run_hydra", "validate_config"]
