from __future__ import annotations

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rp1.utils.config import compose_config, dispatch, get_config_root, run_hydra, validate_config

CONFIG_ROOT = get_config_root()


def _options(group: str) -> list[str]:
    return [path.stem for path in sorted((CONFIG_ROOT / group).glob("*.yaml"))]


def _entrypoints() -> list[tuple[str, str, list[str]]]:
    commands = [("training", "pretrain", []), ("training", "posttrain", [])]
    commands += [("training", f"phases/agent/{name}", []) for name in _options("training/phases/agent")]
    commands += [("inference", "evaluate", [f"benchmark={name}"]) for name in _options("inference/benchmark")]
    commands += [("training/data", "prepare", [f"job={name}"]) for name in _options("training/data/job")]
    for group in ("solver", "policy", "value"):
        options = _options(f"core/agent/{group}")
        commands += [("inference", "evaluate", [f"core/agent/{group}={name}"]) for name in options]
    return commands


@pytest.mark.parametrize(("config_dir", "config_name", "overrides"), _entrypoints())
def test_every_command_composes(config_dir: str, config_name: str, overrides: list[str]) -> None:
    config = compose_config(Path(config_dir), config_name, overrides)
    assert config.entrypoint._target_.startswith("rp1.")


@pytest.mark.parametrize("benchmark", ["cube_lewm", "reacher", "tworoom"])
def test_benchmarks_need_no_further_values(benchmark: str) -> None:
    config = compose_config(Path("inference"), "evaluate", [f"benchmark={benchmark}"])
    assert config.environment.env_name
    assert not OmegaConf.missing_keys(config)


def test_pretraining_needs_no_further_values() -> None:
    config = compose_config(Path("training"), "pretrain", [])
    assert not OmegaConf.missing_keys(config)


def test_config_root_can_be_overridden(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RP1_CONFIG_DIR", str(tmp_path))
    assert get_config_root() == tmp_path


def test_run_hydra_records_the_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["pretrain", "runtime.seed=7"])
    config = run_hydra(lambda cfg: cfg, config_dir="training", config_name="pretrain")
    assert config.core.world_model.name == "lewm"
    assert config.runtime.seed == 7
    run_directory = Path(config.run.directory)
    assert run_directory.parent.parent == tmp_path / "logs"
    assert (run_directory / "config.yaml").is_file()
    assert (run_directory / "metadata.json").is_file()
    assert (run_directory / "run.log").is_file()


def test_run_hydra_selects_a_group_option(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    arguments = ["prepare", "job=cache_latents", "preparation.wm=model.pt", "preparation.dataset=data.lance"]
    monkeypatch.setattr(sys, "argv", arguments)
    config = run_hydra(lambda cfg: cfg, config_dir="training/data", config_name="prepare")
    assert config.entrypoint._target_ == "rp1.training.data.job.cache_latents.run"
    assert config.preparation.wm == "model.pt"


def test_run_hydra_takes_a_config_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["posttrain", "--config-name", "phases/agent/metric"])
    config = run_hydra(lambda cfg: cfg, config_dir="training", config_name="posttrain")
    assert config.entrypoint._target_ == "rp1.training.phases.agent.metric.run"


def test_run_hydra_records_validation_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["posttrain", "-cn", "phases/agent/metric"])
    with pytest.raises(SystemExit) as failure:
        run_hydra(dispatch, config_dir="training", config_name="posttrain")

    assert failure.value.code == 1
    run_directory = next((tmp_path / "logs").glob("*/*"))
    metadata = OmegaConf.load(run_directory / "metadata.json")
    assert metadata.status == "failed"
    assert "Missing required configuration" in metadata.error


def test_pipeline_caches_default_outside_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RP1_DATA_HOME", str(tmp_path))
    config = compose_config(Path("training"), "posttrain", [])
    assert config.training.cache_directory == str(tmp_path / "caches")


def test_config_validation_rejects_invalid_cross_field_values() -> None:
    config = compose_config(Path("inference"), "evaluate", ["planning.budget=4"])
    with pytest.raises(ValueError, match="receding_horizon"):
        validate_config(config)


def test_wandb_mode_is_validated() -> None:
    offline = compose_config(Path("training"), "pretrain", ["logging.wandb.mode=offline"])
    invalid = compose_config(Path("training"), "pretrain", ["logging.wandb.mode=invalid"])
    validate_config(offline)
    with pytest.raises(ValueError, match="logging.wandb.mode"):
        validate_config(invalid)


def test_hydra_configs_compose() -> None:
    cfg = compose_config(Path("inference"), "evaluate", ["core/agent/solver=adam"])
    assert cfg.environment.env_name == "swm/OGBCube-v0"
    assert cfg.core.agent.solver._target_ == "rp1.core.agent.solver.GradientSolver"

    cfg = compose_config(Path("training"), "pretrain", ["data=tworoom_lewm"])
    assert cfg.core.world_model.architecture._target_ == "stable_worldmodel.wm.lewm.LeWM"


def test_a_job_without_a_seed_validates() -> None:
    validate_config(OmegaConf.create({"runtime": {"device": "cpu"}}))
