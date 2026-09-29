"""Contracts for canonical run directories, metadata, and structured logs."""

from __future__ import annotations

import json
import re
import sys
import warnings
from datetime import datetime
from io import StringIO
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rp1.utils.logging import configured_logging, logger
from rp1.utils.run import RunMetadata, RunPaths, save_config


def test_run_paths_create_complete_collision_safe_layout(tmp_path: Path) -> None:
    moment = datetime.fromisoformat("2026-08-03T21:36:21-07:00")
    first = RunPaths.create(tmp_path / "logs", now=moment)
    second = RunPaths.create(tmp_path / "logs", now=moment)

    assert first.directory == (tmp_path / "logs" / "2026-08-03" / "21-36-21").resolve()
    assert second.directory.name == "21-36-21-02"
    for name in ("checkpoints", "metrics", "videos", "artifacts", "tracking", "stages"):
        assert getattr(first, name).is_dir()


def test_run_config_and_metadata_are_reproducible(tmp_path: Path) -> None:
    paths = RunPaths.create(tmp_path / "logs")
    config = OmegaConf.create({"entrypoint": {"_target_": "tests.fake"}, "value": 7})
    paths.attach(config)
    save_config(config, paths.config)
    metadata = RunMetadata(paths, config)
    metadata.finish("succeeded")

    saved_config = OmegaConf.load(paths.config)
    saved_metadata = json.loads(paths.metadata.read_text())
    assert saved_config.run.directory == str(paths.directory)
    assert saved_metadata["status"] == "succeeded"
    assert saved_metadata["entrypoint"] == "tests.fake"
    assert saved_metadata["hostname"]
    assert saved_metadata["packages"]["stable-worldmodel"]
    assert saved_metadata["finished_at"] is not None
    assert saved_metadata["duration_seconds"] >= 0


def test_every_physical_log_line_has_a_timestamp_and_level(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    with configured_logging(log_path, "INFO"):
        logger.info("first\nsecond")
        print("captured stdout")
        sys.stderr.write("captured stderr\n")
        try:
            raise RuntimeError("broken")
        except RuntimeError:
            logger.exception("example failure")

    lines = log_path.read_text().splitlines()
    prefix = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[[A-Z]+\] ")
    assert len(lines) >= 7
    assert all(prefix.match(line) for line in lines)
    assert any("captured stdout" in line and "[INFO]" in line for line in lines)
    assert any("captured stderr" in line and "[ERROR]" in line for line in lines)


def test_handler_failure_does_not_reenter_loguru(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = StringIO()
    monkeypatch.setattr(sys, "__stderr__", terminal)

    def broken_sink(message: object) -> None:
        del message
        raise RuntimeError("broken sink")

    with configured_logging(tmp_path / "run.log", "INFO"):
        handler_id = logger.add(broken_sink, catch=True)
        try:
            logger.error("trigger handler failure")
        finally:
            logger.remove(handler_id)

    diagnostic = terminal.getvalue()
    assert "Logging error in Loguru Handler" in diagnostic
    assert "RecursionError" not in diagnostic


def test_python_warnings_are_logged_as_warnings(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    with configured_logging(log_path, "INFO"):
        warnings.warn("optional dependency unavailable", UserWarning, stacklevel=1)

    matching = [line for line in log_path.read_text().splitlines() if "optional dependency unavailable" in line]
    assert matching
    assert all("[WARNING]" in line for line in matching)
    assert all("[ERROR]" not in line for line in matching)


def test_loguru_handler_can_use_redirected_stdout_as_its_sink(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    with configured_logging(log_path, "INFO"):
        # stable-pretraining performs this replacement when first imported.
        logger.remove()
        logger.add(sys.stdout, format="{message}")
        logger.info("dependency-owned handler")

    assert "dependency-owned handler" in log_path.read_text()
