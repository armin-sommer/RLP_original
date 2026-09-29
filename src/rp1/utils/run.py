"""Timestamped run directories and reproducibility metadata."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import monotonic
from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict

_SUBDIRECTORIES = ("checkpoints", "metrics", "videos", "artifacts", "tracking", "stages")


@dataclass(frozen=True)
class RunPaths:
    """Canonical paths belonging to one command invocation."""

    directory: Path
    checkpoints: Path
    metrics: Path
    videos: Path
    artifacts: Path
    tracking: Path
    stages: Path
    config: Path
    metadata: Path
    log: Path

    @classmethod
    def create(cls, root: str | Path, now: datetime | None = None) -> RunPaths:
        moment = now or datetime.now().astimezone()
        date_directory = Path(root).resolve() / moment.strftime("%Y-%m-%d")
        date_directory.mkdir(parents=True, exist_ok=True)
        stem = moment.strftime("%H-%M-%S")
        suffix = 1
        while True:
            name = stem if suffix == 1 else f"{stem}-{suffix:02d}"
            directory = date_directory / name
            try:
                directory.mkdir()
                break
            except FileExistsError:
                suffix += 1
        children = {name: directory / name for name in _SUBDIRECTORIES}
        for child in children.values():
            child.mkdir()
        return cls(
            directory=directory,
            **children,
            config=directory / "config.yaml",
            metadata=directory / "metadata.json",
            log=directory / "run.log",
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> RunPaths:
        values = {field: Path(cfg.run[field]) for field in cls.__dataclass_fields__}
        return cls(**values)

    def attach(self, cfg: DictConfig) -> None:
        with open_dict(cfg):
            cfg.run = OmegaConf.create({key: str(value) for key, value in asdict(self).items()})


def _git_metadata(cwd: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None}


def _package_versions() -> dict[str, str | None]:
    packages = ("rp1", "stable-worldmodel", "torch", "lightning", "hydra-core")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


class RunMetadata:
    """Incrementally write lifecycle metadata for a run."""

    def __init__(self, paths: RunPaths, cfg: DictConfig):
        self.paths = paths
        self.started = datetime.now().astimezone()
        self._clock = monotonic()
        cwd = Path.cwd().resolve()
        self.data: dict[str, Any] = {
            "status": "running",
            "started_at": self.started.isoformat(),
            "finished_at": None,
            "duration_seconds": None,
            "command": sys.argv,
            "entrypoint": cfg.entrypoint._target_,
            "working_directory": str(cwd),
            "run_directory": str(paths.directory),
            "hostname": socket.gethostname(),
            "python": {"version": platform.python_version(), "executable": sys.executable},
            "packages": _package_versions(),
            "platform": platform.platform(),
            "process_id": os.getpid(),
            "git": _git_metadata(cwd),
            "error": None,
        }
        self._write()

    def finish(self, status: str, error: str | None = None) -> None:
        self.data.update(
            status=status,
            finished_at=datetime.now().astimezone().isoformat(),
            duration_seconds=round(monotonic() - self._clock, 6),
            error=error,
        )
        self._write()

    def _write(self) -> None:
        temporary = self.paths.metadata.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, indent=2) + "\n")
        temporary.replace(self.paths.metadata)


def save_config(cfg: DictConfig, path: Path) -> None:
    """Persist the fully resolved configuration used by the run."""
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True))


def save_stage_config(cfg: DictConfig, stage_directory: Path) -> None:
    stage_directory.mkdir(parents=True, exist_ok=True)
    (stage_directory / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))


__all__ = ["RunMetadata", "RunPaths", "save_config", "save_stage_config"]
