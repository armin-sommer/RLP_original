"""Known external datasets and their canonical local locations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    """A reproducible external dataset source.

    ``kind`` is ``"lance"`` (a lance directory mirrored file-by-file) or
    ``"h5"`` (a single ``archive_file`` tarball extracted into
    ``local_directory``).
    """

    name: str
    repo_id: str
    revision: str
    remote_directory: str
    local_directory: str
    required_columns: tuple[str, ...] = ()
    kind: str = "lance"
    archive_file: str | None = None


DATASETS: dict[str, DatasetSpec] = {
    "ogb_cube": DatasetSpec(
        name="ogb_cube",
        repo_id="galilai-group/ogb_cube_single",
        revision="2f0d4deb19cedaedfc71a55029f93eb9dbd36665",
        remote_directory="ogb_cube_single.lance",
        local_directory="ogb_cube_single.lance",
        required_columns=(
            "episode_idx",
            "step_idx",
            "pixels",
            "action",
            "observation",
            "qpos",
            "qvel",
            "privileged_block_0_pos",
            "privileged_block_0_quat",
        ),
    ),
    "tworoom": DatasetSpec(
        name="tworoom",
        repo_id="quentinll/lewm-tworooms",
        revision="6903a2de048b13819d812da0b4dd661290bc01e4",
        remote_directory="tworoom.tar.zst",
        local_directory="tworoom",
        required_columns=("action",),
        kind="h5",
        archive_file="tworoom.tar.zst",
    ),
    "pusht": DatasetSpec(
        name="pusht",
        repo_id="quentinll/lewm-pusht",
        revision="655cd446b9929369d7d406001da85c15d1457850",
        remote_directory="pusht_expert_train.h5.zst",
        local_directory="pusht",
        required_columns=("action", "episode_idx", "step_idx", "pixels", "proprio", "state"),
        kind="h5",
        archive_file="pusht_expert_train.h5.zst",
    ),
    "reacher": DatasetSpec(
        name="reacher",
        repo_id="quentinll/lewm-reacher",
        revision="e70a080d0d04c6072123c9ebd343acf7fff28dbf",
        remote_directory="reacher.tar.zst",
        local_directory="reacher",
        required_columns=("action",),
        kind="h5",
        archive_file="reacher.tar.zst",
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return a named dataset or raise an error listing valid choices."""

    try:
        return DATASETS[name]
    except KeyError as error:
        choices = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset {name!r}; available datasets: {choices}") from error


def data_home(override: str | Path | None = None) -> Path:
    """Return the data cache root, outside the source tree by default."""

    if override is not None:
        return Path(override).expanduser().resolve()
    configured = os.environ.get("RP1_DATA_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".cache" / "rp1"


def dataset_path(spec: DatasetSpec, cache_root: str | Path | None = None) -> Path:
    """Return the canonical local path for ``spec``."""

    return data_home(cache_root) / "datasets" / spec.local_directory


__all__ = ["DATASETS", "DatasetSpec", "data_home", "dataset_path", "get_dataset_spec"]
