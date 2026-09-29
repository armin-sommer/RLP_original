"""Fetch a registered dataset from Hugging Face into the shared data cache.

Example::

    pixi run prepare job=fetch_dataset preparation.dataset=ogb_cube
"""

from __future__ import annotations

import json
import shutil
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import lance
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.hf_api import DatasetInfo, RepoSibling
from omegaconf import DictConfig

from rp1.data import DatasetSpec, data_home, dataset_path, get_dataset_spec
from rp1.utils.config import phase_config
from rp1.utils.logging import logger


@dataclass(frozen=True)
class FetchResult:
    """The verified location and metadata of a fetched dataset."""

    name: str
    repo_id: str
    revision: str
    path: str
    rows: int
    columns: tuple[str, ...]
    remote_bytes: int
    fetched_at: str


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _dataset_files(info: DatasetInfo, spec: DatasetSpec) -> list[RepoSibling]:
    if spec.archive_file is not None:
        files = [item for item in info.siblings or [] if item.rfilename == spec.archive_file]
    else:
        prefix = f"{spec.remote_directory}/"
        files = [item for item in info.siblings or [] if item.rfilename.startswith(prefix)]
    if not files:
        raise RuntimeError(f"{spec.repo_id}@{spec.revision} does not contain {spec.remote_directory!r}")
    return files


def _remaining_bytes(root: Path, files: list[RepoSibling]) -> tuple[int, int]:
    total = 0
    present = 0
    for item in files:
        if item.size is None:
            raise RuntimeError(f"Hugging Face did not report a size for {item.rfilename!r}")
        size = item.size
        total += size
        relative = PurePosixPath(item.rfilename)
        if ".." in relative.parts or relative.is_absolute():
            raise RuntimeError(f"Unsafe path in dataset repository: {item.rfilename!r}")
        local = root.joinpath(*relative.parts)
        if local.is_file() and local.stat().st_size == size:
            present += size
    return total, max(total - present, 0)


def _disk_free(path: Path) -> int:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _validate_dataset(path: Path, spec: DatasetSpec) -> tuple[int, tuple[str, ...]]:
    if not path.is_dir():
        raise RuntimeError(f"Dataset download did not create {path}")
    if spec.kind == "h5":
        return _validate_h5(path, spec)
    dataset = lance.dataset(path)
    columns = tuple(dataset.schema.names)
    missing = sorted(set(spec.required_columns).difference(columns))
    if missing:
        raise RuntimeError(f"Dataset at {path} is missing required columns: {', '.join(missing)}")
    rows = dataset.count_rows()
    if rows <= 0:
        raise RuntimeError(f"Dataset at {path} contains no rows")
    return rows, columns


def _validate_h5(path: Path, spec: DatasetSpec) -> tuple[int, tuple[str, ...]]:
    import h5py

    with suppress(ImportError):
        import hdf5plugin  # noqa: F401  (registers compression filters used by some h5s)
    candidates = sorted(path.rglob("*.h5"))
    if not candidates:
        raise RuntimeError(f"Dataset at {path} contains no .h5 file")
    with h5py.File(candidates[0], "r") as handle:
        columns = tuple(sorted(handle.keys()))
        missing = sorted(set(spec.required_columns).difference(columns))
        if missing:
            raise RuntimeError(f"Dataset at {candidates[0]} is missing required keys: {', '.join(missing)}")
        rows = int(handle[spec.required_columns[0]].shape[0]) if spec.required_columns else 0
    if rows <= 0:
        raise RuntimeError(f"Dataset at {candidates[0]} contains no rows")
    return rows, columns


def _extract_archive(archive: Path, destination: Path) -> None:
    """Extract a ``.tar.zst`` archive into ``destination`` (flat, path-checked).

    A plain ``.zst`` (not a tarball, e.g. ``pusht_expert_train.h5.zst``) is
    stream-decompressed into ``destination`` under its uncompressed name.
    """
    import tarfile

    import zstandard

    destination.mkdir(parents=True, exist_ok=True)
    with archive.open("rb") as compressed:
        reader = zstandard.ZstdDecompressor().stream_reader(compressed)
        if archive.name.endswith(".tar.zst"):
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                tar.extractall(destination, filter="data")
        else:
            target = destination / archive.name.removesuffix(".zst")
            with target.open("wb") as output:
                shutil.copyfileobj(reader, output, length=16 * 1024 * 1024)


def _manifest_path(spec: DatasetSpec, cache_root: str | Path | None) -> Path:
    return data_home(cache_root) / "datasets" / ".rp1" / f"{spec.name}.json"


def _read_completed_fetch(spec: DatasetSpec, cache_root: str | Path | None) -> FetchResult | None:
    manifest = _manifest_path(spec, cache_root)
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text())
        if not isinstance(payload, dict):
            return None
        columns = payload.get("columns")
        if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
            return None
        payload["columns"] = tuple(columns)
        result = FetchResult(**payload)
    except (OSError, TypeError, ValueError):
        return None
    destination = dataset_path(spec, cache_root)
    if result.revision != spec.revision or Path(result.path) != destination:
        return None
    rows, columns = _validate_dataset(destination, spec)
    if rows != result.rows or columns != result.columns:
        return None
    return result


def _write_manifest(result: FetchResult, spec: DatasetSpec, cache_root: str | Path | None) -> None:
    manifest = _manifest_path(spec, cache_root)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest)


def fetch_dataset(
    name: str,
    *,
    cache_root: str | Path | None,
    dry_run: bool,
    force: bool,
    max_workers: int,
    min_free_gib: float,
) -> FetchResult | None:
    spec = get_dataset_spec(name)
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if min_free_gib < 0:
        raise ValueError("min_free_gib cannot be negative")
    destination = dataset_path(spec, cache_root)
    if not force:
        completed = _read_completed_fetch(spec, cache_root)
        if completed is not None:
            logger.info(f"Dataset {name!r} is already ready at {destination} ({completed.rows:,} rows)")
            return completed

    info = HfApi().dataset_info(spec.repo_id, revision=spec.revision, files_metadata=True)
    if info.sha != spec.revision:
        raise RuntimeError(f"Resolved revision {info.sha!r} does not match pinned revision {spec.revision!r}")

    files = _dataset_files(info, spec)
    download_root = destination.parent
    remote_bytes, remaining_bytes = _remaining_bytes(download_root, files)
    if force:
        remaining_bytes = remote_bytes
    free_bytes = _disk_free(download_root)
    reserve_bytes = int(min_free_gib * 1024**3)
    logger.info(f"Dataset: {name} ({spec.repo_id}@{spec.revision[:12]})")
    logger.info(f"Destination: {destination}")
    logger.info(
        f"Remote size: {_format_bytes(remote_bytes)}; remaining: {_format_bytes(remaining_bytes)}; "
        f"free: {_format_bytes(free_bytes)}"
    )
    if remaining_bytes + reserve_bytes > free_bytes:
        required = _format_bytes(remaining_bytes + reserve_bytes)
        raise RuntimeError(f"Insufficient disk space at {download_root}: need {required} including reserve")
    if dry_run:
        logger.info("Dry run complete; no files were downloaded")
        return None

    download_root.mkdir(parents=True, exist_ok=True)
    allow = [spec.archive_file] if spec.archive_file is not None else [f"{spec.remote_directory}/**"]
    snapshot_download(
        repo_id=spec.repo_id,
        repo_type="dataset",
        revision=spec.revision,
        local_dir=download_root,
        allow_patterns=allow,
        force_download=force,
        max_workers=max_workers,
    )

    _, incomplete_bytes = _remaining_bytes(download_root, files)
    if incomplete_bytes:
        raise RuntimeError(f"Dataset download is incomplete: {_format_bytes(incomplete_bytes)} are missing")

    if spec.archive_file is not None:
        archive = download_root / spec.archive_file
        logger.info(f"Extracting {archive.name} into {destination} (needs roughly the archive size again)")
        _extract_archive(archive, destination)
        archive.unlink()

    rows, columns = _validate_dataset(destination, spec)
    result = FetchResult(
        name=spec.name,
        repo_id=spec.repo_id,
        revision=spec.revision,
        path=str(destination),
        rows=rows,
        columns=columns,
        remote_bytes=remote_bytes,
        fetched_at=datetime.now(UTC).isoformat(),
    )
    _write_manifest(result, spec, cache_root)
    logger.info(f"Dataset ready at {destination} ({rows:,} rows)")
    return result


def run(cfg: DictConfig) -> FetchResult | None:
    args = phase_config(cfg, "preparation")
    return fetch_dataset(
        str(args.dataset),
        cache_root=args.cache_root,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        max_workers=int(args.max_workers),
        min_free_gib=float(args.min_free_gib),
    )
