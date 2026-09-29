"""Tests for the external dataset registry and fetch tool."""

from __future__ import annotations

from pathlib import Path

import lance
import pyarrow as pa
import pytest
from huggingface_hub.hf_api import DatasetInfo

import rp1.training.data.job.fetch_dataset as fetch_module
from rp1.data import dataset_path, get_dataset_spec


def _dataset_info(size: int = 1024) -> DatasetInfo:
    spec = get_dataset_spec("ogb_cube")
    return DatasetInfo(  # type: ignore[no-untyped-call]
        id=spec.repo_id,
        sha=spec.revision,
        siblings=[{"rfilename": f"{spec.remote_directory}/data/example.lance", "size": size}],
    )


def test_dataset_path_uses_external_data_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RP1_DATA_HOME", str(tmp_path))
    assert dataset_path(get_dataset_spec("ogb_cube")) == tmp_path / "datasets" / "ogb_cube_single.lance"


def test_unknown_dataset_lists_available_names() -> None:
    with pytest.raises(ValueError, match="available datasets: ogb_cube"):
        get_dataset_spec("missing")


def test_cube_registry_includes_all_evaluation_pose_columns() -> None:
    required = set(get_dataset_spec("ogb_cube").required_columns)
    assert {"privileged_block_0_pos", "privileged_block_0_quat"}.issubset(required)


def test_dry_run_checks_metadata_without_downloading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    info = _dataset_info()

    class FakeApi:
        def dataset_info(self, repo_id: str, *, revision: str, files_metadata: bool) -> DatasetInfo:
            assert repo_id == info.id
            assert revision == info.sha
            assert files_metadata
            return info

    monkeypatch.setattr(fetch_module, "HfApi", FakeApi)
    monkeypatch.setattr(fetch_module, "_disk_free", lambda path: 10 * 1024**3)
    monkeypatch.setattr(fetch_module, "snapshot_download", lambda **kwargs: pytest.fail("downloaded during dry run"))

    result = fetch_module.fetch_dataset(
        "ogb_cube",
        cache_root=tmp_path,
        dry_run=True,
        force=False,
        max_workers=4,
        min_free_gib=2.0,
    )

    assert result is None
    assert not dataset_path(get_dataset_spec("ogb_cube"), tmp_path).exists()


def test_fetch_validates_and_reuses_completed_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    info = _dataset_info()
    api_calls = 0

    class FakeApi:
        def dataset_info(self, repo_id: str, *, revision: str, files_metadata: bool) -> DatasetInfo:
            nonlocal api_calls
            api_calls += 1
            return info

    def fake_download(**kwargs: object) -> str:
        destination = Path(str(kwargs["local_dir"])) / "ogb_cube_single.lance"
        columns: dict[str, list[object]] = {
            column: [b"pixel" if column == "pixels" else 1.0]
            for column in get_dataset_spec("ogb_cube").required_columns
        }
        lance.write_dataset(pa.table(columns), destination)
        remote_file = Path(str(kwargs["local_dir"])) / "ogb_cube_single.lance" / "data" / "example.lance"
        remote_file.write_bytes(b"0" * 1024)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(fetch_module, "HfApi", FakeApi)
    monkeypatch.setattr(fetch_module, "_disk_free", lambda path: 10 * 1024**3)
    monkeypatch.setattr(fetch_module, "snapshot_download", fake_download)

    first = fetch_module.fetch_dataset(
        "ogb_cube", cache_root=tmp_path, dry_run=False, force=False, max_workers=4, min_free_gib=2.0
    )
    second = fetch_module.fetch_dataset(
        "ogb_cube", cache_root=tmp_path, dry_run=False, force=False, max_workers=4, min_free_gib=2.0
    )

    assert first is not None
    assert first.rows == 1
    assert second == first
    assert api_calls == 1
    assert (tmp_path / "datasets" / ".rp1" / "ogb_cube.json").is_file()


def test_fetch_refuses_insufficient_disk_space(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    info = _dataset_info(size=5 * 1024**3)

    class FakeApi:
        def dataset_info(self, repo_id: str, *, revision: str, files_metadata: bool) -> DatasetInfo:
            return info

    monkeypatch.setattr(fetch_module, "HfApi", FakeApi)
    monkeypatch.setattr(fetch_module, "_disk_free", lambda path: 4 * 1024**3)

    with pytest.raises(RuntimeError, match="Insufficient disk space"):
        fetch_module.fetch_dataset(
            "ogb_cube", cache_root=tmp_path, dry_run=False, force=False, max_workers=4, min_free_gib=1.0
        )


def test_archive_extraction_and_h5_validation(tmp_path: Path) -> None:
    import tarfile

    import h5py
    import numpy as np
    import zstandard

    from rp1.data import DatasetSpec
    from rp1.training.data.job.fetch_dataset import _extract_archive, _validate_dataset

    source = tmp_path / "payload"
    source.mkdir()
    with h5py.File(source / "mini.h5", "w") as handle:
        handle.create_dataset("action", data=np.zeros((12, 2), dtype=np.float32))
    plain_tar = tmp_path / "mini.tar"
    with tarfile.open(plain_tar, "w") as tar:
        tar.add(source / "mini.h5", arcname="mini.h5")
    archive = tmp_path / "mini.tar.zst"
    archive.write_bytes(zstandard.ZstdCompressor().compress(plain_tar.read_bytes()))

    destination = tmp_path / "extracted"
    _extract_archive(archive, destination)
    spec = DatasetSpec(
        name="mini",
        repo_id="unused/unused",
        revision="0" * 40,
        remote_directory="mini.tar.zst",
        local_directory="extracted",
        required_columns=("action",),
        kind="h5",
        archive_file="mini.tar.zst",
    )
    rows, columns = _validate_dataset(destination, spec)
    assert rows == 12
    assert "action" in columns


def test_plain_zst_extraction_and_h5_validation(tmp_path: Path) -> None:
    """A bare ``.h5.zst`` (the PushT release layout) decompresses in place."""
    import h5py
    import numpy as np
    import zstandard

    from rp1.data import DatasetSpec
    from rp1.training.data.job.fetch_dataset import _extract_archive, _validate_dataset

    payload = tmp_path / "pusht_expert_train.h5"
    with h5py.File(payload, "w") as handle:
        handle.create_dataset("action", data=np.zeros((7, 2), dtype=np.float32))
        handle.create_dataset("episode_idx", data=np.zeros(7, dtype=np.int64))
    archive = tmp_path / "pusht_expert_train.h5.zst"
    archive.write_bytes(zstandard.ZstdCompressor().compress(payload.read_bytes()))
    payload.unlink()

    destination = tmp_path / "pusht"
    _extract_archive(archive, destination)
    assert (destination / "pusht_expert_train.h5").is_file()
    spec = DatasetSpec(
        name="pusht",
        repo_id="unused/unused",
        revision="0" * 40,
        remote_directory="pusht_expert_train.h5.zst",
        local_directory="pusht",
        required_columns=("action", "episode_idx"),
        kind="h5",
        archive_file="pusht_expert_train.h5.zst",
    )
    rows, columns = _validate_dataset(destination, spec)
    assert rows == 7
    assert "episode_idx" in columns
