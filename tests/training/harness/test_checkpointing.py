from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import pytest
import stable_worldmodel
import torch
from omegaconf import OmegaConf
from torch import nn

from rp1.training.harness import checkpointing as checkpoint_module


def test_stable_worldmodel_comes_from_pinned_distribution() -> None:
    package_path = Path(stable_worldmodel.__file__).resolve()
    assert importlib.metadata.version("stable-worldmodel") == "0.1.1"
    assert "thirdparty" not in package_path.parts


def test_checkpoint_adapter_accepts_plain_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_save(model: nn.Module, *, run_name: str, config: object, **kwargs: Any) -> None:
        captured.update(model=model, run_name=run_name, config=config, kwargs=kwargs)

    monkeypatch.setattr(checkpoint_module, "_save_pretrained", fake_save)
    model = nn.Linear(2, 2)
    checkpoint_module.save_pretrained(
        model,
        run_name="test",
        config={"_target_": "torch.nn.Identity"},
        filename="weights.pt",
    )
    assert captured["model"] is model
    assert OmegaConf.is_config(captured["config"])
    assert captured["kwargs"] == {"filename": "weights.pt"}


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_checkpoint_loader_resolves_existing_relative_paths(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_load(name: str, cache_dir: str | None = None, extra_args: object = None) -> nn.Module:
        captured.update(name=name, cache_dir=cache_dir, extra_args=extra_args)
        return nn.Identity()

    checkpoint = tmp_path / "checkpoint"
    if kind == "directory":
        checkpoint.mkdir()
    else:
        checkpoint.write_bytes(b"checkpoint")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(checkpoint_module, "_load_pretrained", fake_load)

    model = checkpoint_module.load_pretrained(Path("checkpoint"), cache_dir="cache", extra_args={"value": 1})

    assert isinstance(model, nn.Identity)
    assert captured == {
        "name": str(checkpoint.resolve()),
        "cache_dir": "cache",
        "extra_args": {"value": 1},
    }


def test_checkpoint_loader_preserves_remote_or_missing_names(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    def fake_load(name: str, cache_dir: str | None = None, extra_args: object = None) -> nn.Module:
        del cache_dir, extra_args
        captured.append(name)
        return nn.Identity()

    monkeypatch.setattr(checkpoint_module, "_load_pretrained", fake_load)
    checkpoint_module.load_pretrained("owner/model")
    checkpoint_module.load_pretrained("missing-checkpoint.pt")

    assert captured == ["owner/model", "missing-checkpoint.pt"]


def test_planner_checkpoints_find_their_value_next_to_them(tmp_path: Path) -> None:
    from rp1.core.agent.value import QuasimetricHead
    from rp1.training.harness.checkpointing import load_planner, save_metric

    value = QuasimetricHead(4, hidden_dim=8, embed_dim=4, depth=1, sym_frac=0.5)
    saved = save_metric(value, run_name="value", cache_dir=tmp_path / "run")
    torch.save({"value": "value", "horizon": 5}, saved.parent / "planner.pt")
    moved = tmp_path / "elsewhere"
    saved.parent.rename(moved)

    checkpoint = load_planner(str(moved / "planner.pt"), None)
    assert checkpoint.payload["horizon"] == 5
    state = torch.randn(3, 4)
    assert torch.equal(checkpoint.value(state, state), value(state, state))
