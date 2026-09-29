"""Weight averaging over deployable planner snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from rp1.training.harness.soup import average_planner_checkpoints, average_planner_payloads


def payload(scale: float, *, action_limit: float = 2.5, counter: int = 3) -> dict[str, Any]:
    return {
        "action_limit": action_limit,
        "horizon": 5,
        "state_dict": {
            "net.0.weight": torch.full((4, 3), scale),
            "net.0.bias": torch.full((4,), 2.0 * scale),
            "steps": torch.tensor([counter], dtype=torch.int64),
        },
    }


def test_averages_float_tensors_and_keeps_architecture() -> None:
    merged = average_planner_payloads([payload(1.0), payload(3.0)])
    assert torch.allclose(merged["state_dict"]["net.0.weight"], torch.full((4, 3), 2.0))
    assert torch.allclose(merged["state_dict"]["net.0.bias"], torch.full((4,), 4.0))
    assert merged["action_limit"] == 2.5
    assert merged["horizon"] == 5


def test_integer_buffers_are_copied_not_averaged() -> None:
    merged = average_planner_payloads([payload(1.0), payload(3.0)])
    assert merged["state_dict"]["steps"].dtype == torch.int64
    assert merged["state_dict"]["steps"].tolist() == [3]


def test_single_payload_is_a_copy() -> None:
    source = payload(1.0)
    merged = average_planner_payloads([source])
    merged["state_dict"]["net.0.weight"] += 1.0
    assert torch.allclose(source["state_dict"]["net.0.weight"], torch.full((4, 3), 1.0))


def test_preserves_dtype_of_float_tensors() -> None:
    a, b = payload(1.0), payload(2.0)
    a["state_dict"]["net.0.weight"] = a["state_dict"]["net.0.weight"].to(torch.float16)
    b["state_dict"]["net.0.weight"] = b["state_dict"]["net.0.weight"].to(torch.float16)
    merged = average_planner_payloads([a, b])
    assert merged["state_dict"]["net.0.weight"].dtype == torch.float16


def test_rejects_architecture_mismatch() -> None:
    with pytest.raises(ValueError, match="architecture"):
        average_planner_payloads([payload(1.0), payload(1.0, action_limit=3.0)])


def test_rejects_parameter_set_mismatch() -> None:
    odd = payload(1.0)
    del odd["state_dict"]["net.0.bias"]
    with pytest.raises(ValueError, match="parameter set"):
        average_planner_payloads([payload(1.0), odd])


def test_rejects_shape_mismatch() -> None:
    odd = payload(1.0)
    odd["state_dict"]["net.0.weight"] = torch.ones(5, 3)
    with pytest.raises(ValueError, match="shape/dtype"):
        average_planner_payloads([payload(1.0), odd])


def test_rejects_differing_integer_buffers() -> None:
    with pytest.raises(ValueError, match="non-float"):
        average_planner_payloads([payload(1.0), payload(1.0, counter=9)])


def test_empty_input_is_an_error() -> None:
    with pytest.raises(ValueError, match="no planner payloads"):
        average_planner_payloads([])


def test_round_trips_through_disk(tmp_path: Path) -> None:
    paths = []
    for index, scale in enumerate((1.0, 3.0)):
        p = tmp_path / f"snap{index}.pt"
        torch.save(payload(scale), p)
        paths.append(p)
    out = tmp_path / "nested" / "soup.pt"
    average_planner_checkpoints(paths, out)
    reloaded = torch.load(out, map_location="cpu", weights_only=False)
    assert torch.allclose(reloaded["state_dict"]["net.0.weight"], torch.full((4, 3), 2.0))
    assert reloaded["action_limit"] == 2.5
