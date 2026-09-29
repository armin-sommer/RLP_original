from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from lightning import LightningModule, Trainer

from rp1.training.harness import callbacks
from rp1.training.harness.callbacks import NonFiniteGradientGuard, PortableCheckpointCallback


@dataclass
class TrainerDouble:
    global_step: int
    current_epoch: int = 0
    max_epochs: int = 1
    is_global_zero: bool = True


class ModuleDouble(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Identity()


def test_nonfinite_gradient_guard_skips_and_aborts() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    trainer = cast(Trainer, TrainerDouble(global_step=7))
    guard = NonFiniteGradientGuard(max_skipped=1)
    parameter.grad = torch.tensor(float("nan"))
    guard.on_before_optimizer_step(trainer, cast(LightningModule, torch.nn.Identity()), optimizer)
    observed_grad: torch.Tensor | None = parameter.grad
    assert observed_grad is None
    assert guard.skipped == 1
    parameter.grad = torch.tensor(float("inf"))
    with pytest.raises(ValueError, match="more than 1"):
        guard.on_before_optimizer_step(trainer, cast(LightningModule, torch.nn.Identity()), optimizer)


def test_portable_checkpoint_callback_cadence_and_rank(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    saved: list[tuple[torch.nn.Module, str, object, str, str | Path]] = []

    def fake_save(
        model: torch.nn.Module,
        *,
        run_name: str,
        config: object,
        filename: str,
        cache_dir: str | Path,
        **kwargs: Any,
    ) -> None:
        del kwargs
        saved.append((model, run_name, config, filename, cache_dir))

    monkeypatch.setattr(callbacks, "save_pretrained", fake_save)
    callback = PortableCheckpointCallback(
        "wm",
        {"_target_": "torch.nn.Identity"},
        tmp_path,
        epoch_interval=1,
        step_interval=5,
    )
    module = ModuleDouble()
    trainer_double = TrainerDouble(global_step=5)
    trainer = cast(Trainer, trainer_double)
    callback.on_train_batch_end(trainer, module, None, None, 0)
    callback.on_train_batch_end(trainer, module, None, None, 0)
    callback.on_train_epoch_end(trainer, module)
    assert [item[3] for item in saved] == ["weights_step_5.pt", "weights_epoch_1.pt"]

    trainer_double.global_step = 10
    trainer_double.is_global_zero = False
    callback.on_train_batch_end(trainer, module, None, None, 0)
    assert len(saved) == 2
