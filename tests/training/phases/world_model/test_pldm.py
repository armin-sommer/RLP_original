from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from stable_worldmodel.wm.loss import PLDMLoss, TemporalStraighteningLoss
from torch import nn

from rp1.training.phases.world_model.pldm import pldm_forward
from rp1.utils.config import compose_config


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(6, 8)
        self.predictor = nn.Linear(8, 8)

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"emb": self.encoder(batch["pixels"]), "act_emb": torch.zeros(*batch["pixels"].shape[:2], 8)}

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        return self.predictor(emb + act_emb)


def test_every_enabled_term_enters_the_loss() -> None:
    cfg = compose_config(Path("training"), "phases/world_model/pldm", [])
    logged: dict[str, torch.Tensor] = {}
    module = SimpleNamespace(
        model=_Model(),
        idm=nn.Linear(16, 3),
        pldm=PLDMLoss(),
        path_straight=TemporalStraighteningLoss(),
        log_dict=lambda values, **_: logged.update(values),
    )
    batch = {"pixels": torch.randn(4, 4, 6), "action": torch.randn(4, 4, 3)}
    output = pldm_forward(module, batch, "train", cfg)
    enabled = [name for name, term in cfg.training.loss.items() if term.enabled]
    expected = output["pred_loss"] + sum(cfg.training.loss[name].weight * output[f"{name}_loss"] for name in enabled)
    assert torch.allclose(output["loss"], expected)
    assert "train/std_loss" in logged
    output["loss"].backward()
