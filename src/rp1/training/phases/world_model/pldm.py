"""Pretrain a world model with the PLDM objective, the LeWM paper's PLDM baseline.

The architecture is LeWM's, so the checkpoint is saved in the LeWM layout and
loads wherever a LeWM world model does. Only the objective differs, the
authors' recipe:

    loss = pred_loss                                   (one-step latent prediction)
         + std * std_loss + std_t * std_t_loss         (VCReg variance terms)
         + cov * cov_loss + cov_t * cov_t_loss         (VCReg covariance terms)
         + temp_align * ||z_t - z_{t+1}||^2            (temporal alignment)
         + idm * ||IDM([z_{t+1}, z_t]) - a_t||^2       (inverse-dynamics head)
         + temp_straight * (-cos(v_t, v_{t+1}))        (temporal straightening)

The inverse-dynamics head is a separate module with its own optimizer.

Example::

    pixi run pretrain --config-name phases/world_model/pldm
"""

from functools import partial
from typing import Any, cast

import hydra
import stable_pretraining as spt
import torch
from omegaconf import DictConfig
from stable_worldmodel.wm.loss import PLDMLoss, TemporalStraighteningLoss

from rp1.training.phases.world_model.trainer import fit, loaders, schedule, world_model


def pldm_forward(self: Any, batch: dict[str, torch.Tensor], stage: str, cfg: DictConfig) -> dict[str, torch.Tensor]:
    """Encode the clip, predict the next latent, and apply the enabled PLDM terms."""
    ctx_len = cfg.core.world_model.history_size
    n_preds = cfg.core.world_model.num_predictions

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    raw_output = self.model.encode(batch)
    if not isinstance(raw_output, dict) or not all(torch.is_tensor(raw_output.get(k)) for k in ("emb", "act_emb")):
        raise TypeError("world-model encode must return a mapping with tensor 'emb' and 'act_emb'")
    output = cast(dict[str, torch.Tensor], {k: v for k, v in raw_output.items() if torch.is_tensor(v)})

    emb = output["emb"]  # (B, T, D)
    pred_emb = self.model.predict(emb[:, :ctx_len], output["act_emb"][:, :ctx_len])
    output["pred_loss"] = (pred_emb - emb[:, n_preds:]).pow(2).mean()

    output["idm_emb"] = torch.cat([emb[:, 1:], emb[:, :-1]], dim=-1)
    output["act_label"] = batch["action"][:, :-1].detach()
    output["act_pred"] = self.idm(output["idm_emb"])
    output["temp_straight_loss"] = self.path_straight(emb)
    output.update(self.pldm(emb, output["act_pred"], output["act_label"]))

    loss = output["pred_loss"]
    for name, term in cfg.training.loss.items():
        if term.enabled:
            loss = loss + term.weight * output[f"{name}_loss"]
    output["loss"] = loss

    self.log_dict({f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}, on_step=True, sync_dist=True)
    return output


def run(cfg: DictConfig) -> None:
    dataset, train, val = loaders(cfg)
    model_cfg, model = world_model(cfg, dataset)
    embedding_dim = int(model_cfg.predictor.output_dim)
    idm = hydra.utils.instantiate(
        {
            "_target_": "stable_worldmodel.wm.lewm.module.MLP",
            "input_dim": 2 * embedding_dim,
            "hidden_dim": cfg.training.idm_hidden_dim,
            "output_dim": model_cfg.action_encoder.input_dim,
        }
    )
    module = spt.Module(
        model=model,
        idm=idm,
        pldm=PLDMLoss(),
        path_straight=TemporalStraighteningLoss(),
        forward=partial(pldm_forward, cfg=cfg),
        optim={"model_opt": schedule(cfg, "model", train), "idm_opt": schedule(cfg, "idm", train)},
    )
    fit(cfg, module, model_cfg, train, val)
