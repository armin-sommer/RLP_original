"""Pretrain a LeWM world model: the JEPA prediction loss with SIGReg on the encoder's latents.

The data, model and fitting steps are shared with the PLDM objective in :mod:`.pldm`.
"""

from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any, cast

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import SIGReg
from torch import nn

from rp1.data.base import load_action_stats
from rp1.training.harness.callbacks import NonFiniteGradientGuard, PortableCheckpointCallback
from rp1.training.harness.tracking import make_logger
from rp1.training.harness.transforms import image_preprocessor
from rp1.utils.logging import logger


def lejepa_forward(self: Any, batch: dict[str, torch.Tensor], stage: str, cfg: DictConfig) -> dict[str, torch.Tensor]:
    ctx_len = cfg.core.world_model.history_size
    n_preds = cfg.core.world_model.num_predictions
    lambd = cfg.training.loss.sigreg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    raw_output = self.model.encode(batch)
    if not isinstance(raw_output, dict) or not all(torch.is_tensor(value) for value in raw_output.values()):
        raise TypeError("LeWM encode must return a tensor mapping")
    output = cast(dict[str, torch.Tensor], raw_output)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    # rollout_length K > 0 adds a multi-step term: the model's own predictions are
    # rolled forward K steps under the real actions and matched to the encoder's
    # (stop-gradient) targets, as the planner rolls it out, so it cannot fabricate a
    # plausible terminal over a multi-step plan.
    K = cfg.core.world_model.rollout_length
    if K > 0:
        pred_emb = self.model.predict(emb[:, :ctx_len], act_emb[:, :ctx_len])
        output["pred_loss"] = (pred_emb - emb[:, 1 : ctx_len + 1]).pow(2).mean()
        embs = [emb[:, i] for i in range(ctx_len)]
        roll = torch.zeros((), device=emb.device)
        for k in range(K):
            win_e = torch.stack(embs[-ctx_len:], dim=1)
            nxt = self.model.predict(win_e, act_emb[:, k : k + ctx_len])[:, -1]
            roll = roll + (nxt - emb[:, ctx_len + k].detach()).pow(2).mean()
            embs.append(nxt)
        output["rollout_loss"] = roll / K
    else:
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        tgt_emb = emb[:, n_preds:]
        pred_emb = self.model.predict(ctx_emb, ctx_act)
        output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()

    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
    if K > 0:
        output["loss"] = output["loss"] + cfg.core.world_model.rollout_weight * output["rollout_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def loaders(cfg: DictConfig) -> tuple[Any, Any, Any]:
    """The dataset and its train and validation loaders, over episodes ``[0, data.train_episodes)``."""
    raw_dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    if not isinstance(raw_dataset_cfg, Mapping):
        raise TypeError("data.dataset must resolve to a mapping")
    dataset_cfg: dict[str, Any] = {str(key): value for key, value in raw_dataset_cfg.items()}
    dataset_name = str(dataset_cfg.pop("name"))
    cache_dir = cfg.data.cache_dir
    location = f"local cache: {cache_dir}" if cache_dir else "default location"
    logger.info(f'Loading dataset "{dataset_name}" from {location}')
    dataset = swm.data.load_dataset(dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg)
    transforms = [image_preprocessor(source="pixels", target="pixels", image_size=cfg.data.image_size)]

    action_stats = cfg.data.action_stats

    def make_normalizer(column: str) -> Any:
        if column != "action" or not action_stats:
            return get_column_normalizer(dataset, column, column)

        import numpy as np
        from stable_pretraining.data.transforms import WrapTorchTransform
        from stable_worldmodel.data.normalization import ZScoreScaler

        mean, std = load_action_stats(action_stats)
        scaler = ZScoreScaler(
            mean=np.asarray(mean, dtype=np.float32).reshape(1, -1),
            std=np.asarray(std, dtype=np.float32).reshape(1, -1),
        )
        logger.info(f"Actions normalized with statistics from {action_stats}")
        return WrapTorchTransform(scaler, source=column, target=column)

    for col in cfg.data.dataset.keys_to_load:
        if col.startswith("pixels"):
            continue
        transforms.append(make_normalizer(col))

    for col in cfg.data.dataset.keys_to_merge:
        transforms.append(make_normalizer(col))

    compose = cast(Callable[..., Any], spt.data.transforms.Compose)
    transform = compose(*transforms)
    dataset.transform = transform

    pool: Any = dataset
    if cfg.data.train_episodes is not None:
        # clips are (episode, start) pairs; the episodes above the cap are the evaluation draws
        keep = [i for i, (episode, _) in enumerate(dataset.clip_indices) if episode < cfg.data.train_episodes]
        if not keep:
            raise ValueError(f"no clips in episodes [0, {cfg.data.train_episodes})")
        logger.info(f"Training on episodes [0, {cfg.data.train_episodes}): {len(keep)}/{len(dataset)} clips")
        pool = cast(Any, torch.utils.data.Subset(dataset, keep))

    rnd_gen = torch.Generator().manual_seed(cfg.runtime.seed)
    train_set, val_set = spt.data.random_split(
        pool,
        lengths=[cfg.data.train_split, 1 - cfg.data.train_split],
        generator=rnd_gen,
    )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.data.loader,
        generator=rnd_gen,
    )
    val_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.data.loader, resolve=True))
    val_cfg["shuffle"] = False
    val_cfg["drop_last"] = False
    val = torch.utils.data.DataLoader(val_set, **val_cfg)
    return dataset, train, val


def world_model(cfg: DictConfig, dataset: Any) -> tuple[DictConfig, nn.Module]:
    """The architecture config, its action width set from the dataset, and the model built from it."""
    model_cfg = cast(
        DictConfig, OmegaConf.create(OmegaConf.to_container(cfg.core.world_model.architecture, resolve=True))
    )
    model_cfg.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
    model = cast(nn.Module, hydra.utils.instantiate(model_cfg))
    init_weights = cfg.training.initial_weights
    if init_weights:
        state_dict = torch.load(init_weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        logger.info(f"Initialized world-model weights from {init_weights}")
    return model_cfg, model


def schedule(cfg: DictConfig, modules: str, train: Any) -> dict[str, Any]:
    """The optimizer of ``modules``: linear warmup over the first 1% of steps, then cosine annealing."""
    total_steps = cfg.training.trainer.max_epochs * len(train)
    return {
        "modules": modules,
        "optimizer": dict(cfg.training.optimizer),
        "scheduler": {
            "type": "LinearWarmupCosineAnnealingLR",
            "warmup_steps": max(1, int(0.01 * total_steps)),
            "max_steps": total_steps,
        },
        "interval": "epoch",
    }


def fit(cfg: DictConfig, module: Any, model_cfg: DictConfig, train: Any, val: Any) -> None:
    """Train ``module``, resuming from the run's trainer checkpoint, and export the world model every epoch."""
    run_dir = Path(cfg.run.directory)
    experiment_logger = make_logger(cfg)
    hyperparameters = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(hyperparameters, dict):
        raise TypeError("training configuration must resolve to a dictionary")
    experiment_logger.log_hyperparams({str(key): value for key, value in hyperparameters.items()})

    object_dump_callback = PortableCheckpointCallback(
        run_name=cfg.output.model_name,
        config=model_cfg,
        cache_dir=run_dir,
        epoch_interval=cfg.output.checkpoint_epoch_interval,
        step_interval=cfg.output.save_every_steps,
    )
    lightning_checkpoint = pl.pytorch.callbacks.ModelCheckpoint(
        dirpath=Path(cfg.run.checkpoints) / "lightning",
        filename="{epoch}-{step}",
        save_last=True,
    )

    trainer = pl.Trainer(
        **cfg.training.trainer,
        callbacks=[
            object_dump_callback,
            NonFiniteGradientGuard(cfg.training.max_skipped_gradients),
            lightning_checkpoint,
        ],
        num_sanity_val_steps=1,
        logger=experiment_logger,
        enable_checkpointing=True,
        enable_progress_bar=False,
        default_root_dir=run_dir,
    )

    ckpt_path = Path(cfg.run.checkpoints) / "trainer.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train, val=val),
        ckpt_path=str(ckpt_path) if ckpt_path.exists() else None,  # ty: ignore[invalid-argument-type]  # Manager accepts None to disable resume despite its narrow annotation.
    )
    manager()


def run(cfg: DictConfig) -> None:
    dataset, train, val = loaders(cfg)
    model_cfg, model = world_model(cfg, dataset)
    module = spt.Module(
        model=model,
        sigreg=SIGReg(**cfg.training.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim={"model_opt": schedule(cfg, "model", train)},
    )
    fit(cfg, module, model_cfg, train, val)
