"""Encode rows ``[row_start, row_end)`` of a dataset into a latent cache shard.

One shard per GPU; shards use the same featurizer as ``cache_latents``, so they
match a single-process cache. Rows are flattened (episode, step) pairs, so shard
on episode boundaries.

Example::

    pixi run prepare job=cache_lance_shard preparation.wm=<world model> \
        preparation.dataset=<dataset> preparation.row_start=<start> preparation.row_end=<end> \
        preparation.out=<shard.pt>
"""

from pathlib import Path
from typing import cast

import stable_worldmodel as swm
from omegaconf import DictConfig

from rp1.core.world_model.featurize import build_featurizer
from rp1.data import encode_dataset
from rp1.data.base import Dataset, RowRange
from rp1.training.harness.checkpointing import load_wm
from rp1.utils.config import phase_config
from rp1.utils.device import pick_device
from rp1.utils.logging import logger


def run(cfg: DictConfig) -> None:
    a = phase_config(cfg, "preparation")

    device = pick_device(a.device)
    wm = load_wm(a.wm, device=device)
    featurizer = build_featurizer(wm, device=device, img_size=a.image_size, train_res=None)
    full = swm.data.load_dataset(a.dataset)
    s, e = a.row_start, a.row_end

    cache = encode_dataset(
        RowRange(cast(Dataset, full), s, e),
        featurizer,
        batch_size=a.batch_size,
        state_key=a.state_key or None,
        meta={"wm": a.wm, "dataset": a.dataset, "rows": [s, e]},
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cache.save(a.out)
    logger.success(f"Cached shard rows=[{s}:{e}] latents={len(cache.z)} path={a.out}")
