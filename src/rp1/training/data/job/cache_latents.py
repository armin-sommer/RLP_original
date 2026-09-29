"""Encode a logged dataset into a latent cache.

The encoder runs once here; the value and planner trainers then work on the
cached latents.

Example::

    pixi run prepare job=cache_latents preparation.wm=assets/core/world_model/cube_lewm \
        preparation.dataset=<dataset> preparation.out=<cache.pt>
"""

from pathlib import Path
from typing import cast

import numpy as np
import stable_worldmodel as swm
from omegaconf import DictConfig

from rp1.core.world_model.featurize import build_featurizer
from rp1.data import encode_dataset
from rp1.data.base import Dataset, RowRange, episode_index
from rp1.training.harness.checkpointing import load_wm
from rp1.utils.config import phase_config
from rp1.utils.device import pick_device
from rp1.utils.logging import logger


def run(cfg: DictConfig) -> None:
    args = phase_config(cfg, "preparation")

    device = pick_device(args.device)
    wm = load_wm(args.wm, device=device)
    featurizer = build_featurizer(wm, device=device, img_size=args.image_size, train_res=args.train_res)

    dataset = cast(Dataset, swm.data.load_dataset(args.dataset))
    state_key: str | None = str(args.state_key) if args.state_key else None

    max_rows: int | None = int(args.max_rows) if args.max_rows is not None else None
    if args.max_episodes is not None:
        # Restrict to the first N episodes (the paper's train split): episode
        # rows are stored contiguously, so this is a row prefix.
        episodes = episode_index(dataset).astype(np.int64)
        in_split = int(np.count_nonzero(episodes < int(args.max_episodes)))
        if not np.array_equal(np.flatnonzero(episodes < int(args.max_episodes)), np.arange(in_split)):
            raise ValueError("episodes are not stored contiguously; cannot apply max_episodes as a prefix")
        max_rows = in_split if max_rows is None else min(max_rows, in_split)

    if max_rows is not None:
        dataset = RowRange(dataset, 0, max_rows)

    cache = encode_dataset(
        dataset,
        featurizer,
        batch_size=args.batch_size,
        state_key=state_key,
        meta={
            "wm": args.wm,
            "dataset": args.dataset,
            "device": device,
            "train_res": args.train_res,
        },
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cache.save(args.out)
    logger.success(f"Cached {len(cache.z)} latents (dim={cache.latent_dim}) at {args.out}")
