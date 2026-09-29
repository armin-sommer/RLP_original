"""Extract a dataset's actions into an h5 of ``action``, ``ep_offset`` and ``ep_len``.

The rp1 and baseline trainers read only these; episodes keep the dataset's
order, which is the latent cache's.
"""

from typing import cast

import h5py
import numpy as np
import stable_worldmodel as swm
from omegaconf import DictConfig

from rp1.data.base import Dataset, episode_index
from rp1.utils.config import phase_config
from rp1.utils.logging import logger


def run(cfg: DictConfig) -> None:
    args = phase_config(cfg, "preparation")

    ds = swm.data.load_dataset(args.dataset)
    epi = episode_index(cast(Dataset, ds)).astype(np.int64)
    act = np.asarray(ds.get_col_data("action")).reshape(len(epi), -1).astype(np.float32)

    bounds = np.flatnonzero(np.diff(epi)) + 1
    ep_offset = np.concatenate([[0], bounds]).astype(np.int64)
    ep_len = np.diff(np.concatenate([ep_offset, [len(epi)]])).astype(np.int64)

    with h5py.File(args.output, "w") as f:
        f.create_dataset("action", data=act)
        f.create_dataset("ep_offset", data=ep_offset)
        f.create_dataset("ep_len", data=ep_len)

    logger.success(
        f"wrote {args.output}: action{act.shape} episodes={len(ep_offset)} "
        f"ep_len(min/med/max)={ep_len.min()}/{int(np.median(ep_len))}/{ep_len.max()}"
    )
