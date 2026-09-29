"""Keep one latent-cache row per action block of ``frameskip`` primitive steps.

The planner acts in blocks of ``frameskip`` steps, so its value is trained on
the same horizon: every ``frameskip``-th frame of each episode, with ``step_idx``
counted in blocks. The latents themselves are unchanged.

``phases > 1`` keeps further residue classes of the stride, each as its own
episode: phase ``k`` of episode ``e`` is episode ``e * phases + k``, starting at
primitive step ``k``. The world model sees the same (history, action block)
tuples at every phase, so each is an equally valid block-aligned trajectory,
and the planner gets ``phases`` times the start states without re-encoding.

Example::

    pixi run prepare job=subsample_cache \
        preparation.inp=<fs1 cache> preparation.out=<fs5 cache> preparation.frameskip=5
"""

import numpy as np
import torch
from omegaconf import DictConfig

from rp1.data import LatentCache
from rp1.utils.config import phase_config
from rp1.utils.logging import logger


def run(cfg: DictConfig) -> None:
    args = phase_config(cfg, "preparation")

    c = LatentCache.load(args.inp, mmap=bool(args.cache_mmap))
    fs, phases = int(args.frameskip), int(args.phases)
    if not 1 <= phases <= fs:
        raise ValueError(f"phases must be in [1, {fs}], got {phases}")
    zs, eps, sts, states = [], [], [], []
    for e, rows in c.episodes().items():  # rows sorted by step_idx
        for k in range(phases):
            sub = rows[k::fs]
            if len(sub) < 2:
                continue
            zs.append(c.z[sub])
            eps.append(np.full(len(sub), e * phases + k, np.int64))
            sts.append(np.arange(len(sub), dtype=np.int64))  # re-index in planner-steps
            if c.state is not None:
                states.append(c.state[sub])
    meta = {**(c.meta or {}), "frameskip": fs, "horizon_matched": True}
    if phases > 1:
        meta["phase_multiplex"] = phases
    out = LatentCache(
        z=torch.cat(zs),
        episode_idx=torch.from_numpy(np.concatenate(eps)),
        step_idx=torch.from_numpy(np.concatenate(sts)),
        state=(torch.cat(states) if states else None),
        meta=meta,
    )
    lens = [len(r) for r in out.episodes().values()]
    logger.info(
        f"fs{fs} cache ({phases} phase(s)): {len(out.z)} latents, {len(lens)} episodes, "
        f"len[min/med/max]={min(lens)}/{int(np.median(lens))}/{max(lens)}"
    )
    out.save(args.out)
