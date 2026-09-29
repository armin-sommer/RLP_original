"""Build the two Reacher h5 files from a Stable-WM dataset.

1. ``train_out``: every episode's scalar columns (action, qpos, qvel, ...), no
   pixels. The rp1 trainer indexes ``ep_offset`` by episode id (the latent cache's ids),
   so ``ep_len`` and ``ep_offset`` are id-indexed.
2. ``eval_out``: the first ``eval_episodes`` episodes in file order with every
   column, pixels decoded, for evaluation (task replay, goal images, action
   statistics); its reader consumes ``ep_len`` and ``ep_offset`` positionally.

Both carry explicit ``episode_idx`` and ``step_idx`` datasets, which the generic
h5 writer does not store.
"""

from io import BytesIO
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import stable_worldmodel as swm
from omegaconf import DictConfig

from rp1.data.base import Dataset
from rp1.utils.config import phase_config
from rp1.utils.logging import logger

SCALAR_COLS = [
    "action",
    "qpos",
    "qvel",
    "observation",
    "success",
    "score",
    "target_pos",
    "finger_pos",
    "reward",
    "terminated",
    "truncated",
]


def episode_blocks(ds: Dataset) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    """Contiguous episode blocks in file order: list of (id, start, length).
    Episodes are written whole by the collector but not necessarily in id
    order; each id must appear in exactly one contiguous block."""
    ep = np.asarray(ds.get_col_data("episode_idx")).reshape(-1).astype(np.int64)
    st = np.asarray(ds.get_col_data("step_idx")).reshape(-1).astype(np.int64)
    bounds = np.flatnonzero(np.diff(ep) != 0) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(ep)]])
    blocks = [(int(ep[s]), int(s), int(e - s)) for s, e in zip(starts, ends, strict=True)]
    ids = [b[0] for b in blocks]
    assert len(ids) == len(set(ids)), "an episode id appears in >1 block"
    for bid, s, ln in blocks:
        assert (st[s : s + ln] == np.arange(ln)).all(), f"episode {bid}: step_idx not 0..{ln - 1}"
    return ep, st, blocks


def write_scalars(f: Any, ds: Any, n_rows: int, ep: np.ndarray, st: np.ndarray) -> list[str]:
    cols = [c for c in SCALAR_COLS if c in ds.column_names]
    for c in cols:
        f.create_dataset(c, data=np.asarray(ds.get_col_data(c))[:n_rows])
    f.create_dataset("episode_idx", data=ep[:n_rows])
    f.create_dataset("step_idx", data=st[:n_rows])
    return cols


def run(cfg: DictConfig) -> None:
    args = phase_config(cfg, "preparation")
    train_out = Path(args.train_out)
    eval_out = Path(args.eval_out)

    ds = swm.data.load_dataset(args.dataset)
    ep, st, blocks = episode_blocks(ds)
    n_total = len(blocks)
    logger.info(f"Loaded dataset episodes={n_total} rows={len(ep)}")

    if not train_out.exists():
        ids = np.array([b[0] for b in blocks])
        assert ids.min() == 0 and ids.max() == n_total - 1, "ids not 0..N-1"
        off_by_id = np.zeros(n_total, dtype=np.int64)
        len_by_id = np.zeros(n_total, dtype=np.int64)
        for bid, s, ln in blocks:
            off_by_id[bid] = s
            len_by_id[bid] = ln
        tmp = train_out.with_name(f"{train_out.name}.tmp")
        with h5py.File(tmp, "w") as f:
            cols = write_scalars(f, ds, len(ep), ep, st)
            f.create_dataset("ep_len", data=len_by_id)
            f.create_dataset("ep_offset", data=off_by_id)
        tmp.replace(train_out)
        logger.success(f"Wrote {args.train_out}: episodes={n_total} id_indexed=true columns={cols}")
    else:
        logger.info(f"Training HDF5 already exists: {args.train_out}")

    if not eval_out.exists():
        n_eval = min(args.eval_episodes, n_total)
        sub = blocks[:n_eval]
        n_rows = sub[-1][1] + sub[-1][2]
        assert all(s + ln <= n_rows for _, s, ln in sub)
        tmp = eval_out.with_name(f"{eval_out.name}.tmp")
        with h5py.File(tmp, "w") as f:
            cols = write_scalars(f, ds, n_rows, ep, st)
            f.create_dataset("ep_len", data=np.array([b[2] for b in sub], dtype=np.int64))
            f.create_dataset("ep_offset", data=np.array([b[1] for b in sub], dtype=np.int64))
            from PIL import Image

            first = np.array(Image.open(BytesIO(bytes(ds.get_row_data([0])["pixels"][0]))))
            dset = f.create_dataset(
                "pixels",
                shape=(n_rows, *first.shape),
                dtype=np.uint8,
                chunks=(64, *first.shape),
            )
            B = 512
            for i in range(0, n_rows, B):
                rows = ds.get_row_data(list(range(i, min(i + B, n_rows))))
                imgs = [np.array(Image.open(BytesIO(bytes(q)))) for q in rows["pixels"]]
                dset[i : i + len(imgs)] = np.stack(imgs)
                if (i // B) % 20 == 0:
                    logger.info(f"Decoding pixels rows={i}/{n_rows}")
        tmp.replace(eval_out)
        logger.success(f"Wrote {args.eval_out}: episodes={n_eval} rows={n_rows} columns={cols}+pixels")
    else:
        logger.info(f"Evaluation HDF5 already exists: {args.eval_out}")
    logger.success("HDF5 build completed")
