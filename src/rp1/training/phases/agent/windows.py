"""Planning problems drawn from a latent cache and its action h5, for the baseline trainers.

A problem is a three-frame latent history from a cache at one row per action
block, the two preceding real action blocks, and a goal from the same episode
within ``max_delta`` blocks or, with probability ``p_cross``, from another
episode. This is the sampling of :mod:`rp1.methods.rp1.train`, which
keeps its own copy so that its random stream is unaffected by this module.

Action statistics ignore NaNs: several public datasets pad every episode's
terminal step with NaN actions.
"""

from contextlib import suppress
from dataclasses import dataclass

import h5py

with suppress(ImportError):
    import hdf5plugin  # noqa: F401  (registers HDF5 compression filters, e.g. cube h5)

import numpy as np
import torch

from rp1.data import LatentCache

__all__ = ["WindowBatch", "WindowSampler"]


@dataclass(frozen=True)
class WindowBatch:
    """One training batch of planning problems."""

    z_hist: torch.Tensor  # (B, 3, D) latent history
    a_hist: torch.Tensor  # (B, 2, a_dim) preceding real action blocks
    z_goal: torch.Tensor  # (B, D) goal latent
    a_ref: torch.Tensor  # (B, H, a_dim) the data's next-H real action blocks


class WindowSampler:
    """Draw planning problems from a latent cache plus its action h5."""

    def __init__(
        self,
        cache: str,
        h5: str,
        horizon: int,
        max_delta: int,
        p_cross: float,
        frameskip: int,
        device: str | torch.device,
        mmap: bool,
        seed: int,
    ) -> None:
        self.horizon = int(horizon)
        self.max_delta = int(max_delta)
        self.p_cross = float(p_cross)
        self.frameskip = int(frameskip)
        self.device = device
        self.rng = np.random.default_rng(seed)

        latents = LatentCache.load(cache, mmap=mmap)
        self.z = latents.z.to(device).float()
        episodes = latents.episodes()
        # goals and reference blocks are clamped to the episode end, so an episode only
        # needs the two-frame history and the [2, L-2) query range; horizon-matched
        # TwoRoom episodes are shorter than the goal band
        keys = [key for key in episodes if len(episodes[key]) > 4]
        if not keys:
            lengths = [len(rows) for rows in episodes.values()]
            raise ValueError(f"no episode in {cache} has more than 4 rows (longest has {max(lengths, default=0)})")
        self.ep_rows = {episode: np.asarray(rows) for episode, rows in episodes.items() if episode in set(keys)}
        self.ep_ids = np.array(keys)

        with h5py.File(h5, "r") as handle:
            actions = handle["action"][:]
            self.ep_off = handle["ep_offset"][:]
            self.ep_len = handle["ep_len"][:] if "ep_len" in handle else None
        self.action_mean = np.nanmean(actions, 0)
        self.action_std = np.nanstd(actions, 0) + 1e-6
        self.actions = ((actions - self.action_mean) / self.action_std).astype(np.float32)
        self.a_dim = int(actions.shape[-1]) * self.frameskip

    @property
    def latent_dim(self) -> int:
        return int(self.z.shape[-1])

    def action_bounds(self, action_range: float) -> tuple[np.ndarray, np.ndarray]:
        """The environment's action limits in the trainer's z-scored units.

        Actions are z-scored by the dataset statistics, so the env's raw box
        ``[-action_range, action_range]`` maps to a per-dimension asymmetric
        range; a symmetric clip would search a different set than the environment
        allows. Blocks flatten ``frameskip`` consecutive steps, so the per-dimension
        bounds tile.
        """
        low = (-action_range - self.action_mean) / self.action_std
        high = (action_range - self.action_mean) / self.action_std
        return np.tile(low, self.frameskip), np.tile(high, self.frameskip)

    def _block(self, episode: int, index: int) -> np.ndarray:
        offset = self.frameskip * index
        if self.ep_len is not None:
            offset = min(offset, max(0, int(self.ep_len[episode]) - self.frameskip))
        start = int(self.ep_off[episode] + offset)
        return np.asarray(self.actions[start : start + self.frameskip]).reshape(-1)

    def sample(self, batch: int) -> WindowBatch:
        z_hist: list[torch.Tensor] = []
        a_hist: list[np.ndarray] = []
        z_goal: list[torch.Tensor] = []
        a_ref: list[np.ndarray] = []
        for _ in range(batch):
            episode = int(self.ep_ids[self.rng.integers(len(self.ep_ids))])
            rows = self.ep_rows[episode]
            length = len(rows)
            t = int(self.rng.integers(2, length - 2))
            z_hist.append(torch.stack([self.z[rows[t - 2]], self.z[rows[t - 1]], self.z[rows[t]]]))
            a_hist.append(np.stack([self._block(episode, t - 2), self._block(episode, t - 1)]))
            # clamp at length - 2, the last full block in the episode: block length - 1
            # starts at the final primitive step, so its read spills into the next episode
            a_ref.append(np.stack([self._block(episode, min(t + k, length - 2)) for k in range(self.horizon)]))
            if self.rng.random() < self.p_cross:
                other = int(self.ep_ids[self.rng.integers(len(self.ep_ids))])
                other_rows = self.ep_rows[other]
                z_goal.append(self.z[other_rows[self.rng.integers(len(other_rows))]])
            else:
                delta = int(self.rng.integers(1, self.max_delta + 1))
                z_goal.append(self.z[rows[min(t + delta, length - 1)]])
        return WindowBatch(
            z_hist=torch.stack(z_hist),
            a_hist=torch.from_numpy(np.stack(a_hist)).to(self.device),
            z_goal=torch.stack(z_goal),
            a_ref=torch.from_numpy(np.stack(a_ref)).to(self.device).float(),
        )
