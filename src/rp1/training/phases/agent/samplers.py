"""Samplers of training pairs over a :class:`LatentCache`, one per value learner.

Sampling is part of the method: horizon-matched supervision draws balanced,
full-horizon temporal separations, so the value sees the same long-range
reachability scale the planner queries.

* :class:`BalancedHorizonPairSampler` -- regression: pairs ``(z_i, z_j)`` labelled ``|t_i - t_j|``.
* :class:`NStepGoalSampler` -- TD: n-step transitions with hindsight goals.
* :class:`GeometricFutureSampler` -- contrastive: an anchor and a geometric-future positive.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import torch

from rp1.data import LatentCache


class PairBatch(TypedDict):
    z_i: torch.Tensor
    z_j: torch.Tensor
    label: torch.Tensor


class NStepBatch(TypedDict):
    z_t: torch.Tensor
    z_tn: torch.Tensor
    z_g: torch.Tensor
    n_eff: torch.Tensor
    reached: torch.Tensor
    dist: torch.Tensor
    # cache rows of z_t / z_tn / z_g, from which window values rebuild their inputs
    t_idx: torch.Tensor
    tn_idx: torch.Tensor
    g_idx: torch.Tensor


class FutureBatch(TypedDict):
    z_s: torch.Tensor
    z_g: torch.Tensor


class _BaseSampler:
    def __init__(self, cache: LatentCache, seed: int, min_len: int):
        self.cache = cache
        self.z = cache.z
        self.rng = np.random.default_rng(seed)
        eps = cache.episodes()
        self.episodes = {e: rows for e, rows in eps.items() if len(rows) >= min_len}
        self.ep_ids = np.array(sorted(self.episodes.keys()))
        self.ep_lens = np.array([len(self.episodes[e]) for e in self.ep_ids])
        self.max_len = int(self.ep_lens.max())
        assert len(self.ep_ids) > 0, "no episodes long enough to sample"


class BalancedHorizonPairSampler(_BaseSampler):
    """Balanced full-horizon temporal-separation pairs (regression supervision).

    Args:
        n_buckets: number of separation buckets used to equalise coverage.
        max_delta: optional cap on the temporal separation (the paper's
            ``max-Delta`` ablation; ``None`` = full episode horizon).
        random_order: if ``True`` randomly swap ``(z_i, z_j)`` per pair so the
            head is encouraged to be symmetric.
    """

    def __init__(
        self,
        cache: LatentCache,
        n_buckets: int,
        max_delta: int | None,
        random_order: bool,
        seed: int,
    ) -> None:
        super().__init__(cache, seed=seed, min_len=2)
        self.n_buckets = n_buckets
        self.max_delta = max_delta
        self.random_order = random_order

    def _sample_delta(self, L: int) -> int:
        hi = L - 1
        if self.max_delta is not None:
            hi = min(hi, self.max_delta)
        if hi < 1:
            return 1
        # balanced over buckets: choose a bucket overlapping [1, hi], then uniform within
        edges = np.linspace(1, hi + 1, self.n_buckets + 1)
        b = self.rng.integers(0, self.n_buckets)
        lo_b, hi_b = edges[b], edges[b + 1]
        delta = int(
            self.rng.integers(
                int(np.floor(lo_b)),
                max(int(np.ceil(hi_b)), int(np.floor(lo_b)) + 1),
            )
        )
        return int(np.clip(delta, 1, hi))

    def sample(self, batch_size: int) -> PairBatch:
        i_idx = np.empty(batch_size, dtype=np.int64)
        j_idx = np.empty(batch_size, dtype=np.int64)
        labels = np.empty(batch_size, dtype=np.float32)
        for b in range(batch_size):
            e = self.ep_ids[self.rng.integers(0, len(self.ep_ids))]
            rows = self.episodes[e]
            L = len(rows)
            delta = self._sample_delta(L)
            t = int(self.rng.integers(0, L - delta))
            ri, rj = rows[t], rows[t + delta]
            if self.random_order and self.rng.random() < 0.5:
                ri, rj = rj, ri
            i_idx[b], j_idx[b], labels[b] = ri, rj, delta
        return {
            "z_i": self.z[i_idx],
            "z_j": self.z[j_idx],
            "label": torch.from_numpy(labels),
        }


class NStepGoalSampler(_BaseSampler):
    """HER n-step transitions with balanced full-horizon hindsight goals.

    For each draw returns ``(z_t, z_tn, z_g, n_eff, reached, dist)`` where:
      * ``z_tn`` is the state ``n_eff = min(n, L-1-t)`` steps ahead (n-step bootstrap target),
      * ``z_g`` is a **hindsight goal**: with prob ``p_cross`` a random cross-episode state
        (enables Bellman stitching of long/cross-room pairs), else a future state in the
        same episode at a **balanced full-horizon** offset ``δ`` (paper's lesson applied to HER),
      * ``reached`` / ``dist``: if the in-episode goal is within ``n_eff`` steps, the exact
        distance ``δ`` is known (Monte-Carlo target); otherwise bootstrap from ``z_tn``.

    ``near_frac`` of the in-episode goals are drawn 1..``near_max`` steps ahead, so the last
    steps before a goal are fitted too; balanced buckets rarely land there.
    """

    def __init__(
        self,
        cache: LatentCache,
        n_step: int,
        p_cross: float,
        n_buckets: int,
        balanced: bool,
        seed: int,
        max_delta: int | None,
        near_frac: float,
        near_max: int,
    ) -> None:
        super().__init__(cache, seed=seed, min_len=2)
        self.n = n_step
        self.max_delta = max_delta
        self.p_cross = p_cross
        self.n_buckets = n_buckets
        self.balanced = balanced
        self.n_total = len(cache.z)
        self.near_frac = float(near_frac)
        self.near_max = int(near_max)

    def _offset(self, hi: int) -> int:
        if hi < 1:
            return 1
        if not self.balanced:
            return int(self.rng.integers(1, hi + 1))
        edges = np.linspace(1, hi + 1, self.n_buckets + 1)
        b = self.rng.integers(0, self.n_buckets)
        lo_b, hi_b = (
            int(np.floor(edges[b])),
            max(int(np.ceil(edges[b + 1])), int(np.floor(edges[b])) + 1),
        )
        return int(np.clip(self.rng.integers(lo_b, hi_b), 1, hi))

    def sample(self, batch_size: int) -> NStepBatch:
        t_idx = np.empty(batch_size, np.int64)
        tn_idx = np.empty(batch_size, np.int64)
        g_idx = np.empty(batch_size, np.int64)
        n_eff = np.empty(batch_size, np.float32)
        reached = np.zeros(batch_size, np.float32)
        dist = np.zeros(batch_size, np.float32)
        for b in range(batch_size):
            e = self.ep_ids[self.rng.integers(0, len(self.ep_ids))]
            rows = self.episodes[e]
            L = len(rows)
            t = int(self.rng.integers(0, L - 1))
            ne = min(self.n, L - 1 - t)
            t_idx[b], tn_idx[b], n_eff[b] = rows[t], rows[t + ne], ne
            if self.rng.random() < self.p_cross:
                g_idx[b] = int(self.rng.integers(0, self.n_total))  # cross-episode goal
            else:
                _hi = L - 1 - t
                if self.max_delta is not None:
                    _hi = min(_hi, self.max_delta)
                if self.near_frac > 0 and self.rng.random() < self.near_frac:
                    delta = int(self.rng.integers(1, min(self.near_max, _hi) + 1))
                else:
                    delta = self._offset(_hi)
                g_idx[b] = rows[t + delta]
                if delta <= ne:  # goal reached within the n-step window
                    reached[b], dist[b] = 1.0, float(delta)
        return {
            "z_t": self.z[t_idx],
            "z_tn": self.z[tn_idx],
            "z_g": self.z[g_idx],
            "n_eff": torch.from_numpy(n_eff),
            "reached": torch.from_numpy(reached),
            "dist": torch.from_numpy(dist),
            "t_idx": torch.from_numpy(t_idx),
            "tn_idx": torch.from_numpy(tn_idx),
            "g_idx": torch.from_numpy(g_idx),
        }


class GeometricFutureSampler(_BaseSampler):
    """Anchor / geometric-future-goal pairs for contrastive value learning.

    ``k ~ Geom(1 - gamma)`` (clipped to the episode end). Negatives are taken
    in-batch by the contrastive loss (every other goal in the batch).
    """

    def __init__(self, cache: LatentCache, gamma: float, seed: int):
        super().__init__(cache, seed=seed, min_len=2)
        self.gamma = gamma

    def sample(self, batch_size: int) -> FutureBatch:
        s_idx = np.empty(batch_size, dtype=np.int64)
        g_idx = np.empty(batch_size, dtype=np.int64)
        for b in range(batch_size):
            e = self.ep_ids[self.rng.integers(0, len(self.ep_ids))]
            rows = self.episodes[e]
            L = len(rows)
            t = int(self.rng.integers(0, L - 1))
            k = int(self.rng.geometric(1.0 - self.gamma))  # >= 1
            g = min(t + k, L - 1)
            s_idx[b], g_idx[b] = rows[t], rows[g]
        return {"z_s": self.z[s_idx], "z_g": self.z[g_idx]}


__all__ = [
    "BalancedHorizonPairSampler",
    "NStepGoalSampler",
    "GeometricFutureSampler",
    "FutureBatch",
    "NStepBatch",
    "PairBatch",
]
