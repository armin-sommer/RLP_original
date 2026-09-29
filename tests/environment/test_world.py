"""First-hit scoring at an explicit tolerance.

Reacher's qpos-match task hardcodes a 0.05 rad termination threshold, so the
paper's tau=0.1 column cannot be produced from environment termination. The
evaluation world scores those columns itself; this guards the rule it applies.
"""

from __future__ import annotations

import numpy as np

from rp1.environment import World
from rp1.environment.world import _resize_images_like_env


def test_worst_joint_rule() -> None:
    current = np.array([[0.00, 0.00], [0.00, 0.08], [0.20, 0.00]])
    target = np.zeros_like(current)
    assert World.threshold_hits(current, target, 0.1).tolist() == [True, True, False]
    assert World.threshold_hits(current, target, 0.05).tolist() == [True, False, False]


def test_history_axis_uses_the_current_step() -> None:
    # infos carry a per-env history axis; only the current step counts
    current = np.zeros((2, 3, 2))
    current[:, :-1] = 5.0  # stale frames far from the goal
    target = np.zeros((2, 3, 2))
    assert World.threshold_hits(current, target, 0.05).tolist() == [True, True]


def test_tolerance_is_strict() -> None:
    current = np.array([[0.05]])
    assert World.threshold_hits(current, np.zeros_like(current), 0.05).tolist() == [False]


def test_dataset_images_are_resized_to_environment_shape() -> None:
    images = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    env_pixels = np.zeros((2, 1, 16, 12, 3), dtype=np.uint8)
    resized = _resize_images_like_env(images, env_pixels)
    assert resized.shape == (2, 16, 12, 3)
    assert resized.dtype == np.uint8
