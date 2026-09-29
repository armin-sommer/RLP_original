"""Collect a mixed TwoRoom dataset of expert and random episodes.

The random episodes run into the central wall, so a world model trained on them
learns the wall: blocked plans then end on the wrong side of it, where latent
Euclidean distance mis-ranks them and a learned reachability value does not.

Example::

    pixi run prepare job=collect_tworoom_mixed \
        preparation.expert=300 preparation.random=300 preparation.out=tworoom_mixed.lance
"""

from pathlib import Path

import numpy as np
import stable_worldmodel as swm
from omegaconf import DictConfig
from stable_worldmodel.envs.two_room import ExpertPolicy
from stable_worldmodel.policy import RandomPolicy

from rp1.environment import World
from rp1.utils.config import phase_config
from rp1.utils.logging import logger


def run(cfg: DictConfig) -> None:
    args = phase_config(cfg, "preparation")

    path = Path(swm.data.utils.get_cache_dir()) / "datasets" / args.out
    rng = np.random.default_rng(args.seed)

    world = World(
        "swm/TwoRoom-v1",
        num_envs=args.num_envs,
        max_episode_steps=args.max_steps,
        image_shape=(224, 224),
        render_mode="rgb_array",
    )

    world.set_policy(ExpertPolicy(action_noise=2.0, action_repeat_prob=0.05))
    world.collect(path, episodes=args.expert, seed=rng.integers(0, 1_000_000).item())

    world.set_policy(RandomPolicy())
    world.collect(path, episodes=args.random, seed=rng.integers(0, 1_000_000).item())

    logger.success(f"Collected mixed TwoRoom dataset expert={args.expert} random={args.random} path={path}")
