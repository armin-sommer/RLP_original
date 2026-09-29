"""Train a goal-conditioned value on a latent cache.

Every learner produces a module with ``cost(z_pred, z_goal)``:

* ``regression``  -- horizon-matched temporal regression
* ``shuffled``    -- the same regression on shuffled labels, the negative control
* ``td``          -- offline goal-conditioned temporal-distance TD
* ``contrastive`` -- contrastive value learning (InfoNCE)
* ``l2``          -- latent L2 distance, untrained

Example::

    pixi run posttrain --config-name phases/agent/metric training.cache=<cache> \
        training.learner=regression training.scale=100
"""

import numpy as np
from omegaconf import DictConfig
from torch import nn

from rp1.core.agent.value import build_metric
from rp1.data import LatentCache
from rp1.training.harness.checkpointing import save_metric
from rp1.training.phases.agent import learners
from rp1.training.phases.agent.learners.contrastive import ContrastiveConfig
from rp1.training.phases.agent.learners.regression import RegressionConfig
from rp1.training.phases.agent.learners.td import TDConfig
from rp1.utils.config import phase_config
from rp1.utils.device import pick_device
from rp1.utils.logging import logger


def run(cfg: DictConfig) -> None:
    args = phase_config(cfg, "training")
    value = cfg.core.agent.value  # the architecture; ``args`` holds the training settings

    device = pick_device(args.device)
    base_cache = LatentCache.load(args.cache, mmap=bool(args.cache_mmap)).first_episodes(args.max_episodes)
    cache = base_cache.windowed(int(args.window_frames), int(args.window_lag))
    logger.info(f"Loaded cache: {len(cache.z)} latents dim={cache.latent_dim} on {device}")

    module: nn.Module
    if args.learner == "l2":
        module = build_metric("l2", cache.latent_dim, {})
    elif args.learner in ("regression", "shuffled"):
        scale = args.scale
        if scale is None:  # about the episode horizon, so the targets are not dwarfed
            lengths = [len(rows) for rows in cache.episodes().values()]
            scale = float(np.percentile(lengths, 90))
        regression_cfg = RegressionConfig(
            hidden_dim=value.hidden_dim,
            depth=value.depth,
            scale=scale,
            softplus=value.softplus,
            symmetric=value.symmetric,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            steps=args.steps,
            n_buckets=args.n_buckets,
            max_delta=args.max_delta,
            shuffle_labels=args.learner == "shuffled",
            seed=args.seed,
            huber_beta=args.huber_beta,
        )
        module = learners.regression.fit(cache, regression_cfg, device)
    elif args.learner == "td":
        td_cfg = TDConfig(
            head=value.head,
            hidden_dim=value.hidden_dim,
            depth=value.depth,
            embed_dim=value.embedding_dim,
            n_step=args.n_step,
            gamma=args.gamma,
            expectile=args.expectile,
            p_cross=args.p_cross,
            balanced=(not args.no_balanced),
            max_delta=args.max_delta,
            n_buckets=args.n_buckets,
            batch_size=args.batch_size,
            steps=args.steps,
            save_every=args.save_every,
            seed=args.seed,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            tau=args.target_update_rate,
            huber_beta=args.huber_beta,
            symmetric=value.symmetric,
            num_components=value.num_components,
            softplus=value.softplus,
            sym_frac=value.sym_frac,
            alpha_init=value.alpha_init,
            near_frac=args.near_frac,
            near_max=args.near_max,
        )

        def snapshot(head: nn.Module, step: int) -> None:
            path = save_metric(head, run_name=f"{args.output.checkpoint}_step{step}", cache_dir=args.run.directory)
            logger.info(f"Saved TD snapshot at step {step} to {path}")

        module = learners.td.fit(cache, td_cfg, device, snapshot)
    else:  # contrastive
        contrastive_cfg = ContrastiveConfig(
            hidden_dim=value.hidden_dim,
            rep_dim=value.representation_dim,
            depth=value.depth,
            gamma=args.contrastive_gamma,
            temperature=args.temperature,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed,
        )
        module = learners.contrastive.fit(cache, contrastive_cfg, device)
    checkpoint = save_metric(module.cpu(), run_name=args.output.checkpoint, cache_dir=args.run.directory)
    logger.success(f"Saved {args.learner} metric to {checkpoint}")
