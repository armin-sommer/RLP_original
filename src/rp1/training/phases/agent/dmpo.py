"""Train DMPO's learned MPC inner loop against a frozen goal-conditioned critic.

DMPO (Sacks et al., ICRA 2024, arXiv:2310.04590) learns the update rule of an
MPC optimizer: at every decision it samples ``N`` plans, rolls them out, and an
MLP turns the ``N`` costs plus the current sampling distribution into the next
distribution (:class:`rp1.core.agent.planner.dmpo.DMPONet`). This trainer fits that
rule in the setting rp1's planner is trained in: frozen pretrained world model,
frozen quasimetric value supplied through ``init_value``, latent windows drawn
from the offline cache:

    loss = V( z_T( mu_K ), z_g )

with ``mu_K`` the mean the learned optimizer arrives at after ``iterations``
inner steps. The ``N`` sampled rollouts are computed without gradients (their
costs are *features* of the update rule); the gradient reaches the actor and
shift networks through ``mu_K`` and the differentiable world-model unroll.

**Deviation from the paper.** DMPO is published with PPO, because its costs
come from a real quadrotor and no analytic gradient exists. The world model here
is differentiable, so the same networks are trained by pathwise gradients, as
rp1's planner is: identical data, frozen value and objective, differing only in
the learned planning procedure. The actor's stochastic search distributions
(PPO's exploration noise) are therefore absent; everything on the forward path
is the reference computation. :mod:`rp1.training.phases.agent.dmpo_ppo` trains
with the paper's objective.

The critic is never updated here: DMPO is a baseline whose job is to isolate
the planner, so it plans against the same offline critic (``value_td``) the
other baselines are evaluated with.

Example, once the agent pipeline's value stage has produced the caches and ``value_td``::

    pixi run posttrain --config-name phases/agent/dmpo \
        training.wm=assets/core/world_model/tworoom_lewm \
        training.cache=$RP1_DATA_HOME/caches/tworoom_fs5.pt \
        training.h5=$RP1_DATA_HOME/caches/tworoom_actions.h5 \
        training.init_value=<run>/checkpoints/value_td
"""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig

from rp1.core.agent.planner.dmpo import DMPONet
from rp1.core.agent.value.temporal import ValueFunction, trajectory_value, windowed_terminal_value
from rp1.core.world_model.base import LatentWorldModel
from rp1.core.world_model.rollout import rollout_traj
from rp1.training.harness.checkpointing import load_metric, load_pretrained, save_metric
from rp1.training.harness.schedule import cosine_interpolate
from rp1.training.phases.agent.windows import WindowSampler
from rp1.utils.config import phase_config
from rp1.utils.device import pick_device
from rp1.utils.logging import logger


def run(cfg: DictConfig) -> None:
    a = phase_config(cfg, "training", cfg.core.agent.planner)
    if not isinstance(a, DictConfig):
        raise TypeError("merged planner configuration must be a mapping")
    if not a.init_value:
        raise ValueError("the dmpo phase trains against a frozen value: pass training.init_value=<value_td>")
    if a.temporal_objective not in {"terminal", "tel-exact", "tel-stopprev"}:
        raise ValueError(f"unsupported temporal objective: {a.temporal_objective}")
    dev = pick_device(str(a.device))
    torch.manual_seed(a.seed)

    wm_module = load_pretrained(a.wm).to(dev).eval()
    wm_module.requires_grad_(False)
    wm = cast(LatentWorldModel, wm_module)

    sampler = WindowSampler(
        cache=a.cache,
        h5=a.h5,
        horizon=a.horizon,
        max_delta=a.max_delta,
        p_cross=a.p_cross,
        frameskip=a.frameskip,
        device=dev,
        mmap=bool(a.cache_mmap),
        seed=a.seed,
    )

    critic = load_metric(a.init_value, device=dev)
    critic_dim = int(cast(int, critic.latent_dim))
    if critic_dim % sampler.latent_dim:
        raise ValueError(f"critic latent dim {critic_dim} is not a multiple of {sampler.latent_dim}")
    # window critics (the Reacher three-frame quasimetric) score a stack of the
    # last `context` frames; MetricCost applies the same convention at eval
    context = critic_dim // sampler.latent_dim
    if context > 1:
        if a.temporal_objective != "terminal":
            raise ValueError("window critics support the terminal objective only")
        if context > a.horizon:
            raise ValueError(f"critic context {context} exceeds the plan horizon {a.horizon}")
        logger.info(f"DMPO scoring a {context}-frame value window")
    critic.eval()
    critic.requires_grad_(False)
    value = cast(ValueFunction, critic)

    # the environment's own action limits; action_range=null falls back to a symmetric action_limit
    if a.action_range is None:
        action_lows = action_highs = None
        logger.info(f"DMPO clipping plans to the symmetric fallback +-{float(a.action_limit)}")
    else:
        low, high = sampler.action_bounds(float(a.action_range))
        action_lows = torch.from_numpy(low).float()
        action_highs = torch.from_numpy(high).float()
        logger.info(
            f"DMPO clipping plans to the environment's action bounds in z-scored units: "
            f"[{low.min():.2f}, {high.max():.2f}] across {low.size} plan dimensions"
        )
    net = DMPONet(
        horizon=a.horizon,
        a_dim=sampler.a_dim,
        num_samples=a.num_samples,
        hidden=a.hidden,
        amax=a.action_limit,
        init_std=a.init_std,
        temperature=a.temperature,
        step_size=a.step_size,
        scale_costs=a.scale_costs,
        gated=a.gated,
        residual=a.residual,
        learn_std=a.learn_std,
        use_shift=a.use_shift,
        gate_activation=a.gate_activation,
        init_scale=a.init_scale,
        halton=a.halton,
        seed_val=a.seed_val,
        action_lows=action_lows,
        action_highs=action_highs,
    ).to(dev)
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.actor_lr, weight_decay=a.weight_decay)

    def score(
        z_hist: torch.Tensor,
        a_hist: torch.Tensor,
        z_goal: torch.Tensor,
        plan: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable cost of one plan per problem."""
        trajectory = rollout_traj(wm, z_hist, a_hist, plan)
        if context > 1:
            return windowed_terminal_value(value, trajectory, z_goal, context)
        return trajectory_value(value, trajectory, z_goal, z_hist[:, -1], a.temporal_objective)

    def cost_fn(
        z_hist: torch.Tensor,
        a_hist: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        chunk = int(a.cost_chunk)

        def cost(plans: torch.Tensor) -> torch.Tensor:
            batch, samples = plans.shape[0], plans.shape[1]
            flat = plans.reshape(batch * samples, plans.shape[2], plans.shape[3])
            zh = z_hist.repeat_interleave(samples, dim=0)
            ah = a_hist.repeat_interleave(samples, dim=0)
            zg = z_goal.repeat_interleave(samples, dim=0)
            size = flat.shape[0] if chunk <= 0 else chunk
            scored = [
                score(
                    zh[start : start + size],
                    ah[start : start + size],
                    zg[start : start + size],
                    flat[start : start + size],
                )
                for start in range(0, flat.shape[0], size)
            ]
            return torch.cat(scored).view(batch, samples)

        return cost

    def advance(
        z_hist: torch.Tensor,
        a_hist: torch.Tensor,
        plan: torch.Tensor,
        blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute ``blocks`` of ``plan`` in imagination; return the next window."""
        executed = plan[:, :blocks]
        trajectory = rollout_traj(wm, z_hist, a_hist, executed)
        window = torch.cat([z_hist, trajectory], dim=1)[:, -3:]
        history = torch.cat([a_hist, executed], dim=1)[:, -2:]
        return window, history

    def step(warm_start: bool) -> tuple[float, float]:
        batch = sampler.sample(a.batch)
        z_hist, a_hist, z_goal = batch.z_hist, batch.a_hist, batch.z_goal
        mean, std = net.initial(z_hist.shape[0], dev)
        if warm_start:
            # a preceding decision, executed in imagination, so the shift model
            # is trained on the parameters it sees at deployment
            with torch.no_grad():
                previous, previous_std, _ = net.plan(cost_fn(z_hist, a_hist, z_goal), mean, std, a.iterations)
                z_hist, a_hist = advance(z_hist, a_hist, previous, int(a.train_receding))
            mean, std = net.warm_start(previous, previous_std, int(a.train_receding))
        mean, std, history = net.plan(cost_fn(z_hist, a_hist, z_goal), mean, std, a.iterations)
        final = score(z_hist, a_hist, z_goal, mean).mean()
        loss = final
        if a.mean_weight > 0 and len(history) > 1:
            path = torch.stack([score(z_hist, a_hist, z_goal, candidate).mean() for candidate in history])
            loss = loss + a.mean_weight * path.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch 2.7 Tensor.backward lacks a typed signature.
        torch.nn.utils.clip_grad_norm_(net.parameters(), a.max_grad_norm)
        optimizer.step()
        with torch.no_grad():
            first = score(z_hist, a_hist, z_goal, torch.zeros_like(mean)).mean()
        return float(first.item()), float(final.item())

    warm_every = int(a.warm_start_every)
    for i in range(a.steps):
        lr = cosine_interpolate(a.actor_lr, a.actor_lr_final, i, a.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        zero_cost, final_cost = step(warm_every > 0 and i % warm_every == 0)
        if i % 100 == 0:
            logger.info(f"step {i}: E_final {final_cost:.3f} E_zero {zero_cost:.3f} lr {lr:.2e}")

    value_checkpoint = save_metric(critic.cpu(), run_name=a.output.value_checkpoint, cache_dir=a.run.directory)
    logger.success(f"Copied the frozen critic to {value_checkpoint}")
    planner_checkpoint = Path(a.run.checkpoints) / a.output.planner_checkpoint
    net.eval()
    torch.save(
        {
            "kind": "dmpo",
            "sd": net.cpu().state_dict(),
            "z_dim": sampler.latent_dim,
            "horizon": int(a.horizon),
            "a_dim": sampler.a_dim,
            "iters": int(a.iterations),
            "num_samples": int(a.num_samples),
            "hidden": int(a.hidden),
            "amax": float(a.action_limit),
            "action_range": None if a.action_range is None else float(a.action_range),
            "init_std": float(a.init_std),
            "temperature": float(a.temperature),
            "step_size": float(a.step_size),
            "scale_costs": bool(a.scale_costs),
            "gated": bool(a.gated),
            "residual": bool(a.residual),
            "learn_std": bool(a.learn_std),
            "use_shift": bool(a.use_shift),
            "gate_activation": str(a.gate_activation),
            "halton": bool(a.halton),
            "seed_val": int(a.seed_val),
            "value": str(a.output.value_checkpoint),
            "value_context": context,
            "temporal_objective": str(a.temporal_objective),
        },
        planner_checkpoint,
    )
    logger.success(f"Saved the DMPO optimizer to {planner_checkpoint}")
