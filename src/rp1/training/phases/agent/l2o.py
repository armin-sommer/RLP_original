"""Train L2O-MPC by DAgger against a many-sample MPPI expert.

L2O-MPC (Sacks & Boots, ICRA 2022, arXiv:2212.02603) learns the update rule
of a sampling-based MPC optimizer as a gated replacement of the hand-written
MPPI step (:class:`rp1.core.agent.planner.l2o.L2ONet`). Unlike DMPO's PPO or
rp1's pathwise training, its training method transfers to the offline setting
without modification: the learner imitates an MPPI **expert that is
the same optimizer with a larger sample budget**, and that expert is exactly
computable here (the hand-written update through the frozen world model and
the frozen critic). No policy gradient, no gradient through the rollouts:
supervised regression on the expert's updated mean, with DAgger mixing so the
learner is eventually trained on the iterate distribution it induces itself.

Per training step, one planning problem batch runs ``iterations`` inner steps:

    target_i  = MPPI_N( mean_i )                  # expert, N samples
    loss     += || m_theta(mean_i, costs_M) - target_i ||^2   # learner, M samples
    mean_{i+1} = target_i  with probability beta_k  else  learner's update

``beta_k = decay^k`` over ``dagger_rounds`` rounds (paper: 0.8^k over 20),
so early training visits the expert's iterate states (the paper's bootstrap)
and later training visits the learner's own.

The paper's sequential state is the receding-horizon loop with a warm-start
shift; the evaluation protocol here is open loop (cold start every decision), so
the iterate sequence is the deployed inner loop itself, the same ``K`` chained
updates the solver runs.

The critic is never updated here: L2O-MPC is a baseline whose job is to
isolate the planner, so it plans against the same offline critic
(``value_td``) the other baselines are evaluated with.

Example, once the agent pipeline's value stage has produced the caches and ``value_td``::

    pixi run posttrain --config-name phases/agent/l2o \
        training.wm=assets/core/world_model/tworoom_lewm \
        training.cache=$RP1_DATA_HOME/caches/tworoom_fs5.pt \
        training.h5=$RP1_DATA_HOME/caches/tworoom_actions.h5 \
        training.init_value=<run>/checkpoints/value_td
"""

import copy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig

from rp1.core.agent.planner.dmpo import gaussian_halton
from rp1.core.agent.planner.l2o import L2ONet, mppi_update
from rp1.core.agent.value.temporal import ValueFunction, trajectory_value, windowed_terminal_value
from rp1.core.world_model.base import LatentWorldModel
from rp1.core.world_model.rollout import rollout_traj
from rp1.training.harness.checkpointing import load_metric, load_pretrained, save_metric
from rp1.training.harness.schedule import cosine_interpolate
from rp1.training.phases.agent.windows import WindowSampler
from rp1.utils.config import phase_config
from rp1.utils.device import pick_device
from rp1.utils.logging import logger

__all__ = ["dagger_beta"]


def dagger_beta(step: int, steps: int, rounds: int, decay: float) -> float:
    """The paper's mixing schedule: ``decay^k`` with ``k`` the DAgger round."""
    if steps <= 0 or rounds <= 0:
        return 0.0
    k = min(step * rounds // steps, rounds - 1)
    return float(decay**k)


def run(cfg: DictConfig) -> None:
    a = phase_config(cfg, "training", cfg.core.agent.planner)
    if not isinstance(a, DictConfig):
        raise TypeError("merged planner configuration must be a mapping")
    if not a.init_value:
        raise ValueError("the l2o phase trains against a frozen value: pass training.init_value=<value_td>")
    if a.temporal_objective not in {"terminal", "tel-exact", "tel-stopprev"}:
        raise ValueError(f"unsupported temporal objective: {a.temporal_objective}")
    if int(a.expert_samples) < int(a.num_samples):
        raise ValueError("the DAgger expert must not be weaker than the learner: expert_samples >= num_samples")
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
    context = critic_dim // sampler.latent_dim
    if context > 1:
        if a.temporal_objective != "terminal":
            raise ValueError("window critics support the terminal objective only")
        if context > a.horizon:
            raise ValueError(f"critic context {context} exceeds the plan horizon {a.horizon}")
        logger.info(f"L2O scoring a {context}-frame value window")
    critic.eval()
    critic.requires_grad_(False)
    value = cast(ValueFunction, critic)

    net = L2ONet(
        horizon=a.horizon,
        a_dim=sampler.a_dim,
        num_samples=a.num_samples,
        hidden=a.hidden,
        amax=a.action_limit,
        init_std=a.init_std,
        dropout=a.dropout,
        learn_std=a.learn_std,
        gate_bias=a.gate_bias,
        init_scale=a.init_scale,
        halton=a.halton,
        seed_val=a.seed_val,
    ).to(dev)
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.actor_lr, weight_decay=a.weight_decay)

    # The expert's own fixed sample set — wider than the learner's, never part
    # of the checkpoint (the expert exists only at training time).
    plan_size = int(a.horizon) * sampler.a_dim
    expert_base = gaussian_halton(int(a.expert_samples) - 1, plan_size, int(a.seed_val) + 1, device=dev)
    expert_base = torch.cat([torch.zeros(1, plan_size, device=dev), expert_base], dim=0).view(
        int(a.expert_samples), int(a.horizon), sampler.a_dim
    )

    def score(
        z_hist: torch.Tensor,
        a_hist: torch.Tensor,
        z_goal: torch.Tensor,
        plan: torch.Tensor,
    ) -> torch.Tensor:
        """Cost of one plan per problem (no gradients needed anywhere here)."""
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

    generator = torch.Generator(device="cpu").manual_seed(int(a.seed) + 1)

    def step(beta: float) -> tuple[float, float, float]:
        batch = sampler.sample(a.batch)
        z_hist, a_hist, z_goal = batch.z_hist, batch.a_hist, batch.z_goal
        cost = cost_fn(z_hist, a_hist, z_goal)
        mean, std = net.initial(z_hist.shape[0], dev)
        # one coin per step: this iterate sequence is either the expert's
        # (probability beta_k) or the learner's own — DAgger's state mixing
        follow_expert = bool(torch.rand((), generator=generator).item() < beta)
        loss = torch.zeros((), device=dev)
        for _ in range(int(a.iterations)):
            with torch.no_grad():
                learner_costs = cost(net.plans(mean, std))
                expert_plans = (mean.unsqueeze(1) + std.unsqueeze(1) * expert_base).clamp(-net.amax, net.amax)
                target = mppi_update(
                    mean,
                    expert_plans,
                    cost(expert_plans),
                    temperature=a.temperature,
                    step_size=a.step_size,
                    scale_costs=a.scale_costs,
                )
            predicted, std = net(mean, std, learner_costs)
            loss = loss + torch.nn.functional.mse_loss(predicted, target)
            mean = target if follow_expert else predicted.detach()
            std = std.detach()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(net.parameters(), a.max_grad_norm)
        optimizer.step()
        with torch.no_grad():
            final = score(z_hist, a_hist, z_goal, mean).mean()
        return float(loss.item()), float(final.item()), beta

    # saved before training, since snapshots record its path; a copy, as training still uses the original
    value_checkpoint = save_metric(
        copy.deepcopy(critic).cpu(), run_name=a.output.value_checkpoint, cache_dir=a.run.directory
    )
    logger.success(f"Copied the frozen critic to {value_checkpoint}")
    planner_checkpoint = Path(a.run.checkpoints) / a.output.planner_checkpoint

    def payload() -> dict[str, object]:
        return {
            "kind": "l2o",
            "sd": {key: value.detach().cpu().clone() for key, value in net.state_dict().items()},
            "z_dim": sampler.latent_dim,
            "horizon": int(a.horizon),
            "a_dim": sampler.a_dim,
            "iters": int(a.iterations),
            "num_samples": int(a.num_samples),
            "hidden": int(a.hidden),
            "amax": float(a.action_limit),
            "init_std": float(a.init_std),
            "learn_std": bool(a.learn_std),
            "gate_bias": float(a.gate_bias),
            "halton": bool(a.halton),
            "seed_val": int(a.seed_val),
            "value": str(a.output.value_checkpoint),
            "value_context": context,
            "temporal_objective": str(a.temporal_objective),
        }

    for i in range(a.steps):
        lr = cosine_interpolate(a.actor_lr, a.actor_lr_final, i, a.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        imitation, final_cost, beta = step(dagger_beta(i, int(a.steps), int(a.dagger_rounds), float(a.beta_decay)))
        if i % 100 == 0:
            logger.info(f"step {i}: imitation {imitation:.4f} E_final {final_cost:.3f} beta {beta:.2f} lr {lr:.2e}")
        if a.save_every and (i + 1) % a.save_every == 0 and i + 1 < a.steps:
            snapshot = planner_checkpoint.with_name(f"{planner_checkpoint.stem}_step{i + 1}.pt")
            torch.save(payload(), snapshot)
            logger.info(f"Saved snapshot at step {i + 1} to {snapshot}")

    net.eval()
    torch.save(payload(), planner_checkpoint)
    logger.success(f"Saved the L2O-MPC optimizer to {planner_checkpoint}")
