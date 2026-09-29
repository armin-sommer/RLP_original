"""Offline DMPO — the paper's PPO objective, closed inside the world model.

:mod:`rp1.training.phases.agent.dmpo` fits the learned optimizer by pathwise gradients on a
*single* decision: plan once, differentiate the terminal value. DMPO's own
objective is the **return of the closed-loop system** over many decisions,
optimized with PPO, because on a quadrotor no gradient exists. This trainer
reproduces that objective without any environment interaction, by closing the
loop inside the frozen world model:

    for each decision t:
        theta~_t     = shift model on the previous decision's parameters
        mu_t, S_t    = K learned iterations, each SAMPLED from the actor's
                       search distributions (the on-policy action)
        z_{t+1}      = imagined execution of the first `receding` plan blocks
        r_t          = V(z_t, z_g) - V(z_{t+1}, z_g)      (progress toward goal)

    maximize E[ sum_t gamma^t r_t ]  with PPO + GAE

Everything the paper's algorithm needs is therefore present: stochastic search
distributions over ``(mu, Sigma)``, a critic over the auxiliary state
``(z_t, theta_{t-1})`` (goal-conditioned here), clipped ratios, GAE, and credit
assignment *across* decisions — which is what the shift model and the
covariance update exist to serve and what a single-decision objective cannot
train. No gradient flows through the world model at all: rollouts are
forward-only, exactly as they would be on hardware.

What is still not the paper: the return is measured against the offline
quasimetric critic under the same frozen world model that the inner loop
plans with, not against a real system. Model error is therefore invisible to
training, so this reproduces DMPO's *algorithm* but not its robustness claim
(which is about compensating for model-reality mismatch). Closing that last gap
requires environment rollouts.
"""

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig

from rp1.core.agent.planner.dmpo import DMPOCritic, DMPONet
from rp1.core.agent.value.temporal import ValueFunction, windowed_terminal_value
from rp1.core.world_model.base import LatentWorldModel
from rp1.core.world_model.rollout import rollout_traj
from rp1.training.harness.checkpointing import load_metric, load_pretrained, save_metric
from rp1.training.phases.agent.windows import WindowSampler
from rp1.utils.config import phase_config
from rp1.utils.device import pick_device
from rp1.utils.logging import logger


@dataclass
class Rollout:
    """One batch of imagined closed-loop decisions, flattened over time."""

    state: torch.Tensor  # (T*B, D) latent at the decision
    goal: torch.Tensor  # (T*B, D)
    mean_in: torch.Tensor  # (T*B, H, a) warm-started mean the actor saw
    std_in: torch.Tensor
    plans: torch.Tensor  # (T*B, N, H, a) sampled plans of that iteration
    costs: torch.Tensor  # (T*B, N)
    mean_out: torch.Tensor  # (T*B, H, a) sampled action
    std_out: torch.Tensor
    log_prob: torch.Tensor  # (T*B,)
    advantage: torch.Tensor  # (T*B,)
    target: torch.Tensor  # (T*B,) value target


def run(cfg: DictConfig) -> None:
    a = phase_config(cfg, "training", cfg.core.agent.planner)
    if not isinstance(a, DictConfig):
        raise TypeError("merged planner configuration must be a mapping")
    if not a.init_value:
        raise ValueError("the dmpo_ppo phase plans against a frozen value: pass training.init_value=<value_td>")
    if int(a.iterations) != 1:
        # each iteration would be its own MDP step; supported, but the reward
        # bookkeeping below assumes one action per decision
        raise ValueError("the on-policy trainer currently supports iterations=1")
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

    reward_metric = load_metric(a.init_value, device=dev)
    reward_metric.eval()
    reward_metric.requires_grad_(False)
    value = cast(ValueFunction, reward_metric)
    critic_dim = int(cast(int, reward_metric.latent_dim))
    if critic_dim % sampler.latent_dim:
        raise ValueError(f"critic latent dim {critic_dim} is not a multiple of {sampler.latent_dim}")
    context = critic_dim // sampler.latent_dim
    if context > a.horizon:
        raise ValueError(f"critic context {context} exceeds the plan horizon {a.horizon}")

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
        learn_search_std=True,  # the on-policy action needs a search width
        mean_search_std=a.mean_search_std,
        std_search_std=a.std_search_std,
        action_lows=action_lows,
        action_highs=action_highs,
    ).to(dev)
    critic = DMPOCritic(
        sampler.latent_dim,
        horizon=a.horizon,
        a_dim=sampler.a_dim,
        hidden=a.critic_hidden,
        learn_std=a.learn_std,
    ).to(dev)
    # reference: [[actor, shift_model], [critic]] with separate learning rates
    actor_parameters = list(net.parameters())
    actor_optimizer = torch.optim.Adam(actor_parameters, lr=a.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=a.critic_lr)

    def plan_costs(
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
            scored: list[torch.Tensor] = []
            for start in range(0, flat.shape[0], size):
                stop = start + size
                trajectory = rollout_traj(wm, zh[start:stop], ah[start:stop], flat[start:stop])
                scored.append(windowed_terminal_value(value, trajectory, zg[start:stop], context))
            return torch.cat(scored).view(batch, samples)

        return cost

    def distance(z_hist: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """Critic cost-to-go at the current window — the reward's potential."""
        window = z_hist[:, -context:] if context > 1 else z_hist[:, -1:]
        return windowed_terminal_value(value, window, z_goal, context)

    @torch.no_grad()
    def collect() -> tuple[Rollout, dict[str, float]]:
        """Roll the MPC-in-the-loop policy forward inside the world model."""
        batch = sampler.sample(a.batch)
        z_hist, a_hist, z_goal = batch.z_hist, batch.a_hist, batch.z_goal
        mean, std = net.initial(a.batch, dev)
        steps: list[dict[str, torch.Tensor]] = []
        cost_to_go = distance(z_hist, z_goal)
        start_distance = cost_to_go.clone()

        for _ in range(int(a.decisions)):
            if net.use_shift and steps:
                mean, std = net.warm_start(mean, std, int(a.receding))
            else:
                mean, std = net.initial(a.batch, dev)
            plans = net.plans(mean, std)
            costs = plan_costs(z_hist, a_hist, z_goal)(plans)
            new_mean, new_std, log_prob, _ = net.sample_step(mean, std, plans, costs)
            state = z_hist[:, -1]
            values = critic(state, z_goal, mean, std)

            executed = new_mean[:, : int(a.receding)]
            trajectory = rollout_traj(wm, z_hist, a_hist, executed)
            z_hist = torch.cat([z_hist, trajectory], dim=1)[:, -3:]
            a_hist = torch.cat([a_hist, executed], dim=1)[:, -2:]
            next_distance = distance(z_hist, z_goal)
            steps.append(
                {
                    "state": state,
                    "goal": z_goal,
                    "mean_in": mean,
                    "std_in": std,
                    "plans": plans,
                    "costs": costs,
                    "mean_out": new_mean,
                    "std_out": new_std,
                    "log_prob": log_prob,
                    "value": values,
                    # progress toward the goal in critic units: the closed-loop
                    # return this optimizer is being credited for
                    "reward": cost_to_go - next_distance,
                }
            )
            cost_to_go = next_distance
            mean, std = new_mean, new_std

        final_value = critic(z_hist[:, -1], z_goal, mean, std)
        advantages: list[torch.Tensor] = []
        running = torch.zeros(a.batch, device=dev)
        following = final_value
        for step in reversed(steps):
            delta = step["reward"] + a.gamma * following - step["value"]
            running = delta + a.gamma * a.gae_lambda * running
            advantages.insert(0, running)
            following = step["value"]
        advantage = torch.cat(advantages)
        target = advantage + torch.cat([step["value"] for step in steps])
        normalized = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
        rollout = Rollout(
            state=torch.cat([step["state"] for step in steps]),
            goal=torch.cat([step["goal"] for step in steps]),
            mean_in=torch.cat([step["mean_in"] for step in steps]),
            std_in=torch.cat([step["std_in"] for step in steps]),
            plans=torch.cat([step["plans"] for step in steps]),
            costs=torch.cat([step["costs"] for step in steps]),
            mean_out=torch.cat([step["mean_out"] for step in steps]),
            std_out=torch.cat([step["std_out"] for step in steps]),
            log_prob=torch.cat([step["log_prob"] for step in steps]),
            advantage=normalized,
            target=target,
        )
        stats = {
            "return": float(torch.cat([step["reward"] for step in steps]).sum().item() / a.batch),
            "progress": float((start_distance - cost_to_go).mean().item()),
            "distance": float(cost_to_go.mean().item()),
        }
        return rollout, stats

    def optimize(rollout: Rollout) -> tuple[float, float]:
        size = rollout.state.shape[0]
        minibatch = min(int(a.minibatch), size)
        policy_loss = value_loss = 0.0
        for _ in range(int(a.ppo_epochs)):
            order = torch.randperm(size, device=dev)
            for start in range(0, size, minibatch):
                index = order[start : start + minibatch]
                step = net.update(
                    rollout.mean_in[index],
                    rollout.std_in[index],
                    rollout.plans[index],
                    rollout.costs[index],
                )
                log_prob = step.log_prob(rollout.mean_out[index], rollout.std_out[index])
                ratio = (log_prob - rollout.log_prob[index]).exp()
                advantage = rollout.advantage[index]
                clipped = ratio.clamp(1.0 - a.clip_epsilon, 1.0 + a.clip_epsilon)
                loss = -torch.min(ratio * advantage, clipped * advantage).mean()
                if a.entropy_penalty:
                    loss = loss - a.entropy_penalty * step.entropy().mean()
                actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]  # PyTorch 2.7 Tensor.backward is untyped.
                torch.nn.utils.clip_grad_norm_(actor_parameters, a.max_grad_norm)
                actor_optimizer.step()

                predicted = critic(
                    rollout.state[index],
                    rollout.goal[index],
                    rollout.mean_in[index],
                    rollout.std_in[index],
                )
                critic_loss = torch.nn.functional.mse_loss(predicted, rollout.target[index])
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(critic.parameters(), a.max_grad_norm)
                critic_optimizer.step()
                policy_loss, value_loss = float(loss.item()), float(critic_loss.item())
        return policy_loss, value_loss

    # saved before training, since snapshots record its path; a copy, as training still uses the original
    value_checkpoint = save_metric(
        copy.deepcopy(reward_metric).cpu(), run_name=a.output.value_checkpoint, cache_dir=a.run.directory
    )
    logger.success(f"Copied the frozen critic to {value_checkpoint}")
    planner_checkpoint = Path(a.run.checkpoints) / a.output.planner_checkpoint

    def payload() -> dict[str, object]:
        return {
            "kind": "dmpo",
            "sd": {key: value.detach().cpu().clone() for key, value in net.state_dict().items()},
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
            "learn_search_std": True,
            "mean_search_std": float(a.mean_search_std),
            "std_search_std": float(a.std_search_std),
            "value": str(a.output.value_checkpoint),
            "value_context": context,
            "temporal_objective": "terminal",
            "objective": "ppo",
        }

    for iteration in range(int(a.ppo_iterations)):
        rollout, stats = collect()
        policy_loss, value_loss = optimize(rollout)
        if iteration % 10 == 0:
            logger.info(
                f"ppo {iteration}: return {stats['return']:.3f} progress {stats['progress']:.3f} "
                f"distance {stats['distance']:.3f} policy {policy_loss:.4f} value {value_loss:.4f}"
            )
        if a.save_every and (iteration + 1) % a.save_every == 0 and iteration + 1 < a.ppo_iterations:
            snapshot = planner_checkpoint.with_name(f"{planner_checkpoint.stem}_step{iteration + 1}.pt")
            torch.save(payload(), snapshot)
            logger.info(f"Saved snapshot at iteration {iteration + 1} to {snapshot}")

    net.eval()
    torch.save(payload(), planner_checkpoint)
    logger.success(f"Saved the DMPO optimizer to {planner_checkpoint}")
