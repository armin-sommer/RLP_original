"""Train the rp1 planner (the actor) and its value (the critic) together.

The planner refines a plan by following the value's gradient through the frozen
world model, so the value it is trained against shapes what it learns. The two
train jointly, in the style of DDPG/TD3, with the planner as a K-step learned
optimizer:

  critic   V_phi(z, z_goal): n-step expectile TD on the dense latent cache (one
           row per primitive step). Only the TD loss reaches phi.
  teacher  an exponential moving average of the critic (``ema_tau``). It gives
           the TD bootstrap targets and the planner's energy, and it is the value
           saved for deployment, so the planner trains on the value it deploys with.
  actor    the planner network: K refinement steps from the zero plan, trained to
           minimise the teacher's value of the imagined end state.

Schedule: ``pretrain`` critic-only steps (-1 picks 2000 for a fresh critic and 0
for one warm-started from ``init_value``), then one critic step per actor step.
After ``freeze_critic_frac`` of the actor steps the critic and teacher freeze, so
the planner settles on a stationary energy.

``expand_weight > 0`` adds value expansion: every imagined plan becomes a TD
backup ``V(z_0, z_goal) <- cost(H blocks) + discount * V_teacher(z_H, z_goal)``
on the states the planner actually visits. With a low expectile this is close to
a one-sided bound: good plans tighten the value, bad plans barely raise it.

The planner checkpoint names the teacher saved next to it, so the solver loads the
value the planner was trained against.
"""

import copy
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py

with suppress(ImportError):
    import hdf5plugin  # noqa: F401  (registers HDF5 compression filters, e.g. cube h5)
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from rp1.core.agent.value import build_metric
from rp1.core.agent.value.temporal import ValueFunction, window_pair
from rp1.core.world_model.base import LatentWorldModel
from rp1.core.world_model.rollout import rollout_traj
from rp1.data import LatentCache
from rp1.data.base import load_action_stats
from rp1.methods.rp1.energy import plan_energy
from rp1.methods.rp1.net import PlannerNet
from rp1.training.harness.checkpointing import load_metric, load_pretrained, save_metric
from rp1.training.harness.schedule import cosine_interpolate
from rp1.training.phases.agent.learners.td import expectile_loss, n_step_target
from rp1.training.phases.agent.samplers import NStepGoalSampler
from rp1.utils.config import phase_config
from rp1.utils.device import pick_device
from rp1.utils.logging import logger

# (start state, end state, goal) of the actor's imagined plans, the value-expansion backups.
# With a window value all three are already stacked into the value's input windows.
type Expansion = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class ActionBlocks:
    """Normalized action blocks of ``frameskip`` primitive steps, read from an action h5.

    Episodes are those of the block cache; with ``phases > 1`` (a phase-multiplexed
    cache) episode ``e * phases + k`` is source episode ``e`` starting at step ``k``.
    """

    def __init__(self, path: str, frameskip: int, stats: str | None, phases: int) -> None:
        with h5py.File(path, "r") as file:
            self.actions = file["action"][:]
            self.offsets = file["ep_offset"][:]
            self.lengths = file["ep_len"][:] if "ep_len" in file else None
        # nan-aware: some datasets pad episode-terminal steps with NaN actions;
        # those rows are never sampled but must not poison the statistics
        mean, std = np.nanmean(self.actions, 0), np.nanstd(self.actions, 0) + 1e-6
        if stats is not None:
            mean, std = load_action_stats(stats)
            if mean.shape != (self.actions.shape[-1],):
                raise ValueError(f"action statistics have shape {mean.shape}; actions have {self.actions.shape[-1]}")
            logger.info(f"Actions normalized with statistics from {stats}")
        self.mean, self.std = mean, std
        self.normalized = ((self.actions - mean) / std).astype(np.float32)
        self.frameskip = frameskip
        self.phases = phases
        self.dim = self.actions.shape[-1] * frameskip

    def row(self, episode: int, block: int) -> int:
        """The h5 row of the block's first primitive step."""
        source, phase = divmod(episode, self.phases)
        offset = phase + self.frameskip * block
        if self.lengths is not None:
            offset = min(offset, max(0, int(self.lengths[source]) - self.frameskip))
        return int(self.offsets[source] + offset)

    def __call__(self, episode: int, block: int) -> np.ndarray:
        start = self.row(episode, block)
        return np.asarray(self.normalized[start : start + self.frameskip]).reshape(-1)


@dataclass(frozen=True)
class Problems:
    """A batch of planning problems: latent histories ``(B, 3, D)``, their action blocks ``(B, 2, a)``, goals."""

    histories: torch.Tensor
    actions: torch.Tensor
    goals: torch.Tensor


class PlanningTasks:
    """Samples planning problems from the block cache.

    A problem is three consecutive latents, the two action blocks between them, and a
    goal latent. With probability ``p_cross`` the goal comes from a random other
    episode; otherwise it lies ``1..max_delta`` blocks ahead in the same episode. With
    ``band_mix`` a band is drawn uniformly first and the offset uniformly within it,
    so each deployment horizon gets equal mass instead of the long ones dominating.
    """

    def __init__(
        self,
        cache: LatentCache,
        blocks: ActionBlocks,
        *,
        max_delta: int,
        band_mix: list[int] | None,
        p_cross: float,
        rng: np.random.Generator,
        device: str,
    ) -> None:
        if band_mix and max(band_mix) > max_delta:
            raise ValueError(f"band_mix {band_mix} exceeds max_delta {max_delta}")
        self.z = cache.z.to(device).float()
        episodes = cache.episodes()
        keys = [key for key in episodes if len(episodes[key]) > max_delta + 4]
        if not keys:
            longest = max((len(rows) for rows in episodes.values()), default=0)
            raise ValueError(
                f"no episode is longer than max_delta+4 = {max_delta + 4} blocks "
                f"(longest is {longest}); lower planner.max_delta or use a cache with longer episodes"
            )
        self.rows = {key: np.asarray(episodes[key]) for key in keys}
        self.episodes = np.array(keys)
        self.blocks = blocks
        self.max_delta = max_delta
        self.band_mix = band_mix
        self.p_cross = p_cross
        self.rng = rng
        self.device = device

    def _offset(self, remaining: int) -> int:
        """How many blocks ahead the goal lies."""
        if not self.band_mix:
            return int(self.rng.integers(1, self.max_delta + 1))
        high = min(self.max_delta, remaining)
        if high < 1:
            return 1
        band = self.band_mix[int(self.rng.integers(len(self.band_mix)))]
        return int(self.rng.integers(1, min(band, high) + 1))

    def sample(self, batch: int) -> Problems:
        z, rng = self.z, self.rng
        histories: list[torch.Tensor] = []
        actions: list[np.ndarray] = []
        goals: list[torch.Tensor] = []
        for _ in range(batch):
            episode = int(self.episodes[rng.integers(len(self.episodes))])
            rows = self.rows[episode]
            length = len(rows)
            t = int(rng.integers(2, length - 2))
            histories.append(torch.stack([z[rows[t - 2]], z[rows[t - 1]], z[rows[t]]]))
            actions.append(np.stack([self.blocks(episode, t - 2), self.blocks(episode, t - 1)]))
            if rng.random() < self.p_cross:
                other = self.rows[int(self.episodes[rng.integers(len(self.episodes))])]
                goals.append(z[other[rng.integers(len(other))]])
            else:
                delta = self._offset(length - 1 - t)
                goals.append(z[rows[min(t + delta, length - 1)]])
        return Problems(torch.stack(histories), torch.from_numpy(np.stack(actions)).to(self.device), torch.stack(goals))


class Critic:
    """The TD critic, its EMA teacher, and the optimizer that trains the critic."""

    def __init__(
        self, args: DictConfig, module: nn.Module, cache: LatentCache, frames: int, horizon: int, device: str
    ) -> None:
        module.train()
        self.module = module
        self.teacher = copy.deepcopy(module).to(device)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)  # the actor loss flows through the teacher, never into it
        self.teacher.eval()
        self.cache = cache
        self.frames = frames
        self.args = args
        self.device = device
        # the return of one imagined plan: H action blocks of frameskip unit-cost steps
        steps = horizon * args.frameskip
        if args.gamma >= 1.0:
            self.plan_cost, self.plan_discount = float(steps), 1.0
        else:
            self.plan_discount = args.gamma**steps
            self.plan_cost = (1.0 - self.plan_discount) / (1.0 - args.gamma)
        self.sampler = NStepGoalSampler(
            cache,
            n_step=args.n_step,
            p_cross=args.td_p_cross,
            n_buckets=args.td_n_buckets,
            balanced=True,
            seed=args.seed,
            max_delta=args.td_max_delta,
            near_frac=args.near_frac,
            near_max=args.near_max,
        )
        # a parameter-free value (the latent L2 distance) can only ever be frozen
        trainable = any(True for _ in module.parameters())
        if not trainable and args.freeze_critic_frac > 0:
            raise ValueError(f"{type(module).__name__} has no parameters to co-train; set freeze_critic_frac=0")
        self.optimizer = (
            torch.optim.AdamW(module.parameters(), lr=args.critic_lr, weight_decay=args.critic_wd)
            if trainable
            else None
        )

    @property
    def value(self) -> ValueFunction:
        return cast(ValueFunction, self.module)

    @property
    def target(self) -> ValueFunction:
        return cast(ValueFunction, self.teacher)

    def windows(self, indices: torch.Tensor) -> torch.Tensor:
        """Dense-cache rows stacked into windows of ``frames`` latents one action block apart.

        The same construction as ``LatentCache.windowed``: oldest first, clamped
        to the episode's first row at episode starts.
        """
        step = self.cache.step_idx[indices]
        columns = []
        for k in range(self.frames - 1, -1, -1):
            offset = k * self.args.frameskip
            rows = torch.where(step < offset, indices - step, indices - offset)
            columns.append(self.cache.z[rows])
        return torch.cat(columns, dim=-1)

    def step(self, expansion: Expansion | None, tau: float, lr: float) -> float:
        """One TD step (plus value expansion), then the teacher's EMA update; the loss."""
        if self.optimizer is None:
            raise RuntimeError("a critic step on a critic without parameters")
        args, device = self.args, self.device
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        batch = self.sampler.sample(args.td_batch)
        if self.frames > 1:
            # every query is a window rebuilt from the dense cache; the goal side
            # is the sampled goal frame's own window
            z_t, z_tn, z_g = (self.windows(batch[key]).to(device) for key in ("t_idx", "tn_idx", "g_idx"))
        else:
            z_t, z_tn, z_g = batch["z_t"].to(device), batch["z_tn"].to(device), batch["z_g"].to(device)
        returns = {key: batch[key].to(device) for key in ("n_eff", "reached", "dist")}
        with torch.no_grad():
            target = n_step_target(returns, self.target(z_tn, z_g), args.gamma)
        loss = expectile_loss(self.value(z_t, z_g) - target, tau, args.huber_beta)
        if expansion is not None:
            start, end, goal = expansion
            with torch.no_grad():
                expanded = self.plan_cost + self.plan_discount * self.target(end, goal)
            loss = loss + args.expand_weight * expectile_loss(self.value(start, goal) - expanded, tau, args.huber_beta)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch 2.7 Tensor.backward lacks a typed signature here.
        self.optimizer.step()
        with torch.no_grad():
            for teacher_parameter, parameter in zip(self.teacher.parameters(), self.module.parameters(), strict=True):
                teacher_parameter.mul_(1.0 - args.ema_tau).add_(args.ema_tau * parameter)
        return float(loss.item())


def build_critic(args: DictConfig, value: DictConfig, cache: LatentCache, device: str) -> tuple[nn.Module, int]:
    """The initial critic and the number of latents its input window stacks.

    A warm-started value (``init_value``) may take a window of ``m`` latents (width
    ``m * D``); a fresh one is built from the ``value`` architecture.
    """
    if args.init_value:
        critic = load_metric(args.init_value, device=device)
        width = int(cast(int, critic.latent_dim))
        if width % cache.latent_dim:
            raise ValueError(f"init-value width {width} is not a multiple of the cache latent dim {cache.latent_dim}")
        return critic, width // cache.latent_dim
    architecture = {
        "head": value.head,
        "hidden_dim": value.hidden_dim,
        "depth": value.depth,
        "embed_dim": value.embedding_dim,
        "softplus": True,
        "symmetric": False,
        "sym_frac": value.sym_frac,
        "num_components": value.num_components,
        "alpha_init": value.alpha_init,
        "scale": value.scale,
    }
    return build_metric("td", cache.latent_dim, architecture).to(device), 1


def check_window_lag(args: DictConfig, frames: int) -> None:
    """A window value's frames must be one action block apart, as consecutive imagined latents are."""
    if frames == 1:
        return
    lag = args.frameskip if args.window_lag is None else int(args.window_lag)
    if lag != args.frameskip:
        raise ValueError(
            f"window lag {lag} != action block {args.frameskip}: consecutive imagined latents are one action "
            "block apart, so the deployed window would not match the trained one"
        )
    logger.info(f"Training against a {frames}-frame window value (lag {lag})")


class Actor:
    """The planner network, its optimizer, and the replay of its own imagined end states."""

    def __init__(self, args: DictConfig, planner: DictConfig, action_dim: int, device: str) -> None:
        self.net = PlannerNet(
            horizon=planner.horizon,
            action_dim=action_dim,
            hidden_dim=planner.hidden_dim,
            action_limit=planner.action_limit,
        ).to(device)
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=args.actor_lr, weight_decay=args.actor_weight_decay
        )
        self.args = args
        self.iterations = planner.iterations
        self.action_dim = action_dim
        self.device = device
        # the previous batch's imagined last three latents and goals
        self.replay: tuple[torch.Tensor, torch.Tensor] | None = None

    def _replayed(self, problems: Problems) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Histories, actions and goals, a ``replay_prob`` share restarted where the last batch's plans ended.

        Replanning from an imagined end state is what the deployed planner does after
        each executed plan, and it starts from a zero action history there.
        """
        histories, actions, goals = problems.histories, problems.actions, problems.goals
        if self.args.replay_prob <= 0 or self.replay is None:
            return histories, actions, goals
        picked = torch.nonzero(torch.rand(self.args.batch, device=self.device) < self.args.replay_prob).squeeze(1)
        if not picked.numel():
            return histories, actions, goals
        replay_histories, replay_goals = self.replay
        take = torch.randint(0, replay_histories.shape[0], (picked.numel(),), device=self.device)
        histories, actions, goals = histories.clone(), actions.clone(), goals.clone()
        actions[picked] = 0.0
        histories[picked] = replay_histories[take]
        goals[picked] = replay_goals[take]
        return histories, actions, goals

    def step(
        self, tasks: PlanningTasks, wm: LatentWorldModel, teacher: ValueFunction, frames: int
    ) -> tuple[float, float, Expansion]:
        """One actor update; the mean energy after the first and the last refinement, and the expansion batch."""
        args = self.args
        histories, actions, goals = self._replayed(tasks.sample(args.batch))

        def energy_of(plan: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            trajectory = rollout_traj(wm, histories, actions, plan)
            return plan_energy(teacher, trajectory, goals, frames), trajectory

        # Refinement from the zero plan. Each step's gradient input comes from the previous
        # step's rollout, which stays in the graph; the gradient itself is detached, so it
        # is an input of the learned rule and not a path of the training signal.
        plan = torch.zeros(args.batch, self.net.horizon, self.action_dim, device=self.device)
        scored = plan.detach().requires_grad_(True)
        energy, trajectory = energy_of(scored)
        energies: list[torch.Tensor] = []
        for k in range(self.iterations):
            (gradient,) = torch.autograd.grad(energy.sum(), scored, retain_graph=k > 0)
            plan = self.net(plan, gradient.detach(), energy.detach())
            energy, trajectory = energy_of(plan)
            energies.append(energy.mean())
            scored = plan
        if not energies:
            raise RuntimeError("the planner ran no refinement iteration")

        loss = energies[-1] + args.mean_weight * torch.stack(energies).mean()
        if args.ac_weight > 0:
            # anti-constancy: the batch-level constancy of the net plan displacement,
            # ||E_b[sum_t A]||^2 / E_b||sum_t A||^2 in [0, 1]. A planner that exploits
            # the world model emits a near-constant plan whatever the task.
            displacement = plan.sum(1)
            constancy = displacement.mean(0).pow(2).sum() / (displacement.pow(2).sum(1).mean() + 1e-8)
            loss = loss + args.ac_weight * constancy
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.optimizer.step()

        if args.replay_prob > 0:
            self.replay = (trajectory[:, -3:].detach(), goals.detach())
        if frames > 1:
            # expansion in the window value's input space: the start window from the
            # real history, the end window from the rollout, the goal tiled to match
            start, goal = window_pair(histories, goals, frames)
            end, _ = window_pair(trajectory, goals, frames)
            expansion = (start.detach(), end.detach(), goal.detach())
        else:
            expansion = (histories[:, -1].detach(), trajectory[:, -1].detach(), goals.detach())
        return float(energies[0].item()), float(energies[-1].item()), expansion


def planner_payload(
    args: DictConfig,
    planner: DictConfig,
    state_dict: dict[str, torch.Tensor],
    action_dim: int,
    value: Path,
    frames: int,
    step: int,
) -> dict[str, object]:
    """The deployable planner checkpoint: the network, its shape, and the value it plans against.

    ``seed``, ``step`` and ``teacher`` (the offline value it was trained against) place
    the checkpoint in the teacher x planner-step grid that checkpoint selection scores.
    """
    return {
        "seed": int(args.seed),
        "step": step,
        "teacher": None if args.init_value is None else str(args.init_value),
        "state_dict": state_dict,
        "horizon": planner.horizon,
        "iterations": planner.iterations,
        "action_dim": action_dim,
        "action_limit": planner.action_limit,
        "hidden_dim": planner.hidden_dim,
        "value": str(value),
        "window_frames": frames,
    }


def expectile_at(args: DictConfig, step: int, freeze_at: int) -> float:
    """The TD expectile, annealed linearly to ``expectile_final`` over the critic's live phase."""
    if args.expectile_final is None:
        return float(args.expectile)
    return float(args.expectile + (args.expectile_final - args.expectile) * min(step, freeze_at) / max(freeze_at, 1))


def run(cfg: DictConfig) -> None:
    args = phase_config(cfg, "training")
    planner = cfg.core.agent.planner
    device = pick_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    wm_module = load_pretrained(args.wm).to(device).eval()
    wm_module.requires_grad_(False)
    wm = cast(LatentWorldModel, wm_module)

    cache = LatentCache.load(args.cache, mmap=bool(args.cache_mmap)).first_episodes(args.max_episodes)
    if cache.phase_multiplex > 1:
        logger.info(f"Planner cache is phase-multiplexed x{cache.phase_multiplex}")
    blocks = ActionBlocks(args.h5, args.frameskip, args.action_stats, cache.phase_multiplex)
    band_mix = None if args.band_mix is None else [int(band) for band in args.band_mix]
    tasks = PlanningTasks(
        cache, blocks, max_delta=args.max_delta, band_mix=band_mix, p_cross=args.p_cross, rng=rng, device=device
    )
    dense = LatentCache.load(args.cache_td, mmap=bool(args.cache_mmap)).first_episodes(args.max_episodes)
    if args.near_frac > 0:
        logger.info(f"Critic near-goal oversampling: frac={args.near_frac} max={args.near_max} steps")
    module, frames = build_critic(args, cfg.core.agent.value, dense, device)
    check_window_lag(args, frames)
    critic = Critic(args, module, dense, frames, planner.horizon, device)
    actor = Actor(args, planner, blocks.dim, device)

    planner_checkpoint = Path(args.run.checkpoints) / args.output.planner_checkpoint
    # the teacher is saved next to the planner, so checkpoints refer to it by name
    value_checkpoint = Path(str(args.output.value_checkpoint))

    pretrain = args.pretrain if args.pretrain >= 0 else (0 if args.init_value else 2000)
    for i in range(pretrain):
        loss = critic.step(None, args.expectile, args.critic_lr)
        if i % 500 == 0:
            logger.info(f"Critic pretraining {i}/{pretrain}: td_loss={loss:.4f}")

    freeze_at = int(args.freeze_critic_frac * args.steps)
    expansion: Expansion | None = None
    for step in range(args.steps):
        if step == freeze_at:
            logger.info(f"Step {step}: critic and teacher frozen for the remaining {args.steps - freeze_at} steps")
        # the teacher converges (lr decay) and sharpens (expectile anneal) over the
        # critic's live phase; the actor lr decays over all steps
        tau = expectile_at(args, step, freeze_at)
        critic_lr = cosine_interpolate(args.critic_lr, args.critic_lr_final, step, freeze_at)
        actor_lr = cosine_interpolate(args.actor_lr, args.actor_lr_final, step, args.steps)
        for group in actor.optimizer.param_groups:
            group["lr"] = actor_lr
        td_loss = float("nan")
        if step < freeze_at:
            td_loss = critic.step(expansion if args.expand_weight > 0 else None, tau, critic_lr)
        first, final, expansion = actor.step(tasks, wm, critic.target, frames)
        if args.ckpt_every and (step + 1) % int(args.ckpt_every) == 0 and (step + 1) < args.steps:
            snapshot = planner_checkpoint.with_name(f"{planner_checkpoint.stem}_step{step + 1}.pt")
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            state = {key: value.detach().cpu().clone() for key, value in actor.net.state_dict().items()}
            torch.save(planner_payload(args, planner, state, blocks.dim, value_checkpoint, frames, step + 1), snapshot)
            logger.info(f"Saved planner snapshot at step {step + 1} to {snapshot}")
        if step % 500 == 0:
            logger.info(
                f"step {step}: E_final {final:.3f} E_first {first:.3f} "
                f"td_loss {td_loss:.4f} tau {tau:.3f} clr {critic_lr:.2e} alr {actor_lr:.2e}"
            )

    # the teacher first: the planner checkpoint references it
    saved_value = save_metric(critic.teacher.cpu(), run_name=args.output.value_checkpoint, cache_dir=args.run.directory)
    logger.success(f"Saved teacher value to {saved_value}")
    actor.net.eval()
    torch.save(
        planner_payload(
            args, planner, actor.net.cpu().state_dict(), blocks.dim, value_checkpoint, frames, int(args.steps)
        ),
        planner_checkpoint,
    )
    logger.success(f"Saved learned planner to {planner_checkpoint}")
