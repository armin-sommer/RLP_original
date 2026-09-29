"""DMPO: the learned MPC inner loop of *Deep Model Predictive Optimization*.

Sacks, Rana, Huang, Spitzer, Shi, Boots, ICRA 2024 (arXiv:2310.04590); the
authors' code is ``jisacks/dmpo``.

DMPO keeps MPC's structure — sample ``N`` action sequences, roll them out,
reduce their costs into a new sampling distribution — and *learns* the
reduction. Two MLPs replace the hand-written update:

- **actor** ``m_phi``: takes the current distribution parameters and the
  ``N`` rollout costs, and emits a residual on the MPPI mean update together
  with a gate and a multiplicative covariance update (Eq. 13-14)::

      mu_hat, g, log_sigma = m_phi([costs_z, mu_n, sigma_n])
      mu = (1 - g) * mu_MPPI + g * mu_hat
      sigma = sigma_init * exp(log_sigma)

- **shift model** ``Phi_phi``: learns the warm start, as a residual on the
  standard shift-forward of the previous decision's parameters (Sec. IV-D)::

      mu_tilde = mu_SHIFT + Phi_mu(theta_prev)
      sigma_tilde = sigma_SHIFT * exp(Phi_sigma(theta_prev))

The optimizer never sees the state: its only task-specific signal is the cost
vector, which is what makes the same learned rule reusable across goals.

Deltas from the reference implementation:

- **Pathwise training.** The paper trains ``m_phi`` with PPO, which needs the
  actor to emit search distributions over ``(mu, sigma)``. The world model here
  is differentiable, so :mod:`rp1.training.phases.agent.dmpo` trains the same
  networks by pathwise gradients; the search heads (``learn_search_std``) exist
  only for the PPO trainer, :mod:`rp1.training.phases.agent.dmpo_ppo`. The
  forward path (cost normalization, gating, the MPPI residual, the
  multiplicative covariance update, the shift residual) is the reference
  computation.
- **Gate activation follows the code, not the paper.** The paper describes a
  sigmoid gate in ``[0, 1]``; the reference code, which produced the published
  numbers, uses ``tanh``. ``gate_activation`` selects either.
- **Action bounds** are the environment's per-dimension limits in z-scored
  units (``action_lows``/``action_highs``), as in the reference; without them,
  plans are clipped symmetrically at ``amax``.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

__all__ = ["DMPOCritic", "DMPONet", "DMPOUpdate", "gaussian_halton"]

_STD_MIN = 1e-6
_STD_MAX = 1e3

# cost_fn: plans (B, N, H, a) -> per-plan costs (B, N)
type CostFn = Callable[[torch.Tensor], torch.Tensor]


def _primes(count: int) -> list[int]:
    found: list[int] = []
    candidate = 2
    while len(found) < count:
        if all(candidate % p for p in found):
            found.append(candidate)
        candidate += 1
    return found


def gaussian_halton(
    num_samples: int,
    dim: int,
    seed: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Standard-Gaussian quasi-random samples, ``(num_samples, dim)``.

    A Halton sequence (one prime base per dimension, ``seed`` skipped entries)
    mapped through the inverse Gaussian CDF — DMPO's fixed sample set, drawn
    once and reparameterized by the current distribution at every decision, so
    the actor sees costs of a *consistent* sample pattern instead of fresh
    noise.
    """
    bases = torch.tensor(_primes(dim), dtype=torch.float64)
    indices = torch.arange(seed + 1, seed + 1 + num_samples, dtype=torch.float64).unsqueeze(1)
    uniform = torch.zeros(num_samples, dim, dtype=torch.float64)
    digits = indices.clone()
    factor = bases.clone()
    while bool((digits >= 1).any()):
        uniform += (digits % bases) / factor
        digits = torch.div(digits, bases, rounding_mode="floor")
        factor = factor * bases
    uniform = uniform.clamp(1e-6, 1.0 - 1e-6)
    normal = torch.erfinv(2.0 * uniform - 1.0) * (2.0**0.5)
    return cast(torch.Tensor, normal.to(device=device, dtype=dtype))


_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_HALF_LOG_2PIE = 0.5 * math.log(2.0 * math.pi * math.e)


def _gaussian_log_prob(value: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Diagonal-Gaussian log density, summed over the plan, per batch row."""
    z = (value - loc) / scale
    return (-0.5 * z * z - scale.log() - _LOG_SQRT_2PI).flatten(1).sum(-1)


def _gaussian_entropy(scale: torch.Tensor) -> torch.Tensor:
    return (scale.log() + _HALF_LOG_2PIE).flatten(1).sum(-1)


@dataclass(frozen=True)
class DMPOUpdate:
    """One learned iteration as distributions over the next ``(mean, std)``."""

    mean_loc: torch.Tensor
    mean_scale: torch.Tensor
    std_loc: torch.Tensor
    std_scale: torch.Tensor | None
    mppi: torch.Tensor

    def log_prob(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Joint log-probability of a sampled ``(mean, std)``, per batch row."""
        total = _gaussian_log_prob(mean, self.mean_loc, self.mean_scale)
        if self.std_scale is not None:
            total = total + _gaussian_log_prob(std, self.std_loc, self.std_scale)
        return total

    def entropy(self) -> torch.Tensor:
        total = _gaussian_entropy(self.mean_scale)
        if self.std_scale is not None:
            total = total + _gaussian_entropy(self.std_scale)
        return total


def _mlp(in_size: int, out_size: int, hidden: int, init_scale: float) -> nn.Sequential:
    """The reference MLP: one hidden layer, ReLU, near-zero last layer."""
    last = nn.Linear(hidden, out_size)
    last.weight.data.normal_(0.0, init_scale)
    last.bias.data.fill_(0.0)
    return nn.Sequential(nn.Linear(in_size, hidden), nn.ReLU(), last)


class DMPONet(nn.Module):
    """DMPO's learned update rule and learned warm start.

    ``forward`` is one optimizer iteration: given the current
    ``(mean, std)``, the sampled plans and their costs, return the updated
    ``(mean, std)`` and the MPPI mean the residual was formed on.
    ``plan`` runs the whole inner loop against a caller-supplied cost
    function, and is shared by the solver and the trainer so deployment and
    training optimize the same procedure.
    """

    def __init__(
        self,
        horizon: int = 5,
        a_dim: int = 25,
        num_samples: int = 256,
        hidden: int = 256,
        amax: float = 2.5,
        init_std: float = 1.0,
        temperature: float = 0.05,
        step_size: float = 0.8,
        scale_costs: bool = True,
        gated: bool = True,
        residual: bool = True,
        learn_std: bool = True,
        use_shift: bool = True,
        gate_activation: str = "tanh",
        init_scale: float = 1e-3,
        halton: bool = True,
        seed_val: int = 0,
        learn_search_std: bool = False,
        mean_search_std: float = 0.1,
        std_search_std: float = 0.01,
        action_lows: torch.Tensor | None = None,
        action_highs: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if gate_activation not in {"tanh", "sigmoid"}:
            raise ValueError(f"unsupported gate activation: {gate_activation}")
        self.horizon = int(horizon)
        self.a_dim = int(a_dim)
        self.num_samples = int(num_samples)
        self.hidden = int(hidden)
        self.amax = float(amax)
        self.init_std = float(init_std)
        self.temperature = float(temperature)
        self.step_size = float(step_size)
        self.scale_costs = bool(scale_costs)
        self.gated = bool(gated)
        self.residual = bool(residual)
        self.learn_std = bool(learn_std)
        self.use_shift = bool(use_shift)
        self.gate_activation = gate_activation
        self.halton = bool(halton)
        self.seed_val = int(seed_val)
        # Search distributions exist for the on-policy (PPO) objective: the
        # optimizer's "action" is the updated (mean, covariance), so it must be
        # sampled to get a policy gradient. Off for the pathwise trainer, which
        # differentiates through the world model instead.
        self.learn_search_std = bool(learn_search_std)
        self.mean_search_std = float(mean_search_std)
        self.std_search_std = float(std_search_std)

        plan_size = self.horizon * self.a_dim
        actor_in = self.num_samples + plan_size + (plan_size if self.learn_std else 0)
        # reference head order: mean, gate, mean-search-std, covariance, covariance-search-std
        actor_out = plan_size * (
            1
            + int(self.gated)
            + int(self.learn_search_std)
            + int(self.learn_std)
            + int(self.learn_std and self.learn_search_std)
        )
        self.actor = _mlp(actor_in, actor_out, self.hidden, init_scale)
        self.shift_model = (
            _mlp(
                plan_size + (plan_size if self.learn_std else 0),
                plan_size * (1 + int(self.learn_std)),
                self.hidden,
                init_scale,
            )
            if self.use_shift
            else None
        )

        # The fixed sample set is part of the trained artifact: the actor reads
        # costs positionally, so a checkpoint is only meaningful together with
        # the samples those costs were produced by.
        if self.halton:
            base = gaussian_halton(self.num_samples - 1, plan_size, self.seed_val)
        else:
            generator = torch.Generator().manual_seed(self.seed_val)
            base = torch.randn(self.num_samples - 1, plan_size, generator=generator)
        base = base.view(self.num_samples - 1, self.horizon, self.a_dim)
        # sample 0 is the current mean itself (reference: prepended zeros row)
        self.register_buffer("base_samples", torch.cat([torch.zeros_like(base[:1]), base], dim=0))

        # The reference clips samples, normalizes the mean and scales the residual
        # by the environment's per-dimension limits; `amax` is the symmetric
        # fallback. Unlike the rp1 planner's action limit it is a bound, not a
        # tuned trust region.
        if (action_lows is None) != (action_highs is None):
            raise ValueError("pass both action bounds or neither")
        if action_lows is None:
            low = torch.full((self.a_dim,), -self.amax)
            high = torch.full((self.a_dim,), self.amax)
        else:
            low = torch.as_tensor(action_lows, dtype=torch.float32).reshape(-1)
            high = torch.as_tensor(action_highs, dtype=torch.float32).reshape(-1)
            if low.numel() != self.a_dim or high.numel() != self.a_dim:
                raise ValueError(f"action bounds must have {self.a_dim} entries")
            if bool((high <= low).any()):
                raise ValueError("every action high must exceed its low")
        self.register_buffer("a_low", low)
        self.register_buffer("a_high", high)

    # ------------------------------------------------------------- sampling
    @property
    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return cast(torch.Tensor, self.a_low), cast(torch.Tensor, self.a_high)

    def clip(self, plan: torch.Tensor) -> torch.Tensor:
        """Clamp to the per-dimension action bounds."""
        low, high = self.bounds
        return plan.clamp(low.to(plan.dtype), high.to(plan.dtype))

    def plans(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Reparameterize the fixed samples: ``(B, N, H, a)``, clipped."""
        base = cast(torch.Tensor, self.base_samples).to(dtype=mean.dtype)
        return self.clip(mean.unsqueeze(1) + std.unsqueeze(1) * base)

    def initial(
        self, batch: int, device: str | torch.device, dtype: torch.dtype = torch.float32
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """Cold-start parameters: zero mean, ``init_std`` covariance."""
        shape = (batch, self.horizon, self.a_dim)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.full(shape, self.init_std, device=device, dtype=dtype),
        )

    # ------------------------------------------------------------- features
    def _normalized(
        self, mean: torch.Tensor, std: torch.Tensor | None, std_scale: float | torch.Tensor
    ) -> list[torch.Tensor]:
        batch = mean.shape[0]
        low, high = self.bounds
        feats = [((mean - low) / (high - low)).reshape(batch, -1)]
        if self.learn_std:
            if std is None:
                raise ValueError("learn_std=True requires a covariance input")
            feats.append((std / std_scale).reshape(batch, -1))
        return feats

    @staticmethod
    def _standardized_costs(costs: torch.Tensor) -> torch.Tensor:
        costs = costs.detach().reshape(costs.shape[0], -1)
        return (costs - costs.mean(dim=-1, keepdim=True)) / (costs.std(dim=-1, keepdim=True) + 1e-6)

    # --------------------------------------------------------------- update
    def mppi_mean(self, mean: torch.Tensor, plans: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
        """The hand-written update DMPO learns a residual on (Eq. 5-6)."""
        with torch.no_grad():
            scaled = costs
            if self.scale_costs:
                low = scaled.min(dim=-1, keepdim=True).values
                high = scaled.max(dim=-1, keepdim=True).values
                scaled = (scaled - low) / (high - low + 1e-6)
            weights = torch.softmax(-scaled / self.temperature, dim=1)
            update = (weights[:, :, None, None] * plans).sum(dim=1)
        return (1.0 - self.step_size) * mean + self.step_size * update

    def update(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        plans: torch.Tensor,
        costs: torch.Tensor,
    ) -> DMPOUpdate:
        """One learned iteration, as distribution parameters over ``(mean, std)``.

        The search scales are the on-policy exploration widths; the pathwise
        trainer and deployment use the locations directly (the reference's
        ``use_mean``).
        """
        batch, plan_size = mean.shape[0], self.horizon * self.a_dim
        mppi = self.mppi_mean(mean, plans, costs)
        features = [self._standardized_costs(costs)]
        features += self._normalized(mean, std, self.init_std * 10.0)
        out = cast(torch.Tensor, self.actor(torch.cat(features, dim=-1)))

        def head(index: int) -> torch.Tensor:
            return out[:, plan_size * index : plan_size * (index + 1)].view(batch, self.horizon, self.a_dim)

        low, high = self.bounds
        proposed = torch.tanh(head(0)) * (high - low)
        index = 1
        gate: torch.Tensor | None = None
        if self.gated:
            activation = torch.tanh if self.gate_activation == "tanh" else torch.sigmoid
            gate = activation(head(index))
            index += 1
        anchor = mppi if self.residual else mean
        mean_loc = anchor + proposed if gate is None else (1.0 - gate) * anchor + gate * proposed

        if self.learn_search_std:
            mean_scale = (self.mean_search_std * head(index).exp()).clamp(_STD_MIN, _STD_MAX)
            index += 1
        else:
            mean_scale = torch.full_like(mean_loc, self.mean_search_std)

        if self.learn_std:
            std_loc = (self.init_std * head(index).exp()).clamp(_STD_MIN, _STD_MAX)
            index += 1
            if self.learn_search_std:
                std_scale = (self.std_search_std * head(index).exp()).clamp(_STD_MIN, _STD_MAX)
            else:
                std_scale = torch.full_like(std_loc, self.std_search_std)
        else:
            std_loc, std_scale = std, None
        return DMPOUpdate(mean_loc=mean_loc, mean_scale=mean_scale, std_loc=std_loc, std_scale=std_scale, mppi=mppi)

    def forward(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        plans: torch.Tensor,
        costs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deterministic iteration: the update's locations, clipped."""
        step = self.update(mean, std, plans, costs)
        return self.clip(step.mean_loc), step.std_loc, step.mppi

    # ----------------------------------------------------------- warm start
    def warm_start(self, mean: torch.Tensor, std: torch.Tensor, executed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Learned warm start from the previous decision's parameters.

        ``executed`` is the number of plan blocks consumed since that decision
        (the repository's ``planning.receding_horizon``): the shift-forward
        drops them and zero-pads the mean / repeats the last covariance entry.
        With ``executed >= horizon`` nothing survives the shift and the learned
        residual is the entire warm start.
        """
        keep = max(self.horizon - int(executed), 0)
        pad = self.horizon - keep
        shifted_mean = torch.cat([mean[:, executed:], torch.zeros_like(mean[:, :pad])], dim=1)
        shifted_std = torch.cat([std[:, executed:], std[:, -1:].expand(-1, pad, -1)], dim=1)
        if self.shift_model is None:
            return shifted_mean, shifted_std
        batch, plan_size = mean.shape[0], self.horizon * self.a_dim
        features = self._normalized(mean, std, shifted_std * 10.0)
        out = cast(torch.Tensor, self.shift_model(torch.cat(features, dim=-1)))
        low, high = self.bounds
        residual = torch.tanh(out[:, :plan_size]).view(batch, self.horizon, self.a_dim) * (high - low)
        new_mean = self.clip(shifted_mean + residual)
        if self.learn_std:
            log_std = out[:, plan_size : 2 * plan_size].view(batch, self.horizon, self.a_dim)
            shifted_std = (shifted_std * log_std.exp()).clamp(_STD_MIN, _STD_MAX)
        return new_mean, shifted_std

    # ----------------------------------------------------------- inner loop
    def sample_step(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        plans: torch.Tensor,
        costs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, DMPOUpdate]:
        """One iteration with the update *sampled* — the on-policy action."""
        step = self.update(mean, std, plans, costs)
        if deterministic:
            new_mean, new_std = step.mean_loc, step.std_loc
        else:
            new_mean = step.mean_loc + step.mean_scale * torch.randn_like(step.mean_loc)
            new_std = (
                step.std_loc
                if step.std_scale is None
                else step.std_loc + step.std_scale * torch.randn_like(step.std_loc)
            )
        new_mean = self.clip(new_mean)
        new_std = new_std.clamp(_STD_MIN, _STD_MAX)
        return new_mean, new_std, step.log_prob(new_mean, new_std), step

    def plan(
        self,
        cost_fn: CostFn,
        mean: torch.Tensor,
        std: torch.Tensor,
        iters: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Run ``iters`` learned iterations; returns the final and per-iteration means.

        Costs are features of the update rule, never a path for gradients —
        the same contract the rp1 planner uses for its value-gradient feature.
        One iteration costs ``num_samples`` forward world-model rollouts.
        """
        history: list[torch.Tensor] = []
        for _ in range(iters):
            with torch.no_grad():
                plans = self.plans(mean.detach(), std.detach())
                costs = cost_fn(plans)
            mean, std, _ = self(mean, std, plans, costs)
            history.append(mean)
        return mean, std, history


class DMPOCritic(nn.Module):
    """Value head for the on-policy objective.

    The paper's critic sees the auxiliary MDP state ``(x_t, theta_{t-1})`` —
    the system state plus the previous decision's distribution parameters —
    rather than the optimizer's cost view. Its goal-conditioned analogue here
    is ``(z_t, z_g, mu, sigma)``; it exists only during training and is never
    part of the deployed planner.
    """

    def __init__(self, z_dim: int, horizon: int, a_dim: int, hidden: int = 1024, learn_std: bool = True) -> None:
        super().__init__()
        self.learn_std = bool(learn_std)
        plan_size = horizon * a_dim
        in_size = 2 * z_dim + plan_size * (2 if self.learn_std else 1)
        self.net = _mlp(in_size, 1, hidden, 1e-3)

    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        features = [state, goal, mean.flatten(1)]
        if self.learn_std:
            features.append(std.flatten(1))
        return cast(torch.Tensor, self.net(torch.cat(features, dim=-1))).squeeze(-1)
