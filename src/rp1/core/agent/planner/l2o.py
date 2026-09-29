"""L2O-MPC: the learned sampling-MPC update of *Learning to Optimize in MPC*.

Sacks, Boots, ICRA 2022 (arXiv:2212.02603), the predecessor of DMPO
(:mod:`rp1.core.agent.planner.dmpo`).

The paper unifies sampling-based MPC under dynamic mirror descent and then
*learns the whole update rule*: instead of the hand-designed step toward the
softmax-weighted elite (MPPI, Eq. 12), a two-layer MLP reads the current
sampling-distribution parameters together with the ``N`` rollout costs and
emits a GRU-style gated replacement::

    g_mu, h_mu = m_theta([costs, mu, sigma])  # g through a sigmoid
    mu = (1 - g_mu).mu + g_mu.h_mu

This is the structural difference from DMPO, which keeps the hand-written
MPPI reduction and learns a *residual* on it: L2O-MPC's network never sees an
MPPI mean — the sampled costs are its only guidance, and untrained it is not
a working optimizer. The paper's experiments fix a diagonal covariance
(``learn_std=False`` default); the gated covariance update from the paper's
formulation is available behind ``learn_std=True``.

Training is DAgger imitation of an MPPI expert with a larger sample budget
(:mod:`rp1.training.phases.agent.l2o`). :func:`mppi_update` is that expert's
one-step update.

Deltas from the reference:

- Costs are standardized before entering the network (the sibling DMPO
  reference code does the same to its cost features); the paper does not
  specify its cost conditioning.
- Action bounds are the symmetric plan clip ``[-amax, amax]`` shared with the
  rest of this repository.
- ``gate_bias`` initializes the gate head's bias; at 0.0, the paper's plain
  init, an untrained gate passes half the proposal.
"""

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn

from rp1.core.agent.planner.dmpo import gaussian_halton

__all__ = ["L2ONet", "mppi_update"]

_STD_MIN = 1e-6
_STD_MAX = 1e3

# cost_fn: plans (B, N, H, a) -> per-plan costs (B, N)
type CostFn = Callable[[torch.Tensor], torch.Tensor]


def mppi_update(
    mean: torch.Tensor,
    plans: torch.Tensor,
    costs: torch.Tensor,
    temperature: float = 0.05,
    step_size: float = 0.8,
    scale_costs: bool = True,
) -> torch.Tensor:
    """One hand-written DMD-MPC (MPPI) update — the DAgger expert's step.

    The same computation as :meth:`rp1.core.agent.planner.dmpo.DMPONet.mppi_mean`,
    as a free function so the L2O trainer can run it with an arbitrary
    (larger) sample set than any network's fixed width.
    """
    with torch.no_grad():
        scaled = costs
        if scale_costs:
            low = scaled.min(dim=-1, keepdim=True).values
            high = scaled.max(dim=-1, keepdim=True).values
            scaled = (scaled - low) / (high - low + 1e-6)
        weights = torch.softmax(-scaled / temperature, dim=1)
        update = (weights[:, :, None, None] * plans).sum(dim=1)
    return (1.0 - step_size) * mean + step_size * update


def _mlp(in_size: int, out_size: int, hidden: int, dropout: float, init_scale: float) -> nn.Sequential:
    """The reference network: two ReLU hidden layers with dropout, near-zero head."""
    last = nn.Linear(hidden, out_size)
    last.weight.data.normal_(0.0, init_scale)
    last.bias.data.fill_(0.0)
    return nn.Sequential(
        nn.Linear(in_size, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        last,
    )


class L2ONet(nn.Module):
    """L2O-MPC's learned update rule.

    ``forward`` is one optimizer iteration: given the current ``(mean, std)``
    and the sampled plans' costs, return the updated ``(mean, std)``.
    ``plan`` runs the whole inner loop against a caller-supplied cost function
    and is shared by the solver and the trainer, so deployment and training
    execute the same procedure.
    """

    def __init__(
        self,
        horizon: int = 5,
        a_dim: int = 25,
        num_samples: int = 64,
        hidden: int = 1024,
        amax: float = 2.5,
        init_std: float = 1.0,
        dropout: float = 0.1,
        learn_std: bool = False,
        gate_bias: float = 0.0,
        init_scale: float = 1e-3,
        halton: bool = True,
        seed_val: int = 0,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.a_dim = int(a_dim)
        self.num_samples = int(num_samples)
        self.hidden = int(hidden)
        self.amax = float(amax)
        self.init_std = float(init_std)
        self.learn_std = bool(learn_std)
        self.gate_bias = float(gate_bias)
        self.halton = bool(halton)
        self.seed_val = int(seed_val)

        plan_size = self.horizon * self.a_dim
        # inputs: N costs + mean, plus the covariance when it is learned
        in_size = self.num_samples + plan_size + (plan_size if self.learn_std else 0)
        # outputs: (g_mu, h_mu) and, when learned, (g_sigma, h_sigma)
        out_size = 2 * plan_size * (1 + int(self.learn_std))
        self.actor = _mlp(in_size, out_size, self.hidden, dropout, init_scale)
        head = self.actor[-1]
        assert isinstance(head, nn.Linear)
        # bias the gate slices: at gate_bias=0 an untrained gate is 0.5 (the
        # paper's plain init); pushed negative it opens from the identity
        head.bias.data[plan_size : 2 * plan_size] = self.gate_bias
        if self.learn_std:
            head.bias.data[3 * plan_size : 4 * plan_size] = self.gate_bias

        # The fixed sample set is part of the trained artifact: the network
        # reads costs positionally, so a checkpoint is only meaningful together
        # with the samples those costs were produced by (sample 0 = the mean).
        if self.halton:
            base = gaussian_halton(self.num_samples - 1, plan_size, self.seed_val)
        else:
            generator = torch.Generator().manual_seed(self.seed_val)
            base = torch.randn(self.num_samples - 1, plan_size, generator=generator)
        base = base.view(self.num_samples - 1, self.horizon, self.a_dim)
        self.register_buffer("base_samples", torch.cat([torch.zeros_like(base[:1]), base], dim=0))

    # ------------------------------------------------------------- sampling
    def plans(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Reparameterize the fixed samples: ``(B, N, H, a)``, clipped."""
        base = cast(torch.Tensor, self.base_samples).to(dtype=mean.dtype)
        return (mean.unsqueeze(1) + std.unsqueeze(1) * base).clamp(-self.amax, self.amax)

    def initial(
        self, batch: int, device: str | torch.device, dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cold-start parameters: zero mean, ``init_std`` covariance."""
        shape = (batch, self.horizon, self.a_dim)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.full(shape, self.init_std, device=device, dtype=dtype),
        )

    # ------------------------------------------------------------- features
    @staticmethod
    def _standardized_costs(costs: torch.Tensor) -> torch.Tensor:
        costs = costs.detach().reshape(costs.shape[0], -1)
        return (costs - costs.mean(dim=-1, keepdim=True)) / (costs.std(dim=-1, keepdim=True) + 1e-6)

    # --------------------------------------------------------------- update
    def forward(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        costs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, plan_size = mean.shape[0], self.horizon * self.a_dim
        features = [self._standardized_costs(costs), ((mean + self.amax) / (2.0 * self.amax)).reshape(batch, -1)]
        if self.learn_std:
            features.append((std / (self.init_std * 10.0)).reshape(batch, -1))
        out = cast(torch.Tensor, self.actor(torch.cat(features, dim=-1)))

        proposed = torch.tanh(out[:, :plan_size]).view(batch, self.horizon, self.a_dim) * self.amax
        gate = torch.sigmoid(out[:, plan_size : 2 * plan_size]).view(batch, self.horizon, self.a_dim)
        new_mean = ((1.0 - gate) * mean + gate * proposed).clamp(-self.amax, self.amax)

        if self.learn_std:
            proposed_std = out[:, 2 * plan_size : 3 * plan_size].exp().view(batch, self.horizon, self.a_dim)
            gate_std = torch.sigmoid(out[:, 3 * plan_size :]).view(batch, self.horizon, self.a_dim)
            new_std = ((1.0 - gate_std) * std + gate_std * self.init_std * proposed_std).clamp(_STD_MIN, _STD_MAX)
        else:
            new_std = std
        return new_mean, new_std

    # ----------------------------------------------------------- warm start
    def warm_start(self, mean: torch.Tensor, std: torch.Tensor, executed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The standard DMD-MPC shift — L2O-MPC has no learned warm start.

        ``executed`` plan blocks are dropped, the mean is zero-padded and the
        covariance's last entry repeated, exactly the shift the paper feeds
        its network as ``mu_tilde``.
        """
        keep = max(self.horizon - int(executed), 0)
        pad = self.horizon - keep
        shifted_mean = torch.cat([mean[:, executed:], torch.zeros_like(mean[:, :pad])], dim=1)
        shifted_std = torch.cat([std[:, executed:], std[:, -1:].expand(-1, pad, -1)], dim=1)
        return shifted_mean, shifted_std

    # ----------------------------------------------------------- inner loop
    def plan(
        self,
        cost_fn: CostFn,
        mean: torch.Tensor,
        std: torch.Tensor,
        iters: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Run ``iters`` learned iterations; returns the final and per-iteration means.

        Costs are features of the update rule, never a path for gradients.
        One iteration costs ``num_samples`` forward world-model rollouts.
        """
        history: list[torch.Tensor] = []
        for _ in range(iters):
            with torch.no_grad():
                plans = self.plans(mean.detach(), std.detach())
                costs = cost_fn(plans)
            mean, std = self(mean, std, costs)
            history.append(mean)
        return mean, std, history
