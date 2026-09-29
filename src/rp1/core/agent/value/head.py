"""Goal-conditioned value heads over pairs of latent states.

Every head exposes ``cost(z_pred, z_goal)``, lower meaning more reachable, so
:class:`rp1.core.agent.value.cost.MetricCost` can plan with any of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from torch import nn


def pair_features(z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
    """Feature map ``[z_i, z_j, z_i - z_j, |z_i - z_j|]``.

    Args:
        z_i: Latent tensor of shape ``(..., D)``.
        z_j: Latent tensor of shape ``(..., D)`` (broadcastable to ``z_i``).

    Returns:
        Tensor of shape ``(..., 4 * D)``.
    """
    diff = z_i - z_j
    return torch.cat([z_i, z_j, diff, diff.abs()], dim=-1)


class PairwiseMetricHead(nn.Module):
    """Two-hidden-layer MLP scalar metric over latent pairs.

    Args:
        latent_dim: Dimension ``D`` of the latent states.
        hidden_dim: Width of each hidden layer.
        depth: Number of hidden layers.
        softplus: Apply Softplus to the output so the metric is non-negative.
        symmetric: Average both orderings, making ``m(z_i, z_j) = m(z_j, z_i)``;
            the paper instead samples pairs in random order.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        depth: int,
        softplus: bool,
        symmetric: bool,
        scale: float,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.scale = float(scale)
        self.symmetric = symmetric
        self.use_softplus = softplus

        layers: list[nn.Module] = [
            nn.Linear(4 * latent_dim, hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Softplus() if softplus else nn.Identity()

    def pretrained_config(self) -> dict[str, object]:
        """Return the complete Hydra constructor config for this metric."""
        return {
            "_target_": f"{type(self).__module__}.{type(self).__name__}",
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "softplus": self.use_softplus,
            "symmetric": self.symmetric,
            "scale": self.scale,
        }

    def _raw(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        feats = pair_features(z_i, z_j)
        return torch.as_tensor(self.out(self.mlp(feats).squeeze(-1)))

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Predicted scalar (e.g. temporal separation) for pairs ``(z_i, z_j)``.

        Shapes ``(..., D), (..., D) -> (...)``.
        """
        out = self._raw(z_i, z_j)
        if self.symmetric:
            out = 0.5 * (out + self._raw(z_j, z_i))
        return out

    def cost(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """Terminal cost (lower == more reachable). For regression this is the
        predicted temporal distance directly."""
        return self.forward(z_pred, z_goal)


class QuasimetricHead(nn.Module):
    """Metric Residual Network (MRN) quasimetric head.

    Reachability (steps-to-go) is a *directed* quasimetric: non-negative, and it
    obeys the triangle inequality, which is what lets a value **stitch** long
    distances from short transitions. MRN parameterises::

        d(z_i -> z_j) = ||u(z_i) - u(z_j)||_2  +  max_k ReLU( v(z_j)_k - v(z_i)_k )

    a symmetric Euclidean term + an asymmetric residual; the sum is a quasimetric
    (Liu et al., 2022). Better inductive bias than a generic MLP for goal-
    conditioned TD distance, especially at long range.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        embed_dim: int,
        depth: int,
        sym_frac: float,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.sym_frac = sym_frac
        layers = [nn.Linear(latent_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, embed_dim)]
        self.enc = nn.Sequential(*layers)
        self.sym_dim = max(1, int(embed_dim * sym_frac))

    def pretrained_config(self) -> dict[str, object]:
        """Return the complete Hydra constructor config for this metric."""
        return {
            "_target_": f"{type(self).__module__}.{type(self).__name__}",
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "sym_frac": self.sym_frac,
        }

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        ei, ej = self.enc(z_i), self.enc(z_j)
        s = self.sym_dim
        d_sym = (ei[..., :s] - ej[..., :s]).pow(2).sum(-1).clamp_min(1e-12).sqrt()
        d_asym = torch.relu(ej[..., s:] - ei[..., s:]).max(dim=-1).values
        return torch.as_tensor(d_sym + d_asym)

    def cost(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        return self.forward(z_pred, z_goal)


class L2WindowCost(nn.Module):
    """Parameter-free L2 cost on concatenated latent frames.

    ``latent_dim`` is the complete per-side width: a window of ``m`` frames of a
    ``D``-dimensional world model declares ``m * D``.
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)

    def pretrained_config(self) -> dict[str, object]:
        return {
            "_target_": f"{type(self).__module__}.{type(self).__name__}",
            "latent_dim": self.latent_dim,
        }

    def cost(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, torch.linalg.vector_norm(z_pred - z_goal, dim=-1))

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        return self.cost(z_i, z_j)


def _interval_union_length(starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    """Differentiable union length of directed intervals on the last axis."""
    order = starts.argsort(dim=-1)
    starts = starts.gather(-1, order)
    ends = ends.gather(-1, order)
    cumulative_end = ends.cummax(dim=-1).values
    previous_end = torch.cat(
        [torch.full_like(cumulative_end[..., :1], float("-inf")), cumulative_end[..., :-1]],
        dim=-1,
    )
    return (ends - torch.maximum(starts, previous_end)).clamp_min(0).sum(-1)


class IQEHead(nn.Module):
    """Interval Quasimetric Embedding head (Wang and Isola, 2022)."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        embed_dim: int,
        depth: int,
        num_components: int,
        alpha_init: float,
    ) -> None:
        super().__init__()
        if embed_dim % num_components:
            raise ValueError("embed_dim must be divisible by num_components")
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_components = num_components
        self.alpha_init = alpha_init
        self.component_dim = embed_dim // num_components
        layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, embed_dim))
        self.encoder = nn.Sequential(*layers)
        alpha = min(max(float(alpha_init), 1e-4), 1 - 1e-4)
        self.alpha_raw = nn.Parameter(torch.logit(torch.tensor(alpha)))
        self.log_scale = nn.Parameter(torch.zeros(()))

    def pretrained_config(self) -> dict[str, object]:
        """Return the complete Hydra constructor config for this metric."""
        return {
            "_target_": f"{type(self).__module__}.{type(self).__name__}",
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_components": self.num_components,
            "alpha_init": self.alpha_init,
        }

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        shape = (self.num_components, self.component_dim)
        encoded_i = self.encoder(z_i).unflatten(-1, shape)
        encoded_j = self.encoder(z_j).unflatten(-1, shape)
        component_cost = _interval_union_length(encoded_i, encoded_j)
        alpha = torch.sigmoid(self.alpha_raw)
        aggregate = alpha * component_cost.amax(-1) + (1 - alpha) * component_cost.mean(-1)
        return torch.as_tensor(self.log_scale.exp() * aggregate)

    def cost(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        return self.forward(z_pred, z_goal)


def _mlp(in_dim: int, hidden: int, out_dim: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.SiLU()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class ContrastiveCritic(nn.Module):
    """Bilinear critic with separate state / goal encoders.

    ``forward`` returns the critic similarity ``f(z_a, z_b)`` (higher == more
    reachable); ``cost`` returns its negation for use as a terminal cost.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        rep_dim: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.rep_dim = rep_dim
        self.depth = depth
        self.phi = _mlp(latent_dim, hidden_dim, rep_dim, depth)  # state encoder
        self.psi = _mlp(latent_dim, hidden_dim, rep_dim, depth)  # goal encoder

    def pretrained_config(self) -> dict[str, object]:
        return {
            "_target_": f"{type(self).__module__}.{type(self).__name__}",
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "rep_dim": self.rep_dim,
            "depth": self.depth,
        }

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor((self.phi(z_a) * self.psi(z_b)).sum(dim=-1))

    def cost(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        return -self.forward(z_pred, z_goal)


def _setting(arch: Mapping[str, object], name: str) -> object:
    if name not in arch:
        raise KeyError(f"metric architecture is missing {name!r}")
    return arch[name]


def _int_setting(arch: Mapping[str, object], name: str) -> int:
    value = _setting(arch, name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"metric architecture {name!r} must be an integer")
    return value


def _float_setting(arch: Mapping[str, object], name: str) -> float:
    value = _setting(arch, name)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"metric architecture {name!r} must be numeric")
    return float(value)


def _bool_setting(arch: Mapping[str, object], name: str) -> bool:
    value = _setting(arch, name)
    if not isinstance(value, bool):
        raise TypeError(f"metric architecture {name!r} must be boolean")
    return value


def build_metric(learner: str, latent_dim: int, arch: Mapping[str, object]) -> nn.Module:
    """Construct an untrained metric module from training architecture settings."""
    if learner == "l2":
        return L2WindowCost(latent_dim)
    if learner in ("regression", "td", "shuffled"):
        if arch.get("head") == "iqe":
            return IQEHead(
                latent_dim,
                hidden_dim=_int_setting(arch, "hidden_dim"),
                embed_dim=_int_setting(arch, "embed_dim"),
                depth=_int_setting(arch, "depth"),
                num_components=_int_setting(arch, "num_components"),
                alpha_init=_float_setting(arch, "alpha_init"),
            )
        if arch.get("head") == "quasimetric":
            return QuasimetricHead(
                latent_dim,
                hidden_dim=_int_setting(arch, "hidden_dim"),
                embed_dim=_int_setting(arch, "embed_dim"),
                depth=_int_setting(arch, "depth"),
                sym_frac=_float_setting(arch, "sym_frac"),
            )
        return PairwiseMetricHead(
            latent_dim,
            hidden_dim=_int_setting(arch, "hidden_dim"),
            depth=_int_setting(arch, "depth"),
            softplus=_bool_setting(arch, "softplus"),
            symmetric=_bool_setting(arch, "symmetric"),
            scale=_float_setting(arch, "scale"),
        )
    if learner == "contrastive":
        return ContrastiveCritic(
            latent_dim,
            hidden_dim=_int_setting(arch, "hidden_dim"),
            rep_dim=_int_setting(arch, "rep_dim"),
            depth=_int_setting(arch, "depth"),
        )
    raise ValueError(f"unknown learner '{learner}'")


__all__ = [
    "ContrastiveCritic",
    "IQEHead",
    "L2WindowCost",
    "PairwiseMetricHead",
    "QuasimetricHead",
    "build_metric",
    "pair_features",
]
