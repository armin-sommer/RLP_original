"""L2O-MPC: gated full-replacement update, expert step, and DAgger wiring.

These tests guard the gating algebra, the expert update, the box clip and the
warm start.
"""

from typing import Any

import torch

from rp1.core.agent.planner.l2o import L2ONet, mppi_update

H, A_DIM, N, B = 3, 4, 16, 5


def _net(**kwargs: Any) -> L2ONet:
    defaults: dict[str, Any] = {
        "horizon": H,
        "a_dim": A_DIM,
        "num_samples": N,
        "hidden": 32,
        "amax": 1.0,
        "dropout": 0.0,
    }
    return L2ONet(**{**defaults, **kwargs})


def test_first_sample_is_the_current_mean() -> None:
    net = _net()
    mean, std = net.initial(B, "cpu")
    mean = mean + 0.3
    plans = net.plans(mean, std)
    assert plans.shape == (B, N, H, A_DIM)
    assert torch.allclose(plans[:, 0], mean)


def test_untrained_update_is_the_paper_plain_half_gate() -> None:
    """Zero head + gate_bias 0: g = 0.5, h = 0, so the mean halves."""
    torch.manual_seed(0)
    net = _net(init_scale=1e-8)
    mean, std = net.initial(B, "cpu")
    mean = mean + 0.6
    updated, updated_std = net(mean, std, torch.randn(B, N))
    assert torch.allclose(updated, 0.5 * mean, atol=1e-4)
    assert torch.equal(updated_std, std)  # covariance fixed by default


def test_negative_gate_bias_recovers_the_identity() -> None:
    """A closed gate passes the (warm-started) mean through untouched."""
    torch.manual_seed(0)
    net = _net(init_scale=1e-8, gate_bias=-20.0)
    mean, std = net.initial(B, "cpu")
    mean = mean + 0.6
    updated, _ = net(mean, std, torch.randn(B, N))
    assert torch.allclose(updated, mean, atol=1e-4)


def test_update_stays_inside_the_plan_box() -> None:
    torch.manual_seed(0)
    net = _net(init_scale=1.0)  # large head so proposals saturate
    mean, std = net.initial(B, "cpu")
    updated, _ = net(mean + 0.9, std, torch.randn(B, N))
    assert updated.abs().max() <= net.amax + 1e-6


def test_expert_update_reduces_a_quadratic_cost() -> None:
    """The DAgger expert is the hand-written MPPI step: it must optimize."""
    torch.manual_seed(0)
    net = _net()
    mean, std = net.initial(B, "cpu")
    mean = mean + 0.7
    plans = net.plans(mean, std)
    costs = plans.pow(2).sum(dim=(2, 3))
    updated = mppi_update(mean, plans, costs)
    assert updated.pow(2).sum() < mean.pow(2).sum()


def test_imitation_loss_reaches_every_parameter() -> None:
    """Supervised regression on the expert mean trains the whole network."""
    torch.manual_seed(0)
    net = _net()
    mean, std = net.initial(B, "cpu")
    predicted, _ = net(mean, std, torch.randn(B, N))
    target = torch.full_like(predicted, 0.2)
    torch.nn.functional.mse_loss(predicted, target).backward()  # type: ignore[no-untyped-call]
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None, name
        assert bool(parameter.grad.abs().sum() > 0), name


def test_plan_runs_the_inner_loop_without_cost_gradients() -> None:
    net = _net()
    mean, std = net.initial(B, "cpu")
    calls: list[int] = []

    def cost_fn(plans: torch.Tensor) -> torch.Tensor:
        calls.append(plans.shape[1])
        return plans.pow(2).sum(dim=(2, 3))

    final, _, history = net.plan(cost_fn, mean, std, 3)
    assert len(history) == 3 and calls == [N, N, N]
    assert final.shape == (B, H, A_DIM)
    assert torch.equal(final, history[-1])


def test_warm_start_is_the_plain_shift() -> None:
    """L2O-MPC has no learned shift: drop executed blocks, zero-pad."""
    net = _net()
    mean = torch.arange(float(B * H * A_DIM)).view(B, H, A_DIM)
    std = torch.full((B, H, A_DIM), 0.5)
    shifted_mean, shifted_std = net.warm_start(mean, std, 1)
    assert torch.equal(shifted_mean[:, :-1], mean[:, 1:])
    assert bool((shifted_mean[:, -1] == 0).all())
    assert torch.equal(shifted_std, std)  # constant covariance shifts to itself
