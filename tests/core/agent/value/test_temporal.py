import torch

from rp1.core.agent.value.temporal import trajectory_value, window_pair, windowed_terminal_value


def test_tel_exact_preserves_terminal_gradient_and_adds_start_baseline() -> None:
    def value(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return ((state - goal) ** 2).sum(dim=-1)

    start = torch.tensor([[2.0]], requires_grad=True)
    trajectory = torch.tensor([[[1.5], [1.0], [0.5]]], requires_grad=True)
    goal = torch.zeros(1, 1)
    terminal = trajectory_value(value, trajectory, goal, start, "terminal")
    telescoping = trajectory_value(value, trajectory, goal, start, "tel-exact")
    terminal_gradient = torch.autograd.grad(terminal.sum(), trajectory, retain_graph=True)[0]
    telescoping_gradient = torch.autograd.grad(telescoping.sum(), trajectory)[0]
    assert torch.equal(terminal_gradient, telescoping_gradient)
    assert torch.equal(telescoping, terminal - value(start, goal))


def test_windowed_terminal_value_matches_the_window_pair_layout() -> None:
    def value(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return (state * torch.arange(1.0, state.shape[-1] + 1)).sum(dim=-1) - goal.sum(dim=-1)

    trajectory = torch.randn(2, 4, 3)
    goal = torch.randn(2, 3)
    for context in (1, 2, 3, 6):
        expected = value(*window_pair(trajectory, goal, context))
        assert torch.equal(windowed_terminal_value(value, trajectory, goal, context), expected)
    two = windowed_terminal_value(value, trajectory, goal, 2)
    assert torch.equal(two, value(torch.cat([trajectory[:, -2], trajectory[:, -1]], -1), goal.repeat(1, 2)))
