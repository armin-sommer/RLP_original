import torch
from torch import nn

from rp1.core.agent.value.cost import MetricCost


class _World(nn.Module):
    def get_cost(self, info: dict, actions: torch.Tensor) -> torch.Tensor:
        raise AssertionError("replacement mode scores with the value alone")

    def criterion(self, info: dict) -> torch.Tensor:
        raise AssertionError("replacement mode scores with the value alone")


class _Recorder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def cost(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        self.seen.append((state, goal))
        return state.sum(dim=-1)


def test_two_frame_window_concatenates_frames_and_tiles_the_goal() -> None:
    frames = torch.arange(1.0, 5.0)[:, None].expand(4, 3)
    predicted = frames.reshape(1, 1, 4, 3)
    goal = torch.full((1, 1, 1, 3), 9.0)
    metric = _Recorder(latent_dim=6)
    MetricCost(_World(), metric, "replacement", 0.0, None, "terminal").criterion(
        {"predicted_emb": predicted, "goal_emb": goal}
    )
    state, tiled = metric.seen[0]
    assert torch.equal(state, torch.tensor([[[3.0, 3.0, 3.0, 4.0, 4.0, 4.0]]]))
    assert torch.equal(tiled, torch.full((1, 1, 6), 9.0))
