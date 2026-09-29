"""The graphed refinement step reproduces the eager energy and its gradient."""

from typing import Any, cast

import pytest
import torch

from rp1.core.world_model.rollout import rollout_traj
from rp1.methods.rp1.energy import plan_energy

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")

D, H, A_DIM, B = 16, 5, 6, 4


class TinyWM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_encoder = torch.nn.Linear(A_DIM, D)
        self.mix = torch.nn.Linear(2 * D, D)

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        joint = torch.cat([emb.mean(dim=1), act_emb.mean(dim=1)], dim=-1)
        mixed: torch.Tensor = self.mix(joint)
        return mixed.unsqueeze(1).expand(-1, emb.shape[1], -1)


def test_rejects_cpu() -> None:
    from rp1.methods.rp1.graphed import GraphedRefinement

    if torch.cuda.is_available():
        pytest.skip("CPU-rejection check only meaningful without CUDA")
    with pytest.raises(ValueError, match="CUDA"):
        GraphedRefinement(None, cast(Any, None), H, A_DIM, D, "cpu", warmup_iters=5)


@cuda
def test_graphed_step_matches_eager() -> None:
    from rp1.methods.rp1.graphed import GraphedRefinement

    torch.manual_seed(0)
    dev = "cuda"
    wm = TinyWM().to(dev).eval()
    head = torch.nn.Linear(2 * D, 1).to(dev).eval()
    for p in list(wm.parameters()) + list(head.parameters()):
        p.requires_grad_(False)

    def value(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        scored: torch.Tensor = head(torch.cat([state, goal], dim=-1))
        return scored.squeeze(-1)

    zh = torch.randn(B, 3, D, device=dev)
    ah = torch.zeros(B, 2, A_DIM, device=dev)
    zg = torch.randn(B, D, device=dev)
    A = 0.3 * torch.randn(B, H, A_DIM, device=dev)

    ref = GraphedRefinement(wm, value, H, A_DIM, D, dev, warmup_iters=3)
    ref.bind(zh, ah, zg)
    score_g, grad_g = ref.step(A)

    A_in = A.detach().requires_grad_(True)
    traj_e = rollout_traj(wm, zh, ah, A_in)
    score_e = plan_energy(value, traj_e, zg, frames=1)
    (grad_e,) = torch.autograd.grad(score_e.sum(), A_in)

    assert torch.allclose(score_g, score_e.detach(), atol=1e-6)
    assert torch.allclose(grad_g, grad_e, atol=1e-6)
    # rebinding a different context changes results (buffers actually staged)
    ref.bind(zh + 1.0, ah, zg)
    assert not torch.allclose(ref.step(A)[0], score_e.detach(), atol=1e-3)
