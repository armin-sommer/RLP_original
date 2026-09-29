import pytest
import torch


def test_solver_rejects_a_foreign_checkpoint() -> None:
    from rp1.core.agent.solver.base import PlannerCheckpoint
    from rp1.core.agent.solver.dmpo import DMPOSolver

    checkpoint = PlannerCheckpoint(payload={"kind": "rp1", "sd": {}}, value=torch.nn.Identity())
    with pytest.raises(ValueError, match="unsupported checkpoint kind"):
        DMPOSolver(
            model=torch.nn.Linear(2, 2),
            checkpoint=checkpoint,
            iters=None,
            mppi_mode=False,
            cost_chunk=0,
            report_cost=False,
            graphed=False,
        )
