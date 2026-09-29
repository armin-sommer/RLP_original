"""Value learners. Each ``fit(cache, cfg, device)`` returns a module exposing
``cost(z_pred, z_goal)``, lower meaning more reachable."""

from rp1.training.phases.agent.learners import contrastive, regression, td

__all__ = ["regression", "td", "contrastive"]
