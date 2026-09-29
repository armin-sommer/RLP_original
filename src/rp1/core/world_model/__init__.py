"""The world-model contract the agent plans through.

Dynamics come from Stable-WM's LeWM and PLDM classes, selected by config.
"""

from rp1.core.world_model.base import LatentWorldModel
from rp1.core.world_model.rollout import rollout_terminal, rollout_traj

__all__ = ["LatentWorldModel", "rollout_terminal", "rollout_traj"]
