"""Explicit planning utilities (grid, A*, etc.).

This subpackage is intentionally torch-free, so you can run planning-only code
in lightweight environments. RL algorithms under `algo/ppo` and `algo/sac`
require PyTorch.
"""

from .astar import AStarResult, astar
from .grid_map import GridMap2D, GridSpec
from .landmarks import extract_landmarks_from_path, resample_path_by_distance
from .multi_path import MultiPathConfig, plan_diverse_paths
from .repulsion import add_path_repulsion

__all__ = [
    "AStarResult",
    "GridMap2D",
    "GridSpec",
    "MultiPathConfig",
    "add_path_repulsion",
    "astar",
    "extract_landmarks_from_path",
    "plan_diverse_paths",
    "resample_path_by_distance",
]
