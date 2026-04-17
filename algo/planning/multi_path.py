from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .astar import AStarResult, astar
from .grid_map import GridMap2D
from .repulsion import add_path_repulsion


@dataclass(frozen=True)
class MultiPathConfig:
    n_paths: int = 50
    repulsion_strength: float = 2.0
    repulsion_radius_cells: int = 2
    repulsion_weight: float = 2.0
    detour_ratio_max: float = 1.8
    allow_diagonal: bool = True


def plan_diverse_paths(
    grid: GridMap2D,
    start_rc: Tuple[int, int],
    goal_rc: Tuple[int, int],
    config: MultiPathConfig = MultiPathConfig(),
) -> List[AStarResult]:
    """Iteratively plan multiple routes by repulsing previously used corridors."""
    shortest = astar(grid, start_rc, goal_rc, cost_map=None, allow_diagonal=config.allow_diagonal)
    if shortest is None:
        return []

    paths: List[AStarResult] = [shortest]
    penalty = np.zeros_like(grid.occupancy, dtype=np.float32)

    add_path_repulsion(
        penalty,
        shortest.path_rc,
        strength=config.repulsion_strength,
        radius_cells=config.repulsion_radius_cells,
    )

    shortest_cost = float(shortest.cost)
    for _ in range(config.n_paths - 1):
        cost_map = 1.0 + config.repulsion_weight * penalty
        candidate = astar(grid, start_rc, goal_rc, cost_map=cost_map, allow_diagonal=config.allow_diagonal)
        if candidate is None:
            break
        if candidate.cost > config.detour_ratio_max * shortest_cost:
            # Too long to be realistic; stop early.
            break
        paths.append(candidate)
        add_path_repulsion(
            penalty,
            candidate.path_rc,
            strength=config.repulsion_strength,
            radius_cells=config.repulsion_radius_cells,
        )

    return paths

