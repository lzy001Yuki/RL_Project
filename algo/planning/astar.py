from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .grid_map import GridMap2D


@dataclass(frozen=True)
class AStarResult:
    path_rc: List[Tuple[int, int]]
    path_xy: List[Tuple[float, float]]
    cost: float


def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    dr = float(a[0] - b[0])
    dc = float(a[1] - b[1])
    return (dr * dr + dc * dc) ** 0.5


def astar(
    grid: GridMap2D,
    start_rc: Tuple[int, int],
    goal_rc: Tuple[int, int],
    cost_map: Optional[np.ndarray] = None,
    allow_diagonal: bool = True,
) -> Optional[AStarResult]:
    """A* on a boolean occupancy grid.

    Args:
        grid: GridMap2D with inflated obstacles.
        start_rc: (row, col)
        goal_rc: (row, col)
        cost_map: optional per-cell multiplicative cost (>=1 for free cells).
        allow_diagonal: use 8-neighborhood if True else 4-neighborhood.
    """
    if not grid.in_bounds(*start_rc) or not grid.in_bounds(*goal_rc):
        return None
    if not grid.is_free(*start_rc) or not grid.is_free(*goal_rc):
        return None

    if cost_map is not None:
        if cost_map.shape != grid.occupancy.shape:
            raise ValueError("cost_map must match grid shape")
        if np.any(cost_map < 1.0):
            raise ValueError("cost_map should be >= 1.0 for numerical stability")

    # Neighbor moves.
    if allow_diagonal:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def step_cost(dr: int, dc: int) -> float:
        return (dr * dr + dc * dc) ** 0.5

    open_heap: List[Tuple[float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, start_rc))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start_rc: 0.0}

    goal = goal_rc

    while open_heap:
        _f, current = heapq.heappop(open_heap)
        if current == goal:
            # Reconstruct
            path_rc: List[Tuple[int, int]] = [current]
            while current in came_from:
                current = came_from[current]
                path_rc.append(current)
            path_rc.reverse()

            path_xy = [grid.grid_to_world(r, c) for r, c in path_rc]
            return AStarResult(path_rc=path_rc, path_xy=path_xy, cost=g_score[goal])

        current_g = g_score[current]
        for dr, dc in moves:
            nr = current[0] + dr
            nc = current[1] + dc
            if not grid.is_free(nr, nc):
                continue

            base = step_cost(dr, dc) * grid.resolution
            if cost_map is None:
                tentative = current_g + base
            else:
                tentative = current_g + base * float(cost_map[nr, nc])

            neighbor = (nr, nc)
            prev = g_score.get(neighbor, float("inf"))
            if tentative < prev:
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                f_score = tentative + _heuristic(neighbor, goal) * grid.resolution
                heapq.heappush(open_heap, (f_score, neighbor))

    return None

