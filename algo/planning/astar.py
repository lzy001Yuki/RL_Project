from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .grid_map import GridMap2D


@dataclass(frozen=True)
class AStarResult:
    path_rc: List[Tuple[int, int]]
    path_xy: List[Tuple[float, float]]
    cost: float
def astar(
    grid: GridMap2D,
    start_rc: Tuple[int, int],
    goal_rc: Tuple[int, int],
    cost_map: Optional[np.ndarray] = None,
    allow_diagonal: bool = True,
    goal_tolerance_m: float = 30.0,
) -> Optional[AStarResult]:
    """A* on a boolean occupancy grid.

    Args:
        grid: GridMap2D with inflated obstacles.
        start_rc: (row, col)
        goal_rc: (row, col), can be occupied when `goal_tolerance_m > 0`.
        cost_map: optional per-cell multiplicative cost (>=1 for free cells).
        allow_diagonal: use 8-neighborhood if True else 4-neighborhood.
        goal_tolerance_m: accept any free cell within this radius (meters)
            around `goal_rc` as terminal.
    """
    if not grid.in_bounds(*start_rc) or not grid.in_bounds(*goal_rc):
        return None
    if not grid.is_free(*start_rc):
        return None

    goal_tol = max(0.0, float(goal_tolerance_m))
    radius_cells = int(np.ceil(goal_tol / grid.resolution))
    goal_cells: Set[Tuple[int, int]] = set()

    if radius_cells <= 0:
        if not grid.is_free(*goal_rc):
            return None
        goal_cells.add((int(goal_rc[0]), int(goal_rc[1])))
    else:
        rr, cc = int(goal_rc[0]), int(goal_rc[1])
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                nr = rr + dr
                nc = cc + dc
                if not grid.is_free(nr, nc):
                    continue
                # World-space circular tolerance for consistency with success radius.
                if ((dr * grid.resolution) ** 2 + (dc * grid.resolution) ** 2) <= goal_tol**2:
                    goal_cells.add((nr, nc))
        if len(goal_cells) == 0:
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

    goal = (int(goal_rc[0]), int(goal_rc[1]))

    def heuristic_to_goal_set(rc: Tuple[int, int]) -> float:
        dr = float(rc[0] - goal[0])
        dc = float(rc[1] - goal[1])
        d_m = (dr * dr + dc * dc) ** 0.5 * grid.resolution
        # Lower bound to distance-to-any terminal cell within tolerance ball.
        return max(0.0, d_m - goal_tol)

    while open_heap:
        _f, current = heapq.heappop(open_heap)
        if current in goal_cells:
            # Reconstruct
            # print(f"Current {current} in {goal_cells}")
            path_rc: List[Tuple[int, int]] = [current]
            while current in came_from:
                current = came_from[current]
                path_rc.append(current)
            path_rc.reverse()

            path_xy = [grid.grid_to_world(r, c) for r, c in path_rc]
            # true_dist = float(np.linalg.norm(path_xy[-1] - grid.grid_to_world(goal_rc[0], goal_rc[1])))
            # print(grid.grid_to_world(goal_rc[0], goal_rc[1]))
            true_dist = float(np.linalg.norm(np.asarray(path_xy[-1]) - np.asarray(grid.grid_to_world(goal_rc[0], goal_rc[1]))))
            if true_dist > 30.0:
                print(true_dist)
            return AStarResult(path_rc=path_rc, path_xy=path_xy, cost=g_score[path_rc[-1]])

        current_g = g_score[current]
        for dr, dc in moves:
            nr = current[0] + dr
            nc = current[1] + dc
            if not grid.is_free(nr, nc):
                # print(f"grid {(nr, nc)}is not free")
                # exit(0)
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
                f_score = tentative + heuristic_to_goal_set(neighbor)
                heapq.heappush(open_heap, (f_score, neighbor))
    return None
