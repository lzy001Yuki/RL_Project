from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    origin_x: float
    origin_y: float
    resolution: float

    @property
    def origin(self) -> np.ndarray:
        return np.array([self.origin_x, self.origin_y], dtype=np.float64)


def _xy_to_ij(xy: np.ndarray, grid: GridSpec) -> tuple[int, int]:
    """Convert continuous XY to grid indices (i=x, j=y)."""
    x, y = float(xy[0]), float(xy[1])
    i = int(math.floor((x - grid.origin_x) / grid.resolution))
    j = int(math.floor((y - grid.origin_y) / grid.resolution))
    return i, j


def _ij_to_xy_center(i: int, j: int, grid: GridSpec) -> np.ndarray:
    """Convert grid cell indices to the continuous XY of the cell center."""
    x = grid.origin_x + (i + 0.5) * grid.resolution
    y = grid.origin_y + (j + 0.5) * grid.resolution
    return np.array([x, y], dtype=np.float64)


def build_occupancy_grid(
    points_xy: np.ndarray,
    *,
    bounds: tuple[float, float, float, float],
    resolution: float,
    inflate_radius_m: float,
) -> tuple[np.ndarray, GridSpec]:
    """
    Build a boolean occupancy grid from point obstacles.

    Args:
        points_xy: (N, 2) obstacle points.
        bounds: (min_x, max_x, min_y, max_y) bounds for the grid.
        resolution: meters per grid cell.
        inflate_radius_m: inflate obstacles by this radius (in meters).

    Returns:
        occupancy: (nx, ny) bool array where True means occupied.
        grid: GridSpec containing origin and resolution.
    """
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2)")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if inflate_radius_m < 0:
        raise ValueError("inflate_radius_m must be >= 0")

    min_x, max_x, min_y, max_y = bounds
    nx = int(math.ceil((max_x - min_x) / resolution))
    ny = int(math.ceil((max_y - min_y) / resolution))
    nx = max(nx, 1)
    ny = max(ny, 1)

    grid = GridSpec(origin_x=min_x, origin_y=min_y, resolution=resolution)
    occupancy = np.zeros((nx, ny), dtype=np.bool_)

    # Rasterize points into the grid and inflate by stamping neighbors.
    ij = np.floor((points_xy - grid.origin) / resolution).astype(np.int32)
    ii = ij[:, 0]
    jj = ij[:, 1]
    valid = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
    ii = ii[valid]
    jj = jj[valid]

    inflate_cells = int(math.ceil(inflate_radius_m / resolution))
    if inflate_cells == 0:
        occupancy[ii, jj] = True
        return occupancy, grid

    for di in range(-inflate_cells, inflate_cells + 1):
        for dj in range(-inflate_cells, inflate_cells + 1):
            i2 = ii + di
            j2 = jj + dj
            v2 = (i2 >= 0) & (i2 < nx) & (j2 >= 0) & (j2 < ny)
            occupancy[i2[v2], j2[v2]] = True

    return occupancy, grid


def _snap_to_free(
    occupancy: np.ndarray,
    ij: tuple[int, int],
    *,
    max_radius_cells: int = 30,
) -> tuple[int, int] | None:
    """Find the nearest free cell around ij within a Manhattan radius."""
    nx, ny = occupancy.shape
    si, sj = ij
    # Clamp to bounds to avoid immediate failure when coordinates are slightly out-of-range.
    si = int(np.clip(si, 0, nx - 1))
    sj = int(np.clip(sj, 0, ny - 1))
    if 0 <= si < nx and 0 <= sj < ny and not occupancy[si, sj]:
        return si, sj

    for r in range(1, max_radius_cells + 1):
        for di in range(-r, r + 1):
            dj1 = r - abs(di)
            for dj in (-dj1, dj1):
                i = si + di
                j = sj + dj
                if 0 <= i < nx and 0 <= j < ny and not occupancy[i, j]:
                    return i, j
    return None


def find_free_goal_xy_within_radius(
    occupancy: np.ndarray,
    grid: GridSpec,
    *,
    target_xy: np.ndarray,
    radius_m: float,
) -> np.ndarray | None:
    """
    Pick a reachable goal cell center within a radius of target center.

    This is useful when the target center lies inside an obstacle (e.g., building interior),
    but the task defines success as reaching within a radius around it.

    Returns:
        goal_xy (cell center) or None if no free cell found within radius.
    """
    if radius_m <= 0:
        return None
    nx, ny = occupancy.shape
    ti, tj = _xy_to_ij(target_xy, grid)
    ti = int(np.clip(ti, 0, nx - 1))
    tj = int(np.clip(tj, 0, ny - 1))

    radius_cells = int(math.ceil(radius_m / grid.resolution))
    best_xy: np.ndarray | None = None
    best_dist = float("inf")

    for di in range(-radius_cells, radius_cells + 1):
        i = ti + di
        if i < 0 or i >= nx:
            continue
        for dj in range(-radius_cells, radius_cells + 1):
            j = tj + dj
            if j < 0 or j >= ny:
                continue
            if occupancy[i, j]:
                continue
            xy = _ij_to_xy_center(i, j, grid)
            d = float(np.linalg.norm(xy - target_xy))
            if d <= radius_m and d < best_dist:
                best_dist = d
                best_xy = xy

    return best_xy


def astar_search(
    occupancy: np.ndarray,
    start_ij: tuple[int, int],
    goal_ij: tuple[int, int],
    *,
    resolution: float,
    allow_diagonal: bool = True,
) -> list[tuple[int, int]] | None:
    """
    Run A* on a boolean occupancy grid.

    Args:
        occupancy: (nx, ny) bool array. True = blocked.
        start_ij: (i, j) start indices.
        goal_ij: (i, j) goal indices.
        resolution: meters per cell for costs/heuristic.
        allow_diagonal: use 8-connected moves when True, else 4-connected.

    Returns:
        Path as a list of (i, j) from start to goal (inclusive), or None if no path.
    """
    nx, ny = occupancy.shape
    if nx <= 0 or ny <= 0:
        return None

    def in_bounds(i: int, j: int) -> bool:
        return 0 <= i < nx and 0 <= j < ny

    start = _snap_to_free(occupancy, start_ij)
    goal = _snap_to_free(occupancy, goal_ij)
    if start is None or goal is None:
        return None

    start_i, start_j = start
    goal_i, goal_j = goal
    start_idx = start_i * ny + start_j
    goal_idx = goal_i * ny + goal_j

    moves: list[tuple[int, int, float]] = [
        (1, 0, resolution),
        (-1, 0, resolution),
        (0, 1, resolution),
        (0, -1, resolution),
    ]
    if allow_diagonal:
        diag = math.sqrt(2.0) * resolution
        moves += [
            (1, 1, diag),
            (1, -1, diag),
            (-1, 1, diag),
            (-1, -1, diag),
        ]

    g_score = np.full(nx * ny, np.inf, dtype=np.float32)
    parent = np.full(nx * ny, -1, dtype=np.int32)
    closed = np.zeros(nx * ny, dtype=np.uint8)

    def heuristic(i: int, j: int) -> float:
        return math.hypot(i - goal_i, j - goal_j) * resolution

    g_score[start_idx] = 0.0
    heap: list[tuple[float, float, int]] = [(heuristic(start_i, start_j), 0.0, start_idx)]

    while heap:
        f, g, idx = heapq.heappop(heap)
        if closed[idx]:
            continue
        closed[idx] = 1

        if idx == goal_idx:
            # Reconstruct
            path_rev: list[tuple[int, int]] = []
            cur = idx
            while cur != -1:
                i = int(cur // ny)
                j = int(cur % ny)
                path_rev.append((i, j))
                cur = int(parent[cur])
            path_rev.reverse()
            return path_rev

        i = int(idx // ny)
        j = int(idx % ny)

        for di, dj, cost in moves:
            ni = i + di
            nj = j + dj
            if not in_bounds(ni, nj):
                continue
            if occupancy[ni, nj]:
                continue
            # Avoid cutting corners on diagonal steps.
            if allow_diagonal and di != 0 and dj != 0:
                if occupancy[i + di, j] or occupancy[i, j + dj]:
                    continue

            nidx = ni * ny + nj
            if closed[nidx]:
                continue

            tentative_g = float(g_score[idx]) + cost
            if tentative_g < float(g_score[nidx]):
                g_score[nidx] = tentative_g
                parent[nidx] = idx
                heapq.heappush(heap, (tentative_g + heuristic(ni, nj), tentative_g, nidx))

    return None


def plan_path_xy(
    occupancy: np.ndarray,
    grid: GridSpec,
    *,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    allow_diagonal: bool = True,
) -> list[np.ndarray] | None:
    """
    Plan a path and return continuous XY waypoints (cell centers).
    """
    start_ij = _xy_to_ij(start_xy, grid)
    goal_ij = _xy_to_ij(goal_xy, grid)
    ij_path = astar_search(
        occupancy, start_ij, goal_ij, resolution=grid.resolution, allow_diagonal=allow_diagonal
    )
    if ij_path is None:
        return None
    return [_ij_to_xy_center(i, j, grid) for (i, j) in ij_path]
