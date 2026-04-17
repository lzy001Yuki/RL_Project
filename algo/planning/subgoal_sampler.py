from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import json
import numpy as np

from .astar import astar
from .grid_map import GridMap2D


@dataclass(frozen=True)
class SubgoalSamplerConfig:
    grid_resolution_m: float = 2.0
    grid_padding_m: float = 5.0
    grid_inflation_radius_m: float = 4.0
    allow_diagonal: bool = True

    # Plan to a free proxy cell near the target center (meters). In the official
    # evaluator, you only need to enter within 30m of the target center; the
    # center itself is often inside an occupied building cell.
    goal_success_radius_m: float = 30.0

    # If start is inside an inflated obstacle cell, snap to a nearby free cell
    # within this radius (meters).
    start_snap_radius_m: float = 10.0

    min_subgoal_dist_m: float = 40.0
    max_subgoal_dist_m: float = 160.0

    # If a planned path is shorter than min_subgoal_dist_m, fall back to this.
    fallback_fraction: float = 0.7

    # If no free proxy goal is found within goal_success_radius_m, expand the
    # radius and retry.
    goal_search_expand: float = 2.0
    goal_search_tries: int = 3


def _path_cumlen(path_xy: np.ndarray) -> np.ndarray:
    if len(path_xy) < 2:
        return np.asarray([0.0], dtype=np.float64)
    diffs = np.diff(path_xy, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _sample_point_at_distance(path_xy: np.ndarray, dist_m: float) -> np.ndarray:
    """Linear interpolation along polyline by arc-length."""
    if len(path_xy) == 0:
        raise ValueError("empty path")
    if len(path_xy) == 1:
        return path_xy[0].copy()

    cum = _path_cumlen(path_xy)
    total = float(cum[-1])
    d = float(np.clip(dist_m, 0.0, total))
    idx = int(np.searchsorted(cum, d, side="right") - 1)
    idx = max(0, min(idx, len(path_xy) - 2))

    d0 = float(cum[idx])
    d1 = float(cum[idx + 1])
    if d1 <= d0 + 1e-9:
        return path_xy[idx].copy()
    t = (d - d0) / (d1 - d0)
    return (1.0 - t) * path_xy[idx] + t * path_xy[idx + 1]


def _find_free_cell_within_radius(
    grid: GridMap2D,
    target_xy: np.ndarray,
    radius_m: float,
) -> Optional[Tuple[int, int]]:
    """Return a free cell (row,col) closest to target within radius, or None."""
    radius_m = float(radius_m)
    r0, c0 = grid.world_to_grid(float(target_xy[0]), float(target_xy[1]))
    max_cells = int(np.ceil(radius_m / grid.resolution))
    if max_cells < 0:
        return None

    best_rc: Optional[Tuple[int, int]] = None
    best_dist2 = float("inf")

    for dr in range(-max_cells, max_cells + 1):
        for dc in range(-max_cells, max_cells + 1):
            r = r0 + dr
            c = c0 + dc
            if not grid.is_free(r, c):
                continue
            x, y = grid.grid_to_world(r, c)
            d2 = (x - float(target_xy[0])) ** 2 + (y - float(target_xy[1])) ** 2
            if d2 <= radius_m**2 and d2 < best_dist2:
                best_dist2 = d2
                best_rc = (r, c)

    return best_rc


class PlannedSubgoalSampler:
    """Samples subgoals along a planned global path for each initial.

    This is a training-time helper for the "planning + landmark-conditioned low-level RL"
    method: the low-level controller trains on realistic corridor-following subgoals.
    """

    def __init__(
        self,
        pointcloud_path: str,
        initials_path: str,
        config: SubgoalSamplerConfig = SubgoalSamplerConfig(),
        seed: int = 1,
    ) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)

        points = np.load(pointcloud_path)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("pointcloud must have shape (N,2)")

        with open(initials_path) as f:
            self.initials = json.load(f)
        self._init_by_id = {int(i["initial_id"]): i for i in self.initials}

        # Expand grid bounds to include all starts/goals. Obstacle point clouds
        # often do not cover the full free-space extent.
        extra_pts: List[Tuple[float, float]] = []
        for init in self.initials:
            extra_pts.append((float(init["x_start"]), float(init["y_start"])))
            extra_pts.append((float(init["target_center_x"]), float(init["target_center_y"])))
        extra_arr = np.asarray(extra_pts, dtype=np.float64) if extra_pts else None

        self.grid = GridMap2D.from_pointcloud(
            points_xy=points,
            resolution_m=config.grid_resolution_m,
            padding_m=config.grid_padding_m,
            inflation_radius_m=config.grid_inflation_radius_m,
            extra_points_xy=extra_arr,
        )

        self._paths_xy: Dict[int, np.ndarray] = {}
        self._valid_initial_ids: List[int] = []

        for init in self.initials:
            iid = int(init["initial_id"])
            start_xy = np.array([float(init["x_start"]), float(init["y_start"])], dtype=np.float64)
            goal_xy = np.array([float(init["target_center_x"]), float(init["target_center_y"])], dtype=np.float64)

            start_rc = self.grid.world_to_grid(float(start_xy[0]), float(start_xy[1]))
            if not self.grid.in_bounds(*start_rc):
                continue
            if not self.grid.is_free(*start_rc):
                snapped = _find_free_cell_within_radius(self.grid, start_xy, radius_m=config.start_snap_radius_m)
                if snapped is None:
                    continue
                start_rc = snapped

            goal_rc: Optional[Tuple[int, int]] = None
            radius = float(config.goal_success_radius_m)
            for _ in range(max(1, int(config.goal_search_tries))):
                goal_rc = _find_free_cell_within_radius(self.grid, goal_xy, radius_m=radius)
                if goal_rc is not None:
                    break
                radius *= float(config.goal_search_expand)
            if goal_rc is None:
                continue

            result = astar(self.grid, start_rc, goal_rc, allow_diagonal=config.allow_diagonal)
            if result is None or len(result.path_xy) < 2:
                continue
            self._paths_xy[iid] = np.asarray(result.path_xy, dtype=np.float64)
            self._valid_initial_ids.append(iid)

        if not self._valid_initial_ids:
            raise RuntimeError(
                "No valid initials could be planned on the current grid. "
                "Try increasing grid_padding_m, decreasing grid_inflation_radius_m, "
                "or increasing goal_success_radius_m / start_snap_radius_m."
            )

    def sample(self) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a training task.

        Returns:
          initial_id, start_xy(2,), final_goal_xy(2,), subgoal_xy(2,)
        """
        iid = int(self.rng.choice(self._valid_initial_ids))
        init = self._init_by_id[iid]

        start_xy = np.array([float(init["x_start"]), float(init["y_start"])], dtype=np.float64)
        goal_xy = np.array([float(init["target_center_x"]), float(init["target_center_y"])], dtype=np.float64)

        path_xy = self._paths_xy[iid]
        cum = _path_cumlen(path_xy)
        total = float(cum[-1])

        min_d = float(self.config.min_subgoal_dist_m)
        max_d = float(self.config.max_subgoal_dist_m)

        if total <= min_d:
            dist = total * float(self.config.fallback_fraction)
        else:
            hi = min(max_d, total)
            lo = min_d
            if hi <= lo + 1e-6:
                dist = hi
            else:
                dist = float(self.rng.uniform(lo, hi))

        subgoal_xy = _sample_point_at_distance(path_xy, dist)
        return iid, start_xy, goal_xy, subgoal_xy

    def get_path_xy(self, initial_id: int) -> np.ndarray:
        """Return the cached planned global path for an initial_id (copy)."""
        iid = int(initial_id)
        if iid not in self._paths_xy:
            raise KeyError(f"Unknown or invalid initial_id: {iid}")
        return np.asarray(self._paths_xy[iid], dtype=np.float64).copy()
