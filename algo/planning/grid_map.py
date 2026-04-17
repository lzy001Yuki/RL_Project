from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    resolution_m: float
    min_x: float
    min_y: float
    width: int   # x-axis cells (cols)
    height: int  # y-axis cells (rows)


class GridMap2D:
    """2D occupancy grid with world<->grid transforms.

    Coordinate convention:
      - grid[row, col]
      - col increases with +x
      - row increases with +y (row=0 corresponds to min_y)
    """

    def __init__(self, spec: GridSpec, occupancy: np.ndarray) -> None:
        if occupancy.dtype != np.bool_:
            raise TypeError("occupancy must be boolean")
        if occupancy.shape != (spec.height, spec.width):
            raise ValueError(f"occupancy shape {occupancy.shape} != (H,W)=({spec.height},{spec.width})")
        self.spec = spec
        self.occupancy = occupancy

    @property
    def height(self) -> int:
        return self.spec.height

    @property
    def width(self) -> int:
        return self.spec.width

    @property
    def resolution(self) -> float:
        return self.spec.resolution_m

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        col = int(np.floor((x - self.spec.min_x) / self.spec.resolution_m))
        row = int(np.floor((y - self.spec.min_y) / self.spec.resolution_m))
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        x = self.spec.min_x + (col + 0.5) * self.spec.resolution_m
        y = self.spec.min_y + (row + 0.5) * self.spec.resolution_m
        return float(x), float(y)

    def is_free(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and (not self.occupancy[row, col])

    @staticmethod
    def from_pointcloud(
        points_xy: np.ndarray,
        resolution_m: float = 2.0,
        padding_m: float = 5.0,
        inflation_radius_m: float = 4.0,
        extra_points_xy: Optional[np.ndarray] = None,
    ) -> "GridMap2D":
        """Build an inflated occupancy grid from obstacle point cloud.

        Args:
            points_xy: (N,2) obstacle points in world coordinates.
            resolution_m: meters per cell.
            padding_m: extra border around min/max extents.
            inflation_radius_m: inflate obstacles by this clearance radius.
            extra_points_xy: optional (M,2) points to include in grid bounds
                (e.g., all starts/goals). These points do not create obstacles;
                they only expand the grid's world extents.
        """
        if points_xy.ndim != 2 or points_xy.shape[1] != 2:
            raise ValueError("points_xy must have shape (N,2)")
        if extra_points_xy is not None:
            extra_points_xy = np.asarray(extra_points_xy)
            if extra_points_xy.size > 0 and (extra_points_xy.ndim != 2 or extra_points_xy.shape[1] != 2):
                raise ValueError("extra_points_xy must have shape (M,2)")

        resolution_m = float(resolution_m)
        padding_m = float(padding_m)
        inflation_radius_m = float(inflation_radius_m)

        min_xy = points_xy.min(axis=0)
        max_xy = points_xy.max(axis=0)
        if extra_points_xy is not None and extra_points_xy.size > 0:
            min_xy = np.minimum(min_xy, extra_points_xy.min(axis=0))
            max_xy = np.maximum(max_xy, extra_points_xy.max(axis=0))
        min_x = float(min_xy[0] - padding_m)
        min_y = float(min_xy[1] - padding_m)
        max_x = float(max_xy[0] + padding_m)
        max_y = float(max_xy[1] + padding_m)

        width = int(np.ceil((max_x - min_x) / resolution_m))
        height = int(np.ceil((max_y - min_y) / resolution_m))

        spec = GridSpec(resolution_m=resolution_m, min_x=min_x, min_y=min_y, width=width, height=height)
        occupancy = np.zeros((height, width), dtype=np.bool_)

        # Map points to cells (deduplicate via set on flattened indices).
        cols = np.floor((points_xy[:, 0] - min_x) / resolution_m).astype(np.int64)
        rows = np.floor((points_xy[:, 1] - min_y) / resolution_m).astype(np.int64)
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        rows = rows[valid]
        cols = cols[valid]
        occupancy[rows, cols] = True

        if inflation_radius_m <= 0.0:
            return GridMap2D(spec, occupancy)

        radius_cells = int(np.ceil(inflation_radius_m / resolution_m))
        yy, xx = np.mgrid[-radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1]
        disk = (xx**2 + yy**2) <= (radius_cells**2)
        offsets = np.stack([yy[disk], xx[disk]], axis=1).astype(np.int64)

        inflated = occupancy.copy()
        h, w = occupancy.shape
        for dr, dc in offsets:
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = h - max(0, dr)
            dst_r0 = max(0, dr)
            dst_r1 = h - max(0, -dr)

            src_c0 = max(0, -dc)
            src_c1 = w - max(0, dc)
            dst_c0 = max(0, dc)
            dst_c1 = w - max(0, -dc)

            inflated[dst_r0:dst_r1, dst_c0:dst_c1] |= occupancy[src_r0:src_r1, src_c0:src_c1]

        return GridMap2D(spec, inflated)
