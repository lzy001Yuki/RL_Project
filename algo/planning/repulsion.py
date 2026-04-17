from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np


def add_path_repulsion(
    penalty_map: np.ndarray,
    path_rc: List[Tuple[int, int]],
    strength: float = 1.0,
    radius_cells: int = 2,
) -> np.ndarray:
    """Increase costs around an existing path to encourage alternative routes.

    Args:
        penalty_map: (H,W) float array (modified in-place and returned).
        path_rc: list of (row,col) cells along a path.
        strength: amount to add at the path centerline.
        radius_cells: expand penalty to a disk around each path cell.
    """
    if penalty_map.ndim != 2:
        raise ValueError("penalty_map must be 2D")
    if strength <= 0.0:
        return penalty_map

    h, w = penalty_map.shape
    rr, cc = np.mgrid[-radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1]
    disk = (rr**2 + cc**2) <= (radius_cells**2)
    offsets = np.stack([rr[disk], cc[disk]], axis=1)

    for row, col in path_rc:
        nbrs = offsets + np.array([row, col])[None, :]
        r = nbrs[:, 0]
        c = nbrs[:, 1]
        valid = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        penalty_map[r[valid], c[valid]] += strength
    return penalty_map

