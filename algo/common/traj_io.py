from __future__ import annotations

import os
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_traj_txt(path: str, traj_xy: Sequence[Tuple[float, float]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        for x, y in traj_xy:
            f.write(f"{float(x)} {float(y)}\n")


def reaches_target(
    traj_xy: Iterable[Tuple[float, float]],
    target_xy: np.ndarray,
    success_radius_m: float = 30.0,
) -> bool:
    sx, sy = float(target_xy[0]), float(target_xy[1])
    r2 = float(success_radius_m) ** 2
    for x, y in traj_xy:
        if (float(x) - sx) ** 2 + (float(y) - sy) ** 2 <= r2:
            return True
    return False


def truncate_to_success(
    traj_xy: List[Tuple[float, float]],
    target_xy: np.ndarray,
    success_radius_m: float = 30.0,
) -> List[Tuple[float, float]]:
    sx, sy = float(target_xy[0]), float(target_xy[1])
    r2 = float(success_radius_m) ** 2
    for i, (x, y) in enumerate(traj_xy):
        if (float(x) - sx) ** 2 + (float(y) - sy) ** 2 <= r2:
            return traj_xy[: i + 1]
    return traj_xy


def ensure_at_least_two_points(traj_xy: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """np.loadtxt returns 1D for single-line files; keep >=2 points."""
    if len(traj_xy) >= 2:
        return traj_xy
    if len(traj_xy) == 1:
        return [traj_xy[0], traj_xy[0]]
    return [(0.0, 0.0), (0.0, 0.0)]


def next_traj_index(initial_dir: str) -> int:
    """Resume-friendly: return next available traj_{k}.txt index in the dir."""
    try:
        names = os.listdir(initial_dir)
    except Exception:
        return 0
    ids = []
    for name in names:
        if not (name.startswith("traj_") and name.endswith(".txt")):
            continue
        try:
            ids.append(int(name[len("traj_") : -len(".txt")]))
        except Exception:
            continue
    return (max(ids) + 1) if ids else 0

