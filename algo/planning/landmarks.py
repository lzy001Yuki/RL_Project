from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    denom = float(np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    cos = float(np.clip(np.dot(u, v) / denom, -1.0, 1.0))
    return float(np.arccos(cos))


def extract_landmarks_from_path(
    path_xy: List[Tuple[float, float]],
    max_landmarks: int = 5,
    angle_threshold_deg: float = 30.0,
    min_separation_m: float = 25.0,
) -> List[Tuple[float, float]]:
    """Extract a small set of subgoals from a dense path.

    Heuristic:
      - Prefer points where direction changes significantly (turning points).
      - Enforce a minimum spatial separation to avoid trivial micro-subgoals.

    Returns:
        A list of landmarks (x,y) excluding the first point, and including the
        last point (goal) if space allows.
    """
    if len(path_xy) < 3:
        return list(path_xy[1:])

    pts = np.asarray(path_xy, dtype=np.float64)
    angle_thr = np.deg2rad(angle_threshold_deg)

    candidates: List[int] = []
    for i in range(1, len(pts) - 1):
        u = pts[i] - pts[i - 1]
        v = pts[i + 1] - pts[i]
        if np.linalg.norm(u) < 1e-6 or np.linalg.norm(v) < 1e-6:
            continue
        if _angle_between(u, v) >= angle_thr:
            candidates.append(i)

    selected: List[int] = []
    last_xy = pts[0]
    for idx in candidates:
        if np.linalg.norm(pts[idx] - last_xy) >= min_separation_m:
            selected.append(idx)
            last_xy = pts[idx]
        if len(selected) >= max_landmarks:
            break

    # Always try to include the goal as a final subgoal.
    if len(selected) < max_landmarks:
        if np.linalg.norm(pts[-1] - last_xy) >= 1e-6:
            selected.append(len(pts) - 1)
    return [(float(pts[i, 0]), float(pts[i, 1])) for i in selected]


def resample_path_by_distance(
    path_xy: List[Tuple[float, float]],
    step_m: float,
) -> List[Tuple[float, float]]:
    """Resample polyline at approximately uniform arc-length intervals."""
    if len(path_xy) < 2:
        return list(path_xy)

    pts = np.asarray(path_xy, dtype=np.float64)
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total <= step_m:
        return list(path_xy)

    n = int(np.floor(total / step_m)) + 1
    sample_at = np.linspace(0.0, total, n)
    xs = np.interp(sample_at, cum, pts[:, 0])
    ys = np.interp(sample_at, cum, pts[:, 1])
    return [(float(x), float(y)) for x, y in zip(xs, ys)]

