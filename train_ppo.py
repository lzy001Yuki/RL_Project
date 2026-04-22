"""
Train PPO for the "planning + landmark/subgoal conditioned" method.

High level:
  - Use explicit planning (A* on an inflated occupancy grid) to sample a
    realistic intermediate subgoal along the start->goal route.
  - Train a low-level continuous controller with PPO to reach the subgoal
    while avoiding obstacles.

Usage:
  python train_ppo.py --pointcloud_path data/pointcloud_2d.npy \
                      --initials_path data/eval_initials_100.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointcloud_path", type=str, required=True)
    parser.add_argument("--initials_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)

    parser.add_argument("--max_iter", type=int, default=100000)
    parser.add_argument("--max_steps", type=int, default=300)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--num_steps", type=int, default=256)

    parser.add_argument("--ppo_epoch", type=int, default=4)
    parser.add_argument("--num_mini_batch", type=int, default=4)
    parser.add_argument("--clip_param", type=float, default=0.1)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    # Subgoal sampling / planning config
    parser.add_argument("--subgoal_radius", type=float, default=5.0)
    parser.add_argument("--subgoal_min_m", type=float, default=40.0)
    parser.add_argument("--subgoal_max_m", type=float, default=160.0)
    parser.add_argument("--grid_resolution_m", type=float, default=1.0)
    parser.add_argument("--grid_padding_m", type=float, default=5.0)
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Optional path to a saved policy checkpoint for continuing PPO training.")
    parser.add_argument("--plan_only", action="store_true",
                        help="Only run dynamic-cost A* planning and save planned paths, without PPO training.")
    parser.add_argument("--grid_inflation_radius_m", type=float, default=4.0)
    parser.add_argument("--goal_success_radius_m", type=float, default=30.0,
                        help="Plan to a free proxy cell within this radius of target center.")
    parser.add_argument("--start_snap_radius_m", type=float, default=10.0,
                        help="If start cell is occupied, snap to a nearby free cell within this radius.")

    # Optional residual RL: action_env = action_guide(A*) + action_residual(PPO)
    parser.add_argument("--use_action_guide", action="store_true",
                        help="Enable A* guide action and learn residual action with PPO.")
    parser.add_argument("--residual_frac", type=float, default=0.35,
                        help="Fraction of action limit reserved for PPO residual (rest used by A* guide).")
    parser.add_argument("--guide_lookahead_m", type=float, default=10.0,
                        help="Lookahead distance (meters) along the A* path for guide action.")
    parser.add_argument("--guide_gain_start", type=float, default=1.0,
                        help="Initial multiplier for guide action scale.")
    parser.add_argument("--guide_gain_end", type=float, default=1.0,
                        help="Final multiplier for guide action scale after curriculum decay.")
    parser.add_argument("--guide_gain_decay_iters", type=int, default=0,
                        help="Linear decay length in iterations (<=0 uses max_iter).")
    parser.add_argument("--subgoal_stride_m", type=float, default=40.0,
                        help="Distance between chained subgoals along selected path.")
    parser.add_argument("--final_goal_radius_m", type=float, default=30.0,
                        help="Count as final success when within this radius of final goal center.")

    # Multi-path planning (instead of single fixed A* route).
    parser.add_argument("--n_paths", type=int, default=50,
                        help="Number of diverse A* candidates per initial.")
    parser.add_argument("--path_repulsion_strength", type=float, default=5.0)
    parser.add_argument("--path_repulsion_radius_cells", type=int, default=2)
    parser.add_argument("--path_repulsion_weight", type=float, default=2.0)
    parser.add_argument("--path_detour_ratio_max", type=float, default=1.8)
    parser.add_argument("--path_memory_decay", type=float, default=0.99,
                        help="Decay factor of planning-time memory in dynamic-cost A*.")
    parser.add_argument("--path_memory_max", type=float, default=50.0,
                        help="Clip ceiling of planning-time memory map.")
    parser.add_argument("--path_fade_near_m", type=float, default=60.0,
                        help="Inside this distance to goal, repulsion weight fades toward path_fade_w_min.")
    parser.add_argument("--path_fade_far_frac", type=float, default=0.8,
                        help="Far distance = max(path_fade_near_m+1, frac*start_to_goal).")
    parser.add_argument("--path_fade_w_min", type=float, default=0.1,
                        help="Minimum fade weight near goal, in [0,1].")
    parser.add_argument("--path_cost_noise", type=float, default=0.0,
                        help="Optional random tie-break noise added to planning cost_map.")
    parser.add_argument("--path_select_random_prob", type=float, default=0.15,
                        help="Randomly sample a path with this probability for exploration.")
    parser.add_argument(
        "--path_select_mode",
        type=str,
        default="round_robin",
        choices=["round_robin", "memory", "random"],
        help=(
            "How to choose one path from per-initial path bank for each episode: "
            "round_robin cycles paths for the same initial; memory picks lowest-memory path; "
            "random samples uniformly."
        ),
    )

    # External memory (per initial): path selection + reward penalty.
    parser.add_argument("--memory_decay", type=float, default=0.99)
    parser.add_argument("--memory_max", type=float, default=50.0)
    parser.add_argument("--memory_radius_cells", type=int, default=2)
    parser.add_argument("--memory_strength_success", type=float, default=2.0)
    parser.add_argument("--memory_strength_failure", type=float, default=0.0)
    parser.add_argument("--memory_reward_weight", type=float, default=2.0,
                        help="Penalty scale for stepping onto high-memory cells.")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dist_backend", type=str, default=None,
                        help="DDP backend (default: nccl if cuda else gloo)")
    parser.add_argument("--path_shape_top_k", type=int, default=0,
                        help="Keep top-K shape-diverse paths per initial (<=0 keeps all planned paths).")
    parser.add_argument("--path_shape_turn_thresh_deg", type=float, default=20.0,
                        help="Turn event threshold (degrees) for path-shape feature extraction.")
    # Landmark pool (turning-point based) and selection.
    parser.add_argument("--landmark_turn_thresh_deg", type=float, default=25.0,
                        help="Turning angle threshold (degrees) for landmark candidate extraction.")
    parser.add_argument("--landmark_min_separation_m", type=float, default=20.0,
                        help="Minimum distance between successive turning points on the same path.")
    parser.add_argument("--landmark_dedup_radius_m", type=float, default=18.0,
                        help="Radius-NMS threshold for deduplicating turning-point candidates.")
    parser.add_argument("--landmark_cluster_radius_m", type=float, default=25.0,
                        help="Spatial clustering radius for landmark candidate aggregation.")
    parser.add_argument("--landmark_max_per_initial", type=int, default=24,
                        help="Maximum number of clustered landmarks kept per initial.")
    parser.add_argument("--landmark_min_progress_m", type=float, default=-10.0,
                        help="Discard landmark if expected goal progress is below this value.")
    parser.add_argument("--landmark_score_prog", type=float, default=1.0)
    parser.add_argument("--landmark_score_mem", type=float, default=1.2)
    parser.add_argument("--landmark_score_detour", type=float, default=0.2)
    parser.add_argument("--landmark_score_safe", type=float, default=0.2)
    parser.add_argument("--landmark_max_hops", type=int, default=12,
                        help="Upper bound on intermediate landmark transitions per episode.")
    # Online trajectory collection during training (rank0 only).
    parser.add_argument("--collect_during_train", action="store_true")
    parser.add_argument("--collect_initials_path", type=str, default=None,
                        help="Initials json used for collection filtering (default: initials_path).")
    parser.add_argument("--collect_take_first_n", type=int, default=20)
    parser.add_argument("--collect_trajs_per_initial", type=int, default=100)
    parser.add_argument("--collect_output_dir", type=str, default=None,
                        help="Output dir in baseline format (default: <save_dir>/baseline_trajs).")
    parser.add_argument("--collect_plan_output_dir", type=str, default=None,
                        help="Output dir for planned paths (default: <save_dir>/baseline_plans).")
    parser.add_argument("--stop_when_collected", action="store_true",
                        help="Stop training once collection target is met.")
    return parser.parse_args()


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
    return _sample_point_at_distance_with_cum(path_xy, cum=cum, dist_m=dist_m)


def _sample_point_at_distance_with_cum(path_xy: np.ndarray, cum: np.ndarray, dist_m: float) -> np.ndarray:
    """Linear interpolation along polyline by arc-length, given precomputed cumlen."""
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


def _guide_direction_from_path_with_cum(
    curr_xy: np.ndarray,
    path_xy: np.ndarray,
    cum: np.ndarray,
    lookahead_m: float,
) -> np.ndarray:
    """Unit direction to a lookahead point along a polyline path."""
    if len(path_xy) < 2:
        return np.zeros((2,), dtype=np.float32)
    diffs = path_xy - curr_xy.reshape(1, 2)
    j = int(np.argmin(np.sum(diffs * diffs, axis=1)))
    s = float(cum[j]) + float(lookahead_m)
    target = _sample_point_at_distance_with_cum(path_xy, cum=cum, dist_m=s)
    vec = target - curr_xy
    n = float(np.linalg.norm(vec))
    if n <= 1e-6:
        return np.zeros((2,), dtype=np.float32)
    return (vec / n).astype(np.float32)


def _guide_direction_from_path(curr_xy: np.ndarray, path_xy: np.ndarray, lookahead_m: float) -> np.ndarray:
    """Unit direction to a lookahead point along a polyline path (computes cumlen)."""
    return _guide_direction_from_path_with_cum(curr_xy, path_xy, _path_cumlen(path_xy), lookahead_m)


def _init_distributed(args: argparse.Namespace) -> torch.device:
    """Initialize torch.distributed if launched via torchrun."""
    use_dist = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if not use_dist:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        return torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    backend = args.dist_backend
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    torch.distributed.init_process_group(backend=backend, init_method="env://")
    args.rank = int(torch.distributed.get_rank())
    args.world_size = int(torch.distributed.get_world_size())
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        return torch.device(f"cuda:{args.local_rank}")
    return torch.device("cpu")


def _is_main_process(args: argparse.Namespace) -> bool:
    return int(getattr(args, "rank", 0)) == 0


def _load_initials(path: str, take_first_n: Optional[int]) -> List[dict]:
    with open(path) as f:
        initials = json.load(f)
    if take_first_n is not None and int(take_first_n) > 0:
        initials = initials[: int(take_first_n)]
    return list(initials)


def _initial_id(init: dict, fallback: int) -> int:
    try:
        return int(init.get("initial_id", fallback))
    except Exception:
        return int(fallback)


def _extract_xy(init: dict) -> Tuple[np.ndarray, np.ndarray]:
    start_xy = np.array([float(init["x_start"]), float(init["y_start"])], dtype=np.float64)
    goal_xy = np.array([float(init["target_center_x"]), float(init["target_center_y"])], dtype=np.float64)
    return start_xy, goal_xy


def _find_free_cell_within_radius(grid, target_xy: np.ndarray, radius_m: float) -> Optional[Tuple[int, int]]:
    radius_m = float(radius_m)
    r0, c0 = grid.world_to_grid(float(target_xy[0]), float(target_xy[1]))
    max_cells = int(np.ceil(radius_m / float(grid.resolution)))
    if max_cells < 0:
        return None

    best_rc: Optional[Tuple[int, int]] = None
    best_dist2 = float("inf")
    total_rc = []
    for dr in range(-max_cells, max_cells + 1):
        for dc in range(-max_cells, max_cells + 1):
            r = r0 + dr
            c = c0 + dc
            if not grid.is_free(r, c):
                continue
            x, y = grid.grid_to_world(r, c)
            d2 = (x - float(target_xy[0])) ** 2 + (y - float(target_xy[1])) ** 2
            if d2 <= radius_m**2:
                total_rc.append((int(r), int(c)))
            if d2 <= radius_m**2 and d2 < best_dist2:
                best_dist2 = d2
                best_rc = (int(r), int(c))
    return best_rc, total_rc

def _path_memory_mean(path_xy: np.ndarray, grid, memory_map: np.ndarray) -> float:
    vals: List[float] = []
    last_rc: Optional[Tuple[int, int]] = None
    for x, y in path_xy:
        r, c = grid.world_to_grid(float(x), float(y))
        if not grid.in_bounds(r, c):
            continue
        rc = (int(r), int(c))
        if last_rc is not None and rc == last_rc:
            continue
        vals.append(float(memory_map[rc[0], rc[1]]))
        last_rc = rc
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))


def _compute_fade_weight_field(
    grid,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    near_m: float,
    far_frac: float,
    w_min: float,
) -> np.ndarray:
    """Per-cell weight in [w_min, 1], smaller near goal."""
    near_m = float(near_m)
    w_min = float(w_min)
    if not (0.0 <= w_min <= 1.0):
        raise ValueError("--path_fade_w_min must be within [0,1]")

    start_to_goal = float(np.linalg.norm(start_xy - goal_xy))
    far_m = max(near_m + 1.0, float(far_frac) * start_to_goal)

    rows = np.arange(grid.height, dtype=np.int64)
    cols = np.arange(grid.width, dtype=np.int64)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    x = grid.spec.min_x + (cc + 0.5) * float(grid.resolution)
    y = grid.spec.min_y + (rr + 0.5) * float(grid.resolution)
    dist = np.sqrt((x - float(goal_xy[0])) ** 2 + (y - float(goal_xy[1])) ** 2)

    w = (dist - near_m) / (far_m - near_m)
    w = np.clip(w, 0.0, 1.0)
    w = w_min + (1.0 - w_min) * w
    return w.astype(np.float32)

def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi

def _path_to_subgoals(path_xy: np.ndarray, stride_m: float) -> List[np.ndarray]:
    """Convert a dense path into chained subgoals (excluding start, including end)."""
    if len(path_xy) < 2:
        return [np.asarray(path_xy[-1], dtype=np.float64)]
    cum = _path_cumlen(path_xy)
    total = float(cum[-1])
    if total <= 1e-6:
        return [np.asarray(path_xy[-1], dtype=np.float64)]

    step = max(1e-3, float(stride_m))
    dists = np.arange(step, total, step, dtype=np.float64)
    goals = [_sample_point_at_distance_with_cum(path_xy, cum=cum, dist_m=float(d)) for d in dists]
    goals.append(np.asarray(path_xy[-1], dtype=np.float64))
    return [np.asarray(g, dtype=np.float64) for g in goals]




def _path_shape_feature(path_xy: np.ndarray, turn_thresh_deg: float) -> np.ndarray:
    """Low-dim feature summarizing turn pattern of a path."""
    if len(path_xy) < 3:
        return np.zeros((12,), dtype=np.float64)

    diffs = np.diff(path_xy, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    valid = seg_len > 1e-9
    if int(np.sum(valid)) < 2:
        return np.zeros((12,), dtype=np.float64)
    diffs = diffs[valid]
    seg_len = seg_len[valid]

    heading = np.arctan2(diffs[:, 1], diffs[:, 0])
    dtheta = _wrap_to_pi(np.diff(heading))
    if len(dtheta) == 0:
        return np.zeros((12,), dtype=np.float64)

    total_len = float(np.sum(seg_len))
    abs_turn = float(np.sum(np.abs(dtheta)))
    signed_turn = float(np.sum(dtheta))
    turn_thresh = np.deg2rad(float(turn_thresh_deg))
    left_events = float(np.sum(dtheta > turn_thresh))
    right_events = float(np.sum(dtheta < -turn_thresh))
    turn_events = left_events + right_events
    turn_density = turn_events / max(1.0, total_len)

    # 6-bin turn histogram on [-pi, pi]
    hist, _ = np.histogram(dtheta, bins=6, range=(-np.pi, np.pi))
    hist = hist.astype(np.float64)
    hist = hist / max(1.0, float(np.sum(hist)))

    feat = np.concatenate(
        [
            np.asarray(
                [
                    total_len,
                    abs_turn,
                    signed_turn,
                    left_events,
                    right_events,
                    turn_events,
                    turn_density,
                ],
                dtype=np.float64,
            ),
            hist,
        ],
        axis=0,
    )
    return feat


def _select_shape_diverse_paths(
    paths_xy: List[np.ndarray],
    top_k: int,
    turn_thresh_deg: float,
) -> List[np.ndarray]:
    """Greedy farthest-point selection in turn-shape feature space."""
    if top_k <= 0 or len(paths_xy) <= top_k:
        return paths_xy

    feats = np.stack([_path_shape_feature(p, turn_thresh_deg=turn_thresh_deg) for p in paths_xy], axis=0)
    mean = np.mean(feats, axis=0, keepdims=True)
    std = np.std(feats, axis=0, keepdims=True) + 1e-6
    feats = (feats - mean) / std

    lengths = np.asarray([float(_path_cumlen(p)[-1]) if len(p) >= 2 else 0.0 for p in paths_xy], dtype=np.float64)
    first = int(np.argmin(lengths))
    selected = [first]
    remaining = [i for i in range(len(paths_xy)) if i != first]
    min_d = {i: float(np.linalg.norm(feats[i] - feats[first])) for i in remaining}

    while len(selected) < int(top_k) and remaining:
        best_i = None
        best_d = -1.0
        best_len = float("inf")
        for i in remaining:
            d = float(min_d[i])
            l = float(lengths[i])
            if d > best_d + 1e-12 or (abs(d - best_d) <= 1e-12 and l < best_len):
                best_i = i
                best_d = d
                best_len = l
        if best_i is None:
            break
        selected.append(best_i)
        remaining.remove(best_i)
        for i in remaining:
            d = float(np.linalg.norm(feats[i] - feats[best_i]))
            if d < min_d[i]:
                min_d[i] = d

    return [paths_xy[i] for i in selected]


@dataclass
class LandmarkCandidate:
    xy: np.ndarray
    priority: float
    support: int


def _extract_turn_candidates_from_path(
    path_xy: np.ndarray,
    turn_thresh_deg: float,
    min_separation_m: float,
) -> List[Tuple[np.ndarray, float]]:
    if len(path_xy) < 3:
        return []
    pts = np.asarray(path_xy, dtype=np.float64)
    turn_thresh = np.deg2rad(float(turn_thresh_deg))
    out: List[Tuple[np.ndarray, float]] = []
    last_xy = pts[0]
    for i in range(1, len(pts) - 1):
        u = pts[i] - pts[i - 1]
        v = pts[i + 1] - pts[i]
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu <= 1e-6 or nv <= 1e-6:
            continue
        cos = float(np.clip(np.dot(u, v) / (nu * nv + 1e-9), -1.0, 1.0))
        angle = float(np.arccos(cos))
        if angle < turn_thresh:
            continue
        if float(np.linalg.norm(pts[i] - last_xy)) < float(min_separation_m):
            continue
        out.append((pts[i].copy(), angle))
        last_xy = pts[i]
    return out


def _dedup_candidates_radius_nms(
    candidates: List[LandmarkCandidate],
    radius_m: float,
) -> List[LandmarkCandidate]:
    if len(candidates) <= 1 or float(radius_m) <= 0.0:
        return candidates
    ordered = sorted(candidates, key=lambda x: (float(x.priority), float(x.support)), reverse=True)
    kept: List[LandmarkCandidate] = []
    for cand in ordered:
        keep = True
        for old in kept:
            if float(np.linalg.norm(cand.xy - old.xy)) <= float(radius_m):
                keep = False
                break
        if keep:
            kept.append(cand)
    return kept


def _cluster_candidates_radius(
    candidates: List[LandmarkCandidate],
    radius_m: float,
    max_keep: int,
) -> List[np.ndarray]:
    if len(candidates) == 0:
        return []
    if float(radius_m) <= 0.0:
        pts = [np.asarray(c.xy, dtype=np.float64).copy() for c in candidates]
        if int(max_keep) > 0:
            return pts[: int(max_keep)]
        return pts

    n = len(candidates)
    visited = np.zeros((n,), dtype=bool)
    clusters: List[List[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        queue = [i]
        visited[i] = True
        comp: List[int] = []
        while queue:
            j = queue.pop()
            comp.append(j)
            for k in range(n):
                if visited[k]:
                    continue
                if float(np.linalg.norm(candidates[j].xy - candidates[k].xy)) <= float(radius_m):
                    visited[k] = True
                    queue.append(k)
        clusters.append(comp)

    centers: List[Tuple[np.ndarray, float]] = []
    for comp in clusters:
        weights = np.asarray([max(1e-3, float(candidates[idx].priority)) for idx in comp], dtype=np.float64)
        pts = np.stack([np.asarray(candidates[idx].xy, dtype=np.float64) for idx in comp], axis=0)
        center = np.sum(pts * weights.reshape(-1, 1), axis=0) / float(np.sum(weights))
        score = float(np.sum(weights))
        centers.append((center.astype(np.float64), score))
    centers = sorted(centers, key=lambda x: x[1], reverse=True)

    out = [c[0] for c in centers]
    if int(max_keep) > 0:
        out = out[: int(max_keep)]
    return out


def _build_landmark_pool_for_paths(
    paths: Sequence[np.ndarray],
    turn_thresh_deg: float,
    min_separation_m: float,
    dedup_radius_m: float,
    cluster_radius_m: float,
    max_landmarks: int,
) -> List[np.ndarray]:
    raw: List[LandmarkCandidate] = []
    support_count: Dict[Tuple[int, int], int] = {}
    for path in paths:
        cand = _extract_turn_candidates_from_path(
            np.asarray(path, dtype=np.float64),
            turn_thresh_deg=float(turn_thresh_deg),
            min_separation_m=float(min_separation_m),
        )
        for xy, angle in cand:
            key = (int(round(float(xy[0]))), int(round(float(xy[1]))))
            support_count[key] = int(support_count.get(key, 0)) + 1
            raw.append(LandmarkCandidate(xy=xy.copy(), priority=float(angle), support=1))

    if len(raw) == 0:
        return []

    for idx, item in enumerate(raw):
        key = (int(round(float(item.xy[0]))), int(round(float(item.xy[1]))))
        raw[idx].support = int(support_count.get(key, 1))
        raw[idx].priority = float(item.priority) * (1.0 + 0.25 * float(raw[idx].support - 1))

    deduped = _dedup_candidates_radius_nms(raw, radius_m=float(dedup_radius_m))
    clustered = _cluster_candidates_radius(
        deduped,
        radius_m=float(cluster_radius_m),
        max_keep=int(max_landmarks),
    )
    return [np.asarray(xy, dtype=np.float64).copy() for xy in clustered]


def _mean_memory_around_xy(memory_map: np.ndarray, grid, xy: np.ndarray, radius_cells: int) -> float:
    r, c = grid.world_to_grid(float(xy[0]), float(xy[1]))
    if not grid.in_bounds(r, c):
        return float(np.max(memory_map)) if memory_map.size > 0 else 0.0
    rad = max(0, int(radius_cells))
    r0 = max(0, int(r) - rad)
    r1 = min(memory_map.shape[0], int(r) + rad + 1)
    c0 = max(0, int(c) - rad)
    c1 = min(memory_map.shape[1], int(c) + rad + 1)
    patch = memory_map[r0:r1, c0:c1]
    if patch.size == 0:
        return float(memory_map[int(r), int(c)])
    return float(np.mean(patch))


def _select_next_landmark(
    curr_xy: np.ndarray,
    goal_xy: np.ndarray,
    landmarks: Sequence[np.ndarray],
    visited: Sequence[np.ndarray],
    memory_map: np.ndarray,
    grid,
    env,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    if len(landmarks) == 0:
        return None

    denom = max(1e-6, float(args.memory_max))
    visited_pts = [np.asarray(v, dtype=np.float64) for v in visited]
    curr_goal_dist = float(np.linalg.norm(curr_xy - goal_xy))

    best_xy: Optional[np.ndarray] = None
    best_score = -1e18
    for lm in landmarks:
        xy = np.asarray(lm, dtype=np.float64)
        if any(float(np.linalg.norm(xy - vp)) < max(1e-6, float(args.landmark_cluster_radius_m) * 0.5) for vp in visited_pts):
            continue

        next_goal_dist = float(np.linalg.norm(xy - goal_xy))
        progress = curr_goal_dist - next_goal_dist
        if progress < float(args.landmark_min_progress_m):
            continue
        detour = float(np.linalg.norm(curr_xy - xy) + np.linalg.norm(xy - goal_xy) - curr_goal_dist)
        mem_pen = _mean_memory_around_xy(memory_map, grid=grid, xy=xy, radius_cells=int(args.memory_radius_cells)) / denom
        _rel, obs_dist = env._nearest_obstacle(xy)  # noqa: SLF001
        safety = float(obs_dist)

        score = (
            float(args.landmark_score_prog) * progress
            - float(args.landmark_score_mem) * mem_pen
            - float(args.landmark_score_detour) * detour
            + float(args.landmark_score_safe) * safety
        )

        # tiny tie-break noise for exploration stability
        score += 1e-4 * float(rng.normal())
        if score > best_score:
            best_score = score
            best_xy = xy
    return None if best_xy is None else np.asarray(best_xy, dtype=np.float64).copy()


def main() -> None:
    args = parse_args()

    from algo.ppo.policy import GaussianActorCritic  # noqa: E402
    from algo.ppo.ppo import PPO  # noqa: E402
    from algo.ppo.storage import DictRolloutStorage  # noqa: E402
    from algo.envs.uav_subgoal_env import UAVEnvParams, UAVSubgoalEnv  # noqa: E402
    from algo.common.distributed import (  # noqa: E402
        all_reduce_tensor,
        barrier,
        broadcast_module,
        broadcast_object,
        get_world_size,
    )
    from algo.common.traj_io import ensure_at_least_two_points, next_traj_index, truncate_to_success, write_traj_txt  # noqa: E402
    from algo.memory.repulsion_memory import PerInitialRepulsionMemory, RepulsionMemoryConfig  # noqa: E402
    from algo.planning.astar import astar  # noqa: E402
    from algo.planning.repulsion import add_path_repulsion  # noqa: E402
    from algo.planning.subgoal_sampler import (  # noqa: E402
        PlannedSubgoalSampler,
        SubgoalSamplerConfig,
    )

    device = _init_distributed(args)

    # NOTE: For synchronous gradient averaging, all ranks must start from the
    # same network parameters. We therefore use a rank-invariant seed for model
    # initialization, and a rank-specific seed for environment / sampling noise.
    init_seed = int(args.seed)
    rank = int(getattr(args, "rank", 0))
    data_seed = init_seed + rank * 1000

    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(data_seed)
    random.seed(data_seed)
    torch.set_num_threads(1)

    initials = _load_initials(args.initials_path, take_first_n=args.collect_take_first_n)
    if _is_main_process(args):
        print(f"Loaded {len(initials)} initials")

    sampler_cfg = SubgoalSamplerConfig(
        grid_resolution_m=args.grid_resolution_m,
        grid_padding_m=args.grid_padding_m,
        grid_inflation_radius_m=args.grid_inflation_radius_m,
        goal_success_radius_m=args.goal_success_radius_m,
        start_snap_radius_m=args.start_snap_radius_m,
        min_subgoal_dist_m=args.subgoal_min_m,
        max_subgoal_dist_m=args.subgoal_max_m,
    )
    sampler = PlannedSubgoalSampler(
        pointcloud_path=args.pointcloud_path,
        initials_path=args.initials_path,
        config=sampler_cfg,
        seed=data_seed,
    )
    grid = sampler.grid

    start_by_iid: Dict[int, np.ndarray] = {}
    goal_by_iid: Dict[int, np.ndarray] = {}
    path_bank: Dict[int, List[np.ndarray]] = {}
    landmark_bank: Dict[int, List[np.ndarray]] = {}
    valid_initial_ids: List[int] = []   
    for i, init in enumerate(initials):
        iid = _initial_id(init, i)
        start_xy, goal_xy = _extract_xy(init)

        start_rc = grid.world_to_grid(float(start_xy[0]), float(start_xy[1]))
        if not grid.in_bounds(*start_rc):
            continue
        if not grid.is_free(*start_rc):
            snapped, _ = _find_free_cell_within_radius(grid, start_xy, radius_m=float(args.start_snap_radius_m))
            if snapped is None:
                continue
            start_rc = snapped

        goal_rc: Optional[Tuple[int, int]] = None
        radius = float(args.goal_success_radius_m)
        total_free_rc = []
        for _ in range(3):
            goal_rc, total_free_rc = _find_free_cell_within_radius(grid, goal_xy, radius_m=radius)
            if goal_rc is not None:
                # total_free_rc = [goal_rc]
                break
            radius *= 2.0
        if goal_rc is None:
            continue
        valid_goal_rc = []
        valid_goal = None
        for rc_goal in total_free_rc:
            shortest = astar(grid=grid, start_rc=start_rc, goal_rc=rc_goal, cost_map=None, allow_diagonal=False)
            if shortest is None:
                # print(f"Goal_rc {rc_goal} is invalid, switch to next")
                continue
            # valid_goal = rc_goal
            # goal_rc = rc_goal
            # break
            # print(f"Goal {rc_goal} has success path !")
            valid_goal_rc.append(rc_goal)


        fade_w = _compute_fade_weight_field(
            grid=grid,
            start_xy=start_xy,
            goal_xy=goal_xy,
            near_m=float(args.path_fade_near_m),
            far_frac=float(args.path_fade_far_frac),
            w_min=float(args.path_fade_w_min),
        )

        planning_memory = np.zeros_like(grid.occupancy, dtype=np.float32)
        candidates = []
        base_cost = float(shortest.cost)
        max_attempts = max(int(args.n_paths) * 3, int(args.n_paths) + 10)
        repulsion_weight = float(args.path_repulsion_weight)
        print(f"Calculating {i}th paths")
        for _attempt in range(max_attempts):
            if len(candidates) >= int(args.n_paths):
                break

            cost_map = 1.0 + repulsion_weight * (planning_memory * fade_w)
            if float(args.path_cost_noise) > 0.0:
                cost_map = cost_map + float(args.path_cost_noise) * np.random.random(
                    size=cost_map.shape
                ).astype(np.float32)
            goal_rc = random.choice(valid_goal_rc)
            # path = astar(
            #     grid=grid,
            #     start_rc=start_rc,
            #     goal_rc=goal_rc,
            #     cost_map=cost_map,
            #     allow_diagonal=False,
            # )
            path = astar(
                grid=grid,
                start_rc=start_rc,
                goal_rc=goal_rc,
                cost_map=cost_map,
                allow_diagonal=False,
            )
            # print(goal_xy)
            if path is None:
                repulsion_weight *= 0.7
                continue
            if float(path.cost) > float(args.path_detour_ratio_max) * base_cost:
                repulsion_weight *= 0.85
                continue
            true_dist = float(np.linalg.norm(path.path_xy[-1] - goal_xy))
            if true_dist > 30.0:
                print("Invalid Path")
                # exit(0)

            candidates.append(path)

            planning_memory *= float(args.path_memory_decay)
            add_path_repulsion(
                planning_memory,
                path.path_rc,
                strength=float(args.path_repulsion_strength),
                radius_cells=int(args.path_repulsion_radius_cells),
            )
            if float(args.path_memory_max) > 0.0:
                np.clip(planning_memory, 0.0, float(args.path_memory_max), out=planning_memory)
        
        paths = [np.asarray(res.path_xy, dtype=np.float64) for res in candidates if len(res.path_xy) >= 2]
        if int(args.path_shape_top_k) > 0:
            paths = _select_shape_diverse_paths(
                paths_xy=paths,
                top_k=int(args.path_shape_top_k),
                turn_thresh_deg=float(args.path_shape_turn_thresh_deg),
            )
        print(f"After selection, remaining {len(paths)} paths")
        if len(paths) == 0:
            continue

        valid_initial_ids.append(int(iid))
        start_by_iid[int(iid)] = start_xy
        goal_by_iid[int(iid)] = goal_xy
        path_bank[int(iid)] = paths
    if len(valid_initial_ids) == 0:
        raise RuntimeError("No valid initial could be planned with current grid/planning settings.")
    if _is_main_process(args):
        avg_paths = float(np.mean([len(path_bank[iid]) for iid in valid_initial_ids]))
        avg_landmarks = float(np.mean([len(landmark_bank.get(iid, [])) for iid in valid_initial_ids]))
        print(
            f"Valid planned initials: {len(valid_initial_ids)} / {len(initials)}, "
            f"avg candidate paths={avg_paths:.2f}, avg clustered landmarks={avg_landmarks:.2f}"
        )
        
        
    if True:
        if args.save_dir is None:
            args.save_dir = os.path.join("saved_data", f"ppo_{int(time.time())}")
        args.save_dir = broadcast_object(args.save_dir, src=0)

        if args.collect_plan_output_dir is None:
            collect_plan_output_dir = os.path.join(args.save_dir, "plan/baseline_trajs")
        else:
            collect_plan_output_dir = str(args.collect_plan_output_dir)
        collect_plan_output_dir = broadcast_object(collect_plan_output_dir, src=0)

        if _is_main_process(args):
            os.makedirs(args.save_dir, exist_ok=True)
            os.makedirs(collect_plan_output_dir, exist_ok=True)
            with open(os.path.join(args.save_dir, "config.json"), "w") as f:
                json.dump(vars(args), f, indent=2)

            if args.collect_initials_path is not None:
                plan_initials = _load_initials(args.collect_initials_path, take_first_n=int(args.collect_take_first_n))
                requested_ids = {_initial_id(init, i) for i, init in enumerate(plan_initials)}
                target_ids = [int(iid) for iid in sorted(requested_ids) if int(iid) in path_bank]
            else:
                target_ids = [int(iid) for iid in sorted(valid_initial_ids)]

            total_written = 0
            for iid in target_ids:
                out_init_dir = os.path.join(collect_plan_output_dir, f"initial_{iid}")
                os.makedirs(out_init_dir, exist_ok=True)
                n_write = min(int(args.n_paths), len(path_bank[iid]))
                valid_paths = []
                for k in range(n_write):
                    out_path = os.path.join(out_init_dir, f"traj_{k}.txt")
                    path_xy = np.asarray(path_bank[iid][k], dtype=np.float64)
                    
                    true_dist = float(np.linalg.norm(path_xy[-1] - goal_by_iid[iid]))
                    if true_dist > 30.0:
                        print(f"Invalid Path Planned! id:{k} for {iid} data. Goal {goal_xy}->curEnd{path_xy[-1]} {true_dist}")
                    else:
                        valid_paths.append(path_xy)
                        planned_with_start = [(float(start_by_iid[iid][0]), float(start_by_iid[iid][1]))] + [
                        (float(x), float(y)) for x, y in path_xy
                    ]
                        planned_with_start = ensure_at_least_two_points(planned_with_start)
                        write_traj_txt(out_path, planned_with_start)
                        total_written += 1
                print(f"Valide Path : {len(valid_paths)}")
                path_bank[iid] = valid_paths
            print(
                f"[plan_only] wrote {total_written} planned paths for {len(target_ids)} initials to "
                f"{collect_plan_output_dir}"
            )
        barrier()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    mem_cfg = RepulsionMemoryConfig(
        decay=float(args.memory_decay),
        max_value=float(args.memory_max),
        radius_cells=int(args.memory_radius_cells),
        strength_success=float(args.memory_strength_success),
        strength_failure=float(args.memory_strength_failure),
    )
    memory = PerInitialRepulsionMemory(
        grid=grid,
        initial_ids=valid_initial_ids,
        config=mem_cfg,
        seed=data_seed,
    )
    


    if args.save_dir is None:
        args.save_dir = os.path.join("saved_data", f"ppo_{int(time.time())}")
    # Ensure all ranks use the same save_dir (torchrun spawns processes simultaneously).
    args.save_dir = broadcast_object(args.save_dir, src=0)
    if _is_main_process(args):
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, "controllers"), exist_ok=True)
        with open(os.path.join(args.save_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    barrier()

    collect_ids: Optional[set[int]] = None
    saved_counts: Dict[int, int] = {}
    collect_output_dir: Optional[str] = None
    collect_plan_output_dir: Optional[str] = None
    if bool(args.collect_during_train):
        collect_path = args.collect_initials_path or args.initials_path
        collect_initials = _load_initials(collect_path, take_first_n=int(args.collect_take_first_n))
        requested_ids = {_initial_id(init, i) for i, init in enumerate(collect_initials)}
        collect_ids = requested_ids.intersection(set(valid_initial_ids))
        if args.collect_output_dir is None:
            collect_output_dir = os.path.join(args.save_dir, "baseline_trajs")
        else:
            collect_output_dir = str(args.collect_output_dir)
        if args.collect_plan_output_dir is None:
            collect_plan_output_dir = os.path.join(args.save_dir, "plan/baseline_trajs")
        else:
            collect_plan_output_dir = str(args.collect_plan_output_dir)

        if _is_main_process(args):
            os.makedirs(collect_output_dir, exist_ok=True)
            os.makedirs(collect_plan_output_dir, exist_ok=True)
            for iid in sorted(collect_ids):
                traj_init_dir = os.path.join(collect_output_dir, f"initial_{iid}")
                plan_init_dir = os.path.join(collect_plan_output_dir, f"initial_{iid}")
                os.makedirs(traj_init_dir, exist_ok=True)
                os.makedirs(plan_init_dir, exist_ok=True)
                saved_counts[iid] = int(next_traj_index(traj_init_dir))
            with open(os.path.join(args.save_dir, "collect_config.json"), "w") as f:
                json.dump(
                    {
                        "collect_initials_path": collect_path,
                        "collect_take_first_n": int(args.collect_take_first_n),
                        "collect_trajs_per_initial": int(args.collect_trajs_per_initial),
                        "collect_output_dir": collect_output_dir,
                        "collect_plan_output_dir": collect_plan_output_dir,
                        "final_goal_radius_m": float(args.final_goal_radius_m),
                    },
                    f,
                    indent=2,
                )
            dropped = sorted(int(x) for x in (requested_ids - collect_ids))
            if len(dropped) > 0:
                print(f"[collect] skip {len(dropped)} initials with no valid planned route: {dropped[:10]}")
        barrier()

    def _done_collecting() -> bool:
        if not bool(args.collect_during_train):
            return False
        if collect_ids is None:
            return False
        if len(collect_ids) == 0:
            return False
        target = int(args.collect_trajs_per_initial)
        if target <= 0:
            return False
        return all(int(saved_counts.get(iid, 0)) >= target for iid in collect_ids)

    env = UAVSubgoalEnv(
        pointcloud_path=args.pointcloud_path,
        params=UAVEnvParams(
            max_steps=args.max_steps,
            success_radius=args.subgoal_radius,
            collision_threshold=2.0,
            action_limit=(2.0, 2.0),
        ),
        device=device,
    )

    obs_dim = int(sum(int(np.prod(shape)) for shape in env.observation_shape.values()))
    action_dim = int(np.prod(env.action_shape))
    full_action_limit = tuple(float(x) for x in env.action_limit.tolist())

    use_action_guide = bool(args.use_action_guide)
    residual_frac = float(args.residual_frac)
    if use_action_guide:
        if not (0.0 < residual_frac <= 1.0):
            raise ValueError("--residual_frac must be within (0,1] when --use_action_guide is enabled")
        guide_frac = 1.0 - residual_frac
        policy_action_limit = (full_action_limit[0] * residual_frac, full_action_limit[1] * residual_frac)
        guide_action_limit = (full_action_limit[0] * guide_frac, full_action_limit[1] * guide_frac)
        guide_action_scale = torch.tensor(guide_action_limit, dtype=torch.float32, device=device).unsqueeze(0)
        guide_gain_start = float(args.guide_gain_start)
        guide_gain_end = float(args.guide_gain_end)
        if guide_gain_start < 0.0 or guide_gain_end < 0.0:
            raise ValueError("--guide_gain_start/--guide_gain_end must be >= 0")
        guide_gain_decay_iters = int(args.guide_gain_decay_iters)
        if guide_gain_decay_iters <= 0:
            guide_gain_decay_iters = int(args.max_iter)
    else:
        policy_action_limit = full_action_limit
        guide_action_scale = torch.tensor((0.0, 0.0), dtype=torch.float32, device=device).unsqueeze(0)
        guide_gain_start = 0.0
        guide_gain_end = 0.0
        guide_gain_decay_iters = 1

    def _guide_gain_at_iter(iter_idx: int) -> float:
        if not use_action_guide:
            return 0.0
        if guide_gain_decay_iters <= 0:
            return guide_gain_end
        t = float(np.clip(float(iter_idx) / float(guide_gain_decay_iters), 0.0, 1.0))
        return float((1.0 - t) * guide_gain_start + t * guide_gain_end)

    actor_critic = GaussianActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_limit=policy_action_limit,  # type: ignore[arg-type]
    ).to(device)
    if args.load_checkpoint is not None:
        ckpt_path = str(args.load_checkpoint)
        if _is_main_process(args):
            state = torch.load(ckpt_path, map_location=device)
            actor_critic.load_state_dict(state)
            print(f"Loaded checkpoint: {ckpt_path}")
    # Safety: ensure exact parameter match across ranks.
    broadcast_module(actor_critic, src=0)
    barrier()

    # After model init sync, re-seed torch RNG per-rank to decorrelate exploration
    # (stochastic actions, minibatch order) while keeping params synchronized.
    torch.manual_seed(data_seed)
    torch.cuda.manual_seed_all(data_seed)
    

    agent = PPO(
        actor_critic=actor_critic,
        clip_param=args.clip_param,
        ppo_epoch=args.ppo_epoch,
        num_mini_batch=args.num_mini_batch,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        lr=args.lr,
        eps=1e-5,
        max_grad_norm=args.max_grad_norm,
    )

    rollouts = DictRolloutStorage(
        num_steps=args.num_steps,
        num_envs=1,
        obs_shapes=env.observation_shape,
        action_shape=env.action_shape,
        recurrent_hidden_state_size=actor_critic.recurrent_hidden_state_size,
    )

    rng = np.random.default_rng(int(data_seed))
    path_cursor_by_iid: Dict[int, int] = {int(iid): 0 for iid in valid_initial_ids}

    def _sample_episode_state():
        iid = int(rng.choice(valid_initial_ids))
        start_xy = start_by_iid[iid]
        goal_xy = goal_by_iid[iid]
        paths = path_bank[iid]
        mem_map = memory.get(iid)
        landmarks = landmark_bank.get(iid, [])

        if len(paths) == 1:
            chosen_idx = 0
        elif str(args.path_select_mode) == "random":
            chosen_idx = int(rng.integers(0, len(paths)))
        elif str(args.path_select_mode) == "memory":
            if float(rng.random()) < float(args.path_select_random_prob):
                chosen_idx = int(rng.integers(0, len(paths)))
            else:
                scores = [_path_memory_mean(p, grid=grid, memory_map=mem_map) for p in paths]
                chosen_idx = int(np.argmin(np.asarray(scores, dtype=np.float64)))
        else:
            # round_robin: for the same initial, use different planned routes in turn.
            cursor = int(path_cursor_by_iid.get(iid, 0))
            chosen_idx = cursor % len(paths)
            # path_cursor_by_iid[iid] = int(path_cursor_by_iid.get(iid, 0)) + 1

        chosen_path = np.asarray(paths[chosen_idx], dtype=np.float64).copy()
        first_landmark = _select_next_landmark(
            curr_xy=np.asarray(start_xy, dtype=np.float64),
            goal_xy=np.asarray(goal_xy, dtype=np.float64),
            landmarks=landmarks,
            visited=[],
            memory_map=mem_map,
            grid=grid,
            env=env,
            args=args,
            rng=rng,
        )
        if first_landmark is None:
            first_landmark = np.asarray(goal_xy, dtype=np.float64).copy()
        return iid, start_xy, goal_xy, chosen_path, [np.asarray(x, dtype=np.float64).copy() for x in landmarks], first_landmark

    curr_iid, start_xy, goal_xy, path_xy, curr_landmarks, curr_target_xy = _sample_episode_state()
    visited_landmarks: List[np.ndarray] = []
    curr_landmark_hops = 0
    path_cum = _path_cumlen(path_xy)
    obs = env.reset(
        initial_pose=np.array([start_xy[0], start_xy[1], 0.0], dtype=np.float64),
        subgoal_xy=np.asarray(curr_target_xy, dtype=np.float64),
        final_goal_xy=goal_xy,
    )
    for key in obs:
        rollouts.obs[key][0].copy_(obs[key])
    rollouts.to(device)

    episode_rewards = deque(maxlen=50)
    traj_xy: List[Tuple[float, float]] = [(float(env.curr_pose[0]), float(env.curr_pose[1]))]
    episode_return = 0.0
    start_time = time.time()

    if _is_main_process(args):
        init_gain = _guide_gain_at_iter(0)
        end_gain = _guide_gain_at_iter(int(args.max_iter))
        mode = "residual+guide" if use_action_guide else "direct-goal-ppo"
        print(
            f"\nStarting PPO training ({mode}, guide_gain={init_gain:.2f}->{end_gain:.2f}, "
            f"multi-path n_paths={int(args.n_paths)}, landmark_cluster_r={float(args.landmark_cluster_radius_m):.1f}m) "
            f"(max_iter={args.max_iter}, world_size={getattr(args,'world_size',1)})...\n"
        )

    for j in range(args.max_iter):
        curr_guide_gain = _guide_gain_at_iter(j)
        curr_guide_action_scale = guide_action_scale * float(curr_guide_gain)
        if bool(args.collect_during_train) and bool(args.stop_when_collected):
            local_stop = 1 if (_is_main_process(args) and _done_collecting()) else 0
            if int(get_world_size()) > 1:
                t = torch.tensor([local_stop], device=device, dtype=torch.int64)
                all_reduce_tensor(t, average=False)
                if int(t.item()) > 0:
                    break
            else:
                if local_stop > 0:
                    break

        for step in range(args.num_steps):
            with torch.no_grad():
                obs_step = {k: rollouts.obs[k][step] for k in rollouts.obs}
                act_out = actor_critic.act(obs_step)

            curr_xy = np.array([float(env.curr_pose[0]), float(env.curr_pose[1])], dtype=np.float64)
            if use_action_guide:
                guide_dir = _guide_direction_from_path_with_cum(
                    curr_xy=curr_xy,
                    path_xy=path_xy,
                    cum=path_cum,
                    lookahead_m=float(args.guide_lookahead_m),
                )
                guide_action = torch.tensor(guide_dir, device=device, dtype=torch.float32).unsqueeze(0) * curr_guide_action_scale
                env_action = guide_action + act_out.action
            else:
                env_action = act_out.action

            next_obs, reward, done, infos = env.step(env_action)
            traj_xy.append((float(env.curr_pose[0]), float(env.curr_pose[1])))
            # print(traj_xy)
            # exit(0)

            # Memory-based reward penalty: discourage repeatedly using same corridor.
            if float(args.memory_reward_weight) > 0.0:
                r, c = grid.world_to_grid(float(env.curr_pose[0]), float(env.curr_pose[1]))
                if grid.in_bounds(r, c):
                    mval = float(memory.get(curr_iid)[int(r), int(c)])
                    denom = max(1e-6, float(args.memory_max))
                    reward = reward - float(args.memory_reward_weight) * (mval / denom)
            episode_return += float(reward.item())

            if done[0]:
                info0 = infos[0] if len(infos) > 0 else {}
                won = bool(info0.get("won", False))

                final_success = False
                is_final = False
                if won:
                    curr_xy_now = np.array([float(env.curr_pose[0]), float(env.curr_pose[1])], dtype=np.float64)
                    dist_to_goal = float(np.linalg.norm(curr_xy_now - goal_xy))
                    final_success = dist_to_goal <= float(args.final_goal_radius_m)

                    if not final_success:
                        visited_landmarks.append(np.asarray(curr_target_xy, dtype=np.float64).copy())
                        curr_landmark_hops += 1

                        next_target: Optional[np.ndarray] = None
                        if curr_landmark_hops < int(args.landmark_max_hops):
                            next_target = _select_next_landmark(
                                curr_xy=curr_xy_now,
                                goal_xy=np.asarray(goal_xy, dtype=np.float64),
                                landmarks=curr_landmarks,
                                visited=visited_landmarks,
                                memory_map=memory.get(curr_iid),
                                grid=grid,
                                env=env,
                                args=args,
                                rng=rng,
                            )
                        if next_target is None:
                            next_target = np.asarray(goal_xy, dtype=np.float64).copy()
                            is_final = True
                        else:
                            is_final = float(np.linalg.norm(next_target - goal_xy)) <= float(args.final_goal_radius_m)
                        curr_target_xy = np.asarray(next_target, dtype=np.float64).copy()

                        next_obs = env.set_subgoal(curr_target_xy, is_final)
                        done = [False]
                        infos = [{}]
                        reward = reward - float(env.params.success_bonus)
                        episode_return -= float(env.params.success_bonus)

                if done[0]:
                    episode_rewards.append(float(episode_return))
                    memory.update_with_trajectory(curr_iid, traj_xy, success=final_success)
                    if (
                        str(args.path_select_mode) == "round_robin"
                        and final_success
                        and len(path_bank.get(curr_iid, [])) > 1
                    ):
                        path_cursor_by_iid[curr_iid] = int(path_cursor_by_iid.get(curr_iid, 0)) + 1

                    if (
                        bool(args.collect_during_train)
                        and _is_main_process(args)
                        and collect_ids is not None
                        and collect_output_dir is not None
                        and collect_plan_output_dir is not None
                        and (curr_iid in collect_ids)
                        and final_success
                    ):
                        k = int(saved_counts.get(curr_iid, 0))
                        if k < int(args.collect_trajs_per_initial):
                            out_path = os.path.join(collect_output_dir, f"initial_{curr_iid}", f"traj_{k}.txt")
                            collect_plan_output_dir = os.path.join(collect_output_dir, f"baseline_trajs")
                            plan_out_path = os.path.join(collect_plan_output_dir, f"initial_{curr_iid}", f"traj_{k}.txt")
                            trimmed = truncate_to_success(
                                traj_xy,
                                goal_xy,
                                success_radius_m=float(args.final_goal_radius_m),
                            )
                            trimmed = ensure_at_least_two_points(trimmed)
                            write_traj_txt(out_path, trimmed)
                            planned_with_start = [
                                (float(start_xy[0]), float(start_xy[1]))
                            ] + [(float(x), float(y)) for x, y in path_xy]
                            planned_with_start = ensure_at_least_two_points(planned_with_start)
                            write_traj_txt(plan_out_path, planned_with_start)
                            saved_counts[curr_iid] = k + 1

                    curr_iid, start_xy, goal_xy, path_xy, curr_landmarks, curr_target_xy = _sample_episode_state()
                    visited_landmarks = []
                    curr_landmark_hops = 0
                    path_cum = _path_cumlen(path_xy)
                    next_obs = env.reset(
                        initial_pose=np.array([start_xy[0], start_xy[1], 0.0], dtype=np.float64),
                        subgoal_xy=np.asarray(curr_target_xy, dtype=np.float64),
                        final_goal_xy=goal_xy,
                    )
                    traj_xy = [(float(env.curr_pose[0]), float(env.curr_pose[1]))]
                    episode_return = 0.0

            masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done]).to(device)
            bad_masks = torch.ones_like(masks)
            rhs = torch.zeros(1, actor_critic.recurrent_hidden_state_size, device=device)
            rollouts.insert(
                obs=next_obs,
                recurrent_hidden_states=rhs,
                actions=act_out.action,  # store residual action for PPO update
                action_log_probs=act_out.action_log_prob,
                value_preds=act_out.value,
                rewards=reward,
                masks=masks,
                bad_masks=bad_masks,
            )

        with torch.no_grad():
            next_value = actor_critic.get_value({k: rollouts.obs[k][-1] for k in rollouts.obs}).detach()

        rollouts.compute_returns(next_value, use_gae=True, gamma=args.gamma, gae_lambda=args.gae_lambda)
        stats = agent.update(rollouts)
        rollouts.after_update()

        if j % args.log_interval == 0 and len(episode_rewards) > 0:
        # if True:
            elapsed = time.time() - start_time
            total_steps = (j + 1) * args.num_steps * int(getattr(args, "world_size", 1))
            if _is_main_process(args):
                extra = ""
                if bool(args.collect_during_train) and collect_ids is not None:
                    extra = f"  collect={sum(saved_counts.get(iid, 0) for iid in collect_ids)}"
                print(
                    f"[Iter {j:6d}] steps={total_steps:9d}  "
                    f"reward={np.mean(episode_rewards):7.1f}  "
                    f"guide_gain={curr_guide_gain:.3f}  "
                    f"v_loss={stats.value_loss:.4f}{extra}  "
                    f"elapsed={elapsed:.0f}s"
                )
                with open(os.path.join(args.save_dir, "train_log.txt"), "a") as f:
                    f.write(
                        f"{j}\t{float(np.mean(episode_rewards)):.4f}\t"
                        f"{stats.value_loss:.6f}\t{stats.action_loss:.6f}\t{stats.dist_entropy:.6f}\n"
                    )

        if j % args.save_interval == 0 and j > 0:
            if _is_main_process(args):
                torch.save(actor_critic.state_dict(), os.path.join(args.save_dir, "controllers", f"{j}_policy.pt"))

    if _is_main_process(args):
        torch.save(actor_critic.state_dict(), os.path.join(args.save_dir, "controllers", "final_policy.pt"))
        print("\nTraining complete!")
        if bool(args.collect_during_train) and collect_ids is not None:
            print(f"Collected counts (by initial_id): { {iid: saved_counts.get(iid, 0) for iid in sorted(collect_ids)} }")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
'''
torchrun --standalone --nproc_per_node=2 train_ppo.py \
    --pointcloud_path data/pointcloud_2d.npy \
    --initials_path data/eval_initials_20.json \
    --save_dir saved_data/ppo_residual_multi_debug \
    --collect_during_train \
    --collect_take_first_n 20 \
    --collect_trajs_per_initial 100 \
    --stop_when_collected \
    --n_paths 100 \
    --subgoal_stride_m 35 \
    --final_goal_radius_m 30 \
    --memory_reward_weight 2.0 --path_select_mode random --plan_only --path_shape_top_k 5 --path_shape_turn_thresh_deg 25
    
    
    torchrun --standalone --nproc_per_node=2 train_ppo.py \
    --pointcloud_path data/pointcloud_2d.npy \
    --initials_path data/eval_initials_20.json \
    --save_dir saved_data/ppo_residual_multi_long_topk5_19 \
    --collect_during_train \
    --collect_take_first_n 20 \
    --collect_trajs_per_initial 3 \
    --stop_when_collected \
    --n_paths 50 \
    --subgoal_stride_m 35 \
    --final_goal_radius_m 30 \
    --memory_reward_weight 2.0 --path_select_mode round_robin --residual_frac 0.4  --path_shape_top_k 5 --path_shape_turn_thresh_deg 25 --plan_only
'''
