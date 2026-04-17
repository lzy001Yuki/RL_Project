"""
Generate a `submission/` folder using the "explicit planning + external memory"
method discussed for this assignment.

This script is intentionally usable in two modes:
  1) Planning-only (default): torch-free and deterministic. Produces collision-free
     trajectories on an inflated occupancy grid.
  2) RL execution (optional): follow planned landmarks using a trained low-level
     controller (PPO/SAC). Requires PyTorch and a saved policy.

Core idea:
  - Build an inflated occupancy grid from the obstacle point cloud.
  - For each (start, goal) initial:
      * Plan multiple paths with A* while adding an external "repulsion memory"
        cost to previously used corridors (encourages diverse routes).
      * Apply a goal-fading weight field so repulsion becomes weaker near the
        final goal region (improves realism and success rate).
      * Select 20 diverse candidates (default: greedy Jaccard on grid cells).
  - Save trajectories as `submission/initial_i/traj_j.txt` (x y per line).

Example:
  python generate_submission.py \
    --pointcloud_path data/pointcloud_2d.npy \
    --initials_path data/eval_initials_100.json \
    --output_dir submission \
    --trajs_per_initial 20
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from algo.planning.astar import AStarResult, astar
from algo.planning.grid_map import GridMap2D
from algo.planning.landmarks import extract_landmarks_from_path
from algo.planning.repulsion import add_path_repulsion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointcloud_path", type=str, required=True)
    parser.add_argument("--initials_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="submission")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--initial_ids", type=int, nargs="*", default=None, help="Optional subset of initial IDs")

    parser.add_argument("--trajs_per_initial", type=int, default=20)
    parser.add_argument("--n_candidates", type=int, default=40, help="Candidates per initial before selection")
    parser.add_argument("--selection", type=str, default="jaccard_greedy", choices=["none", "jaccard_greedy"])

    # Grid / planning
    parser.add_argument("--grid_resolution_m", type=float, default=2.0)
    parser.add_argument("--grid_padding_m", type=float, default=5.0)
    parser.add_argument("--grid_inflation_radius_m", type=float, default=4.0)
    parser.add_argument("--allow_diagonal", action="store_true")

    parser.add_argument("--success_radius_m", type=float, default=30.0)
    parser.add_argument("--detour_ratio_max", type=float, default=1.8)

    # External memory / repulsion
    parser.add_argument("--repulsion_strength", type=float, default=2.0)
    parser.add_argument("--repulsion_radius_cells", type=int, default=2)
    parser.add_argument("--repulsion_weight", type=float, default=2.0)
    parser.add_argument("--memory_decay", type=float, default=0.99)
    parser.add_argument("--memory_max", type=float, default=50.0)

    # Goal-fading (repulsion becomes weaker near goal)
    parser.add_argument("--fade_near_m", type=float, default=60.0)
    parser.add_argument("--fade_far_frac", type=float, default=0.8)
    parser.add_argument("--fade_w_min", type=float, default=0.1)

    # Tie-break noise
    parser.add_argument("--cost_noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)

    # Optional RL executor (requires torch)
    parser.add_argument("--executor", type=str, default="plan", choices=["plan", "ppo", "ppo_residual", "sac"])
    parser.add_argument("--policy_path", type=str, default=None, help="Path to final_policy.pt (PPO) or final_actor.pt (SAC)")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--subgoal_radius", type=float, default=5.0)
    parser.add_argument("--max_steps_per_segment", type=int, default=200)
    parser.add_argument("--max_steps_total", type=int, default=300)
    parser.add_argument("--max_landmarks", type=int, default=5)
    parser.add_argument("--guide_lookahead_m", type=float, default=10.0,
                        help="For ppo_residual: lookahead distance along the planned path for the guide action.")
    return parser.parse_args()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_traj(path: str, traj_xy: Sequence[Tuple[float, float]]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        for x, y in traj_xy:
            f.write(f"{float(x)} {float(y)}\n")


def _euclid(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    return float(np.linalg.norm(a_xy - b_xy))


def _path_cumlen(path_xy: np.ndarray) -> np.ndarray:
    if len(path_xy) < 2:
        return np.asarray([0.0], dtype=np.float64)
    diffs = np.diff(path_xy, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _sample_point_at_distance_with_cum(path_xy: np.ndarray, cum: np.ndarray, dist_m: float) -> np.ndarray:
    """Linear interpolation along polyline by arc-length, given precomputed cumlen."""
    if len(path_xy) == 0:
        raise ValueError("empty path")
    if len(path_xy) == 1:
        return path_xy[0].copy()

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


def _find_nearest_free_cell(grid: GridMap2D, xy: np.ndarray, max_radius_cells: int) -> Tuple[int, int]:
    r0, c0 = grid.world_to_grid(float(xy[0]), float(xy[1]))
    if grid.is_free(r0, c0):
        return r0, c0

    for rad in range(1, max_radius_cells + 1):
        r_min = r0 - rad
        r_max = r0 + rad
        c_min = c0 - rad
        c_max = c0 + rad
        # Perimeter scan (cheap)
        for r in range(r_min, r_max + 1):
            for c in (c_min, c_max):
                if grid.is_free(r, c):
                    return r, c
        for c in range(c_min + 1, c_max):
            for r in (r_min, r_max):
                if grid.is_free(r, c):
                    return r, c

    raise RuntimeError("Could not find a free grid cell near requested point")


def _find_free_goal_within_radius(
    grid: GridMap2D,
    target_xy: np.ndarray,
    success_radius_m: float,
) -> Tuple[int, int]:
    """Find a free goal proxy cell within the success radius."""
    r0, c0 = grid.world_to_grid(float(target_xy[0]), float(target_xy[1]))
    max_cells = int(np.ceil(success_radius_m / grid.resolution))
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
            if d2 <= success_radius_m**2 and d2 < best_dist2:
                best_dist2 = d2
                best_rc = (r, c)

    if best_rc is None:
        raise RuntimeError("No free goal proxy cell found within success radius")
    return best_rc


def _compute_fade_weight_field(
    grid: GridMap2D,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    near_m: float,
    far_frac: float,
    w_min: float,
) -> np.ndarray:
    """Compute per-cell fade weights in [w_min, 1]. Smaller near goal."""
    near_m = float(near_m)
    w_min = float(w_min)
    if not (0.0 <= w_min <= 1.0):
        raise ValueError("fade_w_min must be within [0,1]")

    start_to_goal = _euclid(start_xy, target_xy)
    far_m = max(near_m + 1.0, float(far_frac) * start_to_goal)

    rows = np.arange(grid.height, dtype=np.int64)
    cols = np.arange(grid.width, dtype=np.int64)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    x = grid.spec.min_x + (cc + 0.5) * grid.resolution
    y = grid.spec.min_y + (rr + 0.5) * grid.resolution
    dist = np.sqrt((x - float(target_xy[0])) ** 2 + (y - float(target_xy[1])) ** 2)

    w = (dist - near_m) / (far_m - near_m)
    w = np.clip(w, 0.0, 1.0)
    w = w_min + (1.0 - w_min) * w
    return w.astype(np.float32)


def _path_cells_set(path_rc: Sequence[Tuple[int, int]], width: int) -> set[int]:
    return {int(r) * int(width) + int(c) for r, c in path_rc}


def _jaccard_distance(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - (inter / union if union > 0 else 0.0)


def _select_diverse_paths_jaccard(
    candidates: List[AStarResult],
    grid_width: int,
    k: int,
) -> List[AStarResult]:
    if len(candidates) <= k:
        return candidates

    cell_sets = [_path_cells_set(c.path_rc, grid_width) for c in candidates]
    costs = [float(c.cost) for c in candidates]

    # Start from the shortest (most realistic) path.
    first = int(np.argmin(costs))
    selected = [first]
    remaining = [i for i in range(len(candidates)) if i != first]

    # Maintain min-distance-to-selected for each candidate to speed up greedy.
    min_d = {i: _jaccard_distance(cell_sets[i], cell_sets[first]) for i in remaining}

    while len(selected) < k and remaining:
        best_i = None
        best_score = -1.0
        best_cost = float("inf")
        for i in remaining:
            score = float(min_d[i])
            cost = costs[i]
            if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and cost < best_cost):
                best_i = i
                best_score = score
                best_cost = cost

        if best_i is None:
            break

        selected.append(best_i)
        remaining.remove(best_i)
        for i in remaining:
            d = _jaccard_distance(cell_sets[i], cell_sets[best_i])
            if d < min_d[i]:
                min_d[i] = d

    return [candidates[i] for i in selected]


def _truncate_to_success(
    traj_xy: List[Tuple[float, float]],
    target_xy: np.ndarray,
    success_radius_m: float,
) -> List[Tuple[float, float]]:
    sx, sy = float(target_xy[0]), float(target_xy[1])
    r2 = float(success_radius_m) ** 2
    for i, (x, y) in enumerate(traj_xy):
        if (x - sx) ** 2 + (y - sy) ** 2 <= r2:
            return traj_xy[: i + 1]
    return traj_xy


def _reaches_target(
    traj_xy: Sequence[Tuple[float, float]],
    target_xy: np.ndarray,
    success_radius_m: float,
) -> bool:
    sx, sy = float(target_xy[0]), float(target_xy[1])
    r2 = float(success_radius_m) ** 2
    for x, y in traj_xy:
        if (float(x) - sx) ** 2 + (float(y) - sy) ** 2 <= r2:
            return True
    return False


def _ensure_at_least_two_points(traj_xy: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """np.loadtxt returns 1D for single-line files; keep >=2 points."""
    if len(traj_xy) >= 2:
        return traj_xy
    if len(traj_xy) == 1:
        return [traj_xy[0], traj_xy[0]]
    return [(0.0, 0.0), (0.0, 0.0)]


def _execute_with_policy(
    planned_path_xy: List[Tuple[float, float]],
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    pointcloud_path: str,
    executor: str,
    policy_path: str,
    deterministic: bool,
    subgoal_radius: float,
    max_steps_per_segment: int,
    max_steps_total: int,
    max_landmarks: int,
    guide_lookahead_m: float,
):
    """Optional: Follow planned landmarks with a trained low-level controller."""
    import torch  # local import so planning-only runs without torch

    from algo.envs.uav_subgoal_env import UAVEnvParams, UAVSubgoalEnv

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = UAVSubgoalEnv(
        pointcloud_path=pointcloud_path,
        params=UAVEnvParams(
            max_steps=max_steps_per_segment,
            success_radius=subgoal_radius,
            collision_threshold=2.0,
            action_limit=(2.0, 2.0),
        ),
        device=device,
    )

    if executor in ("ppo", "ppo_residual"):
        from algo.ppo.policy import GaussianActorCritic

        policy = GaussianActorCritic(obs_dim=6, action_dim=2, action_limit=(2.0, 2.0)).to(device)
        policy.load_state_dict(torch.load(policy_path, map_location=device))
        policy.eval()

        def act(obs_dict):
            with torch.no_grad():
                out = policy.act(obs_dict, deterministic=deterministic)
            return out.action

    elif executor == "sac":
        from algo.sac.networks import SACActor

        policy = SACActor(obs_dim=6, action_dim=2, action_limit=(2.0, 2.0)).to(device)
        policy.load_state_dict(torch.load(policy_path, map_location=device))
        policy.eval()

        def act(obs_dict):
            with torch.no_grad():
                return policy.act(obs_dict, deterministic=deterministic).unsqueeze(0)

    else:
        raise ValueError(f"Unknown executor {executor}")

    path_xy_arr = np.asarray(planned_path_xy, dtype=np.float64)
    path_cum = _path_cumlen(path_xy_arr) if len(path_xy_arr) > 0 else np.asarray([0.0], dtype=np.float64)

    landmarks = extract_landmarks_from_path(planned_path_xy, max_landmarks=max_landmarks)
    if not landmarks or _euclid(np.asarray(landmarks[-1]), target_xy) > 1e-3:
        landmarks = list(landmarks) + [(float(target_xy[0]), float(target_xy[1]))]

    traj_xy: List[Tuple[float, float]] = [(float(start_xy[0]), float(start_xy[1]))]
    curr_pose = np.array([float(start_xy[0]), float(start_xy[1]), 0.0], dtype=np.float64)
    steps_total = 0

    for subgoal in landmarks:
        if steps_total >= max_steps_total:
            break
        subgoal_xy = np.asarray(subgoal, dtype=np.float64)
        obs = env.reset(initial_pose=curr_pose, subgoal_xy=subgoal_xy, final_goal_xy=target_xy)

        for _ in range(max_steps_per_segment):
            if steps_total >= max_steps_total:
                break
            action = act(obs)

            if executor == "ppo_residual":
                # Residual execution: env_action = A* guide + policy residual.
                curr_xy = np.array([float(env.curr_pose[0]), float(env.curr_pose[1])], dtype=np.float64)
                guide_dir = _guide_direction_from_path_with_cum(
                    curr_xy=curr_xy,
                    path_xy=path_xy_arr,
                    cum=path_cum,
                    lookahead_m=float(guide_lookahead_m),
                )
                full_lim = torch.tensor(env.action_limit, device=device, dtype=torch.float32).unsqueeze(0)
                residual_lim = policy.action_scale.to(device=device)  # (1,2)
                guide_lim = torch.clamp(full_lim - residual_lim, min=0.0)
                guide_action = torch.tensor(guide_dir, device=device, dtype=torch.float32).unsqueeze(0) * guide_lim
                action = action + guide_action

            obs, _reward, done, infos = env.step(action)
            traj_xy.append((float(env.curr_pose[0]), float(env.curr_pose[1])))
            steps_total += 1

            if done[0]:
                info = infos[0]
                curr_pose = np.asarray(info.get("final_pose", env.curr_pose), dtype=np.float64)
                if not bool(info.get("won", False)):
                    # Collision/timeout: abort this trajectory; caller may fall back.
                    return None
                break

    return traj_xy


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.overwrite and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    _ensure_dir(args.output_dir)

    with open(args.initials_path) as f:
        initials = json.load(f)
    init_by_id = {int(i["initial_id"]): i for i in initials}
    all_ids = sorted(init_by_id.keys())
    if args.initial_ids is not None and len(args.initial_ids) > 0:
        target_ids = [int(i) for i in args.initial_ids]
    else:
        target_ids = all_ids

    print("Building occupancy grid...")
    points = np.load(args.pointcloud_path)
    grid = GridMap2D.from_pointcloud(
        points_xy=points,
        resolution_m=args.grid_resolution_m,
        padding_m=args.grid_padding_m,
        inflation_radius_m=args.grid_inflation_radius_m,
    )
    print(f"Grid: {grid.height}x{grid.width}  res={grid.resolution:.2f}m  inflation={args.grid_inflation_radius_m:.1f}m")

    for iid in target_ids:
        init = init_by_id[iid]
        start_xy = np.array([float(init["x_start"]), float(init["y_start"])], dtype=np.float64)
        target_xy = np.array([float(init["target_center_x"]), float(init["target_center_y"])], dtype=np.float64)

        start_rc = _find_nearest_free_cell(grid, start_xy, max_radius_cells=10)
        goal_rc = _find_free_goal_within_radius(grid, target_xy, success_radius_m=args.success_radius_m)

        shortest = astar(grid, start_rc, goal_rc, cost_map=None, allow_diagonal=args.allow_diagonal)
        if shortest is None:
            raise RuntimeError(f"Initial {iid}: could not find any path on the grid")

        fade_w = _compute_fade_weight_field(
            grid=grid,
            start_xy=start_xy,
            target_xy=target_xy,
            near_m=args.fade_near_m,
            far_frac=args.fade_far_frac,
            w_min=args.fade_w_min,
        )

        memory = np.zeros_like(grid.occupancy, dtype=np.float32)
        candidates: List[AStarResult] = []

        base_cost = float(shortest.cost)
        max_attempts = max(args.n_candidates * 3, args.n_candidates + 10)

        repulsion_weight = float(args.repulsion_weight)
        for attempt in range(max_attempts):
            if len(candidates) >= args.n_candidates:
                break

            cost_map = 1.0 + repulsion_weight * (memory * fade_w)
            if args.cost_noise > 0.0:
                cost_map = cost_map + float(args.cost_noise) * rng.random(size=cost_map.shape, dtype=np.float32)

            path = astar(grid, start_rc, goal_rc, cost_map=cost_map, allow_diagonal=args.allow_diagonal)
            if path is None:
                repulsion_weight *= 0.7
                continue
            if path.cost > args.detour_ratio_max * base_cost:
                repulsion_weight *= 0.85
                continue

            candidates.append(path)

            # Update external memory.
            memory *= float(args.memory_decay)
            add_path_repulsion(
                memory,
                path.path_rc,
                strength=float(args.repulsion_strength),
                radius_cells=int(args.repulsion_radius_cells),
            )
            if args.memory_max > 0:
                np.clip(memory, 0.0, float(args.memory_max), out=memory)

        if len(candidates) < args.trajs_per_initial:
            raise RuntimeError(
                f"Initial {iid}: only planned {len(candidates)} candidates "
                f"(need {args.trajs_per_initial}). Try reducing repulsion or detour constraint."
            )

        if args.selection == "none" or args.n_candidates == args.trajs_per_initial:
            selected = candidates[: args.trajs_per_initial]
        else:
            selected = _select_diverse_paths_jaccard(
                candidates=candidates,
                grid_width=grid.width,
                k=args.trajs_per_initial,
            )

        out_dir = os.path.join(args.output_dir, f"initial_{iid}")
        _ensure_dir(out_dir)

        for j, plan in enumerate(selected):
            traj_path = os.path.join(out_dir, f"traj_{j}.txt")

            planned_xy = plan.path_xy
            traj_xy: Optional[List[Tuple[float, float]]]
            if args.executor == "plan":
                traj_xy = [(float(start_xy[0]), float(start_xy[1]))] + [(float(x), float(y)) for x, y in planned_xy]
            else:
                if args.policy_path is None:
                    raise ValueError("--policy_path is required when --executor is not 'plan'")
                traj_xy = _execute_with_policy(
                    planned_path_xy=planned_xy,
                    start_xy=start_xy,
                    target_xy=target_xy,
                    pointcloud_path=args.pointcloud_path,
                    executor=args.executor,
                    policy_path=args.policy_path,
                    deterministic=bool(args.deterministic),
                    subgoal_radius=float(args.subgoal_radius),
                    max_steps_per_segment=int(args.max_steps_per_segment),
                    max_steps_total=int(args.max_steps_total),
                    max_landmarks=int(args.max_landmarks),
                    guide_lookahead_m=float(args.guide_lookahead_m),
                )

                if traj_xy is None:
                    # Fallback to planned path if policy fails on this route.
                    traj_xy = [(float(start_xy[0]), float(start_xy[1]))] + [(float(x), float(y)) for x, y in planned_xy]

            traj_xy = _truncate_to_success(traj_xy, target_xy, success_radius_m=args.success_radius_m)
            if not _reaches_target(traj_xy, target_xy, success_radius_m=args.success_radius_m):
                # In executor mode, the policy can stop early (e.g., max_steps_total)
                # without ever reaching the official success radius. Ensure we only
                # write valid trajectories by falling back to the planned route.
                fallback = [(float(start_xy[0]), float(start_xy[1]))] + [(float(x), float(y)) for x, y in planned_xy]
                traj_xy = _truncate_to_success(fallback, target_xy, success_radius_m=args.success_radius_m)
                if not _reaches_target(traj_xy, target_xy, success_radius_m=args.success_radius_m):
                    raise RuntimeError(f"Initial {iid} traj {j}: did not reach success radius; refusing to write invalid file")

            traj_xy = _ensure_at_least_two_points(traj_xy)
            _write_traj(traj_path, traj_xy)

        print(f"Initial {iid}: saved {args.trajs_per_initial} trajectories to {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
