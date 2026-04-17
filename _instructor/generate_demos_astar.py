"""
Generate expert demonstrations using A* on a rasterized occupancy grid.

The output is a simple dataset for behavioral cloning (BC):
  - obs_sensor: (M, 4) float32, matches env.uav_env observation "sensor"
  - actions:    (M, 2) float32, expert (dx, dy) actions within action limits

Usage:
  cd _instructor
  python generate_demos_astar.py \
    --pointcloud_path ../data/pointcloud_2d.npy \
    --initials_path ../data/eval_initials_20.json \
    --out_path ../data/astar_demos_eval20_res2.npz
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np

from env.pointcloud_utils import load_pointcloud_transposed, find_nearest_point
from planning.astar_grid import build_occupancy_grid, find_free_goal_xy_within_radius, plan_path_xy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pointcloud_path", type=str, required=True)
    p.add_argument("--initials_path", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--resolution", type=float, default=2.0, help="grid resolution (m/cell)")
    p.add_argument("--inflate_radius", type=float, default=2.5, help="obstacle inflation radius (m)")
    p.add_argument("--bounds_margin", type=float, default=10.0, help="extra margin around data bounds (m)")
    p.add_argument("--max_steps", type=int, default=300, help="cap expert rollout steps per episode")
    p.add_argument("--success_radius", type=float, default=30.0, help="success radius around target center (m)")
    p.add_argument("--action_limit", type=float, nargs=2, default=(2.0, 2.0))
    p.add_argument("--allow_diagonal", action="store_true", default=True)
    p.add_argument("--no_diagonal", dest="allow_diagonal", action="store_false")
    return p.parse_args()


def _compute_bounds(points_xy: np.ndarray, initials: list[dict[str, Any]], margin: float) -> tuple[float, float, float, float]:
    xs = [float(points_xy[:, 0].min()), float(points_xy[:, 0].max())]
    ys = [float(points_xy[:, 1].min()), float(points_xy[:, 1].max())]
    for init in initials:
        xs.append(float(init["x_start"]))
        xs.append(float(init["target_center_x"]))
        ys.append(float(init["y_start"]))
        ys.append(float(init["target_center_y"]))
    min_x = min(xs) - margin
    max_x = max(xs) + margin
    min_y = min(ys) - margin
    max_y = max(ys) + margin
    return min_x, max_x, min_y, max_y


def _sensor_obs(all_points_2xN: np.ndarray, curr_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    nearest_rel, _ = find_nearest_point(all_points_2xN, curr_xy)
    target_rel = target_xy - curr_xy
    sensor = np.concatenate([nearest_rel, target_rel]).astype(np.float32)
    return sensor


def main() -> None:
    args = parse_args()

    with open(args.initials_path) as f:
        initials = json.load(f)
    print(f"Loaded {len(initials)} initials from {args.initials_path}")

    points_xy = np.load(args.pointcloud_path)
    print(f"Loaded point cloud: {points_xy.shape} from {args.pointcloud_path}")

    bounds = _compute_bounds(points_xy, initials, args.bounds_margin)
    occupancy, grid = build_occupancy_grid(
        points_xy,
        bounds=bounds,
        resolution=args.resolution,
        inflate_radius_m=args.inflate_radius,
    )
    print(
        "Built occupancy grid: "
        f"shape={occupancy.shape}  res={grid.resolution:.2f}m  origin=({grid.origin_x:.1f},{grid.origin_y:.1f})"
    )

    all_points_2xN = load_pointcloud_transposed(args.pointcloud_path)

    action_limit = np.array(args.action_limit, dtype=np.float32)

    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    ep_id_list: list[int] = []
    t_list: list[int] = []

    n_ok = 0
    n_fail = 0
    failed_ids: list[int] = []
    for ep_id, init in enumerate(initials):
        start_xy = np.array([init["x_start"], init["y_start"]], dtype=np.float64)
        target_xy = np.array([init["target_center_x"], init["target_center_y"]], dtype=np.float64)

        goal_xy = find_free_goal_xy_within_radius(
            occupancy, grid, target_xy=target_xy, radius_m=args.success_radius
        )
        if goal_xy is None:
            # Fallback: try planning to the exact target center (A* will snap to a nearby free cell).
            goal_xy = target_xy

        path = plan_path_xy(
            occupancy,
            grid,
            start_xy=start_xy,
            goal_xy=goal_xy,
            allow_diagonal=args.allow_diagonal,
        )
        if path is None or len(path) < 2:
            n_fail += 1
            failed_ids.append(int(init.get("initial_id", ep_id)))
            continue

        # Rollout expert along consecutive waypoints.
        curr_xy = start_xy.copy()
        steps = 0
        for k in range(1, len(path)):
            if steps >= args.max_steps:
                break
            next_xy = path[k]
            delta = (next_xy - curr_xy).astype(np.float32)
            # Clip to action limits (per axis).
            delta = np.clip(delta, -action_limit, action_limit)
            if float(np.linalg.norm(delta)) < 1e-6:
                continue

            sensor = _sensor_obs(all_points_2xN, curr_xy, target_xy)

            obs_list.append(sensor)
            act_list.append(delta)
            ep_id_list.append(ep_id)
            t_list.append(steps)

            curr_xy = curr_xy + delta.astype(np.float64)
            steps += 1

            # Stop early when close enough to target.
            if float(np.linalg.norm(target_xy - curr_xy)) <= args.success_radius:
                break

        if steps > 0:
            n_ok += 1
        else:
            n_fail += 1
            failed_ids.append(int(init.get("initial_id", ep_id)))

    if len(obs_list) == 0:
        raise RuntimeError("No expert samples generated; check pointcloud/initials/bounds.")

    obs_sensor = np.stack(obs_list, axis=0).astype(np.float32)
    actions = np.stack(act_list, axis=0).astype(np.float32)
    ep_ids = np.asarray(ep_id_list, dtype=np.int32)
    ts = np.asarray(t_list, dtype=np.int32)

    print(f"Generated samples: {len(obs_sensor)} steps from {n_ok} initials, failed={n_fail}")
    if failed_ids:
        preview = failed_ids[:20]
        more = "..." if len(failed_ids) > 20 else ""
        print(f"Failed initial_id preview: {preview}{more}")
    print(f"Saving to {args.out_path} ...")
    np.savez_compressed(
        args.out_path,
        obs_sensor=obs_sensor,
        actions=actions,
        episode_id=ep_ids,
        t=ts,
        resolution=np.array([args.resolution], dtype=np.float32),
        inflate_radius=np.array([args.inflate_radius], dtype=np.float32),
    )
    print("Done.")


if __name__ == "__main__":
    main()
