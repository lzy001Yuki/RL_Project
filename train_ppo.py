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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointcloud_path", type=str, required=True)
    parser.add_argument("--initials_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)

    parser.add_argument("--max_iter", type=int, default=20000)
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
    parser.add_argument("--grid_inflation_radius_m", type=float, default=4.0)
    parser.add_argument("--goal_success_radius_m", type=float, default=30.0,
                        help="Plan to a free proxy cell within this radius of target center.")
    parser.add_argument("--start_snap_radius_m", type=float, default=10.0,
                        help="If start cell is occupied, snap to a nearby free cell within this radius.")

    # Residual RL: action_env = action_guide(A*) + action_residual(PPO)
    parser.add_argument("--residual_frac", type=float, default=0.80,
                        help="Fraction of action limit reserved for PPO residual (rest used by A* guide).")
    parser.add_argument("--guide_lookahead_m", type=float, default=10.0,
                        help="Lookahead distance (meters) along the A* path for guide action.")
    parser.add_argument("--subgoal_stride_m", type=float, default=40.0,
                        help="Distance between chained subgoals along selected path.")
    parser.add_argument("--final_goal_radius_m", type=float, default=30.0,
                        help="Count as final success when within this radius of final goal center.")

    # Multi-path planning (instead of single fixed A* route).
    parser.add_argument("--n_paths", type=int, default=8,
                        help="Number of diverse A* candidates per initial.")
    parser.add_argument("--path_repulsion_strength", type=float, default=2.0)
    parser.add_argument("--path_repulsion_radius_cells", type=int, default=2)
    parser.add_argument("--path_repulsion_weight", type=float, default=2.0)
    parser.add_argument("--path_detour_ratio_max", type=float, default=1.8)
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

    # Online trajectory collection during training (rank0 only).
    parser.add_argument("--collect_during_train", action="store_true")
    parser.add_argument("--collect_initials_path", type=str, default=None,
                        help="Initials json used for collection filtering (default: initials_path).")
    parser.add_argument("--collect_take_first_n", type=int, default=20)
    parser.add_argument("--collect_trajs_per_initial", type=int, default=100)
    parser.add_argument("--collect_output_dir", type=str, default=None,
                        help="Output dir in baseline format (default: <save_dir>/baseline_trajs).")
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
                best_rc = (int(r), int(c))
    return best_rc


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
    from algo.planning.multi_path import MultiPathConfig, plan_diverse_paths  # noqa: E402
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

    initials = _load_initials(args.initials_path, take_first_n=None)
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

    multipath_cfg = MultiPathConfig(
        n_paths=int(args.n_paths),
        repulsion_strength=float(args.path_repulsion_strength),
        repulsion_radius_cells=int(args.path_repulsion_radius_cells),
        repulsion_weight=float(args.path_repulsion_weight),
        detour_ratio_max=float(args.path_detour_ratio_max),
        allow_diagonal=False,
    )

    start_by_iid: Dict[int, np.ndarray] = {}
    goal_by_iid: Dict[int, np.ndarray] = {}
    path_bank: Dict[int, List[np.ndarray]] = {}
    valid_initial_ids: List[int] = []

    for i, init in enumerate(initials):
        iid = _initial_id(init, i)
        start_xy, goal_xy = _extract_xy(init)

        start_rc = grid.world_to_grid(float(start_xy[0]), float(start_xy[1]))
        if not grid.in_bounds(*start_rc):
            continue
        if not grid.is_free(*start_rc):
            snapped = _find_free_cell_within_radius(grid, start_xy, radius_m=float(args.start_snap_radius_m))
            if snapped is None:
                continue
            start_rc = snapped

        goal_rc: Optional[Tuple[int, int]] = None
        radius = float(args.goal_success_radius_m)
        for _ in range(3):
            goal_rc = _find_free_cell_within_radius(grid, goal_xy, radius_m=radius)
            if goal_rc is not None:
                break
            radius *= 2.0
        if goal_rc is None:
            continue

        candidates = plan_diverse_paths(grid=grid, start_rc=start_rc, goal_rc=goal_rc, config=multipath_cfg)
        paths = [np.asarray(res.path_xy, dtype=np.float64) for res in candidates if len(res.path_xy) >= 2]
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
        print(f"Valid planned initials: {len(valid_initial_ids)} / {len(initials)}, avg candidate paths={avg_paths:.2f}")

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
    if bool(args.collect_during_train):
        collect_path = args.collect_initials_path or args.initials_path
        collect_initials = _load_initials(collect_path, take_first_n=int(args.collect_take_first_n))
        requested_ids = {_initial_id(init, i) for i, init in enumerate(collect_initials)}
        collect_ids = requested_ids.intersection(set(valid_initial_ids))
        if args.collect_output_dir is None:
            collect_output_dir = os.path.join(args.save_dir, "baseline_trajs")
        else:
            collect_output_dir = str(args.collect_output_dir)

        if _is_main_process(args):
            os.makedirs(collect_output_dir, exist_ok=True)
            for iid in sorted(collect_ids):
                init_dir = os.path.join(collect_output_dir, f"initial_{iid}")
                os.makedirs(init_dir, exist_ok=True)
                saved_counts[iid] = int(next_traj_index(init_dir))
            with open(os.path.join(args.save_dir, "collect_config.json"), "w") as f:
                json.dump(
                    {
                        "collect_initials_path": collect_path,
                        "collect_take_first_n": int(args.collect_take_first_n),
                        "collect_trajs_per_initial": int(args.collect_trajs_per_initial),
                        "collect_output_dir": collect_output_dir,
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

    residual_frac = float(args.residual_frac)
    if not (0.0 < residual_frac <= 1.0):
        raise ValueError("--residual_frac must be within (0,1]")
    guide_frac = 1.0 - residual_frac
    residual_action_limit = (full_action_limit[0] * residual_frac, full_action_limit[1] * residual_frac)
    guide_action_limit = (full_action_limit[0] * guide_frac, full_action_limit[1] * guide_frac)
    guide_action_scale = torch.tensor(guide_action_limit, dtype=torch.float32, device=device).unsqueeze(0)

    actor_critic = GaussianActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_limit=residual_action_limit,  # type: ignore[arg-type]
    ).to(device)

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
            path_cursor_by_iid[iid] = cursor + 1

        chosen_path = np.asarray(paths[chosen_idx], dtype=np.float64).copy()
        subgoals = _path_to_subgoals(chosen_path, stride_m=float(args.subgoal_stride_m))
        return iid, start_xy, goal_xy, chosen_path, subgoals

    curr_iid, start_xy, goal_xy, path_xy, curr_subgoals = _sample_episode_state()
    curr_subgoal_idx = 0
    path_cum = _path_cumlen(path_xy)
    obs = env.reset(
        initial_pose=np.array([start_xy[0], start_xy[1], 0.0], dtype=np.float64),
        subgoal_xy=curr_subgoals[curr_subgoal_idx],
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
        print(
            f"\nStarting PPO training (residual mode: guide_frac={guide_frac:.2f}, residual_frac={residual_frac:.2f}, "
            f"multi-path n_paths={int(args.n_paths)}, chained_subgoals stride={float(args.subgoal_stride_m):.1f}m) "
            f"(max_iter={args.max_iter}, world_size={getattr(args,'world_size',1)})...\n"
        )

    for j in range(args.max_iter):
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
            guide_dir = _guide_direction_from_path_with_cum(
                curr_xy=curr_xy,
                path_xy=path_xy,
                cum=path_cum,
                lookahead_m=float(args.guide_lookahead_m),
            )
            guide_action = torch.tensor(guide_dir, device=device, dtype=torch.float32).unsqueeze(0) * guide_action_scale
            # print(guide_action)
            # Residual RL: environment action = A* guide + PPO residual.
            env_action = guide_action + act_out.action
            # env_action = act_out.action

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
                        # Intermediate subgoal reached: move to next subgoal and keep same episode alive.
                        curr_subgoal_idx += 1
                        if curr_subgoal_idx >= len(curr_subgoals):
                            is_final = True
                            # print(f"==========Success Near!======= cu")
                            curr_subgoals.append(np.array(goal_xy, dtype=np.float64).copy())
                        # print(f"Reach subgoal{curr_subgoal_idx}, forward to next goal, total length of subgoal is {len(curr_subgoals)}")
                        next_obs = env.set_subgoal(curr_subgoals[curr_subgoal_idx], is_final)
                        done = [False]
                        infos = [{}]
                        # Remove terminal bonus for intermediate milestones.
                        reward = reward - float(env.params.success_bonus)
                        episode_return -= float(env.params.success_bonus)

                if done[0]:
                    episode_rewards.append(float(episode_return))
                    memory.update_with_trajectory(curr_iid, traj_xy, success=final_success)

                    if (
                        bool(args.collect_during_train)
                        and _is_main_process(args)
                        and collect_ids is not None
                        and collect_output_dir is not None
                        and (curr_iid in collect_ids)
                        and final_success
                        # and (curr_subgoal_idx == 9)
                    ):
                        k = int(saved_counts.get(curr_iid, 0))
                        if k < int(args.collect_trajs_per_initial):
                            out_path = os.path.join(collect_output_dir, f"initial_{curr_iid}", f"traj_{k}.txt")
                            trimmed = truncate_to_success(
                                traj_xy,
                                goal_xy,
                                success_radius_m=float(args.final_goal_radius_m),
                            )
                            trimmed = ensure_at_least_two_points(trimmed)
                            write_traj_txt(out_path, trimmed)
                            saved_counts[curr_iid] = k + 1

                    curr_iid, start_xy, goal_xy, path_xy, curr_subgoals = _sample_episode_state()
                    curr_subgoal_idx = 0
                    path_cum = _path_cumlen(path_xy)
                    next_obs = env.reset(
                        initial_pose=np.array([start_xy[0], start_xy[1], 0.0], dtype=np.float64),
                        subgoal_xy=curr_subgoals[curr_subgoal_idx],
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
            elapsed = time.time() - start_time
            total_steps = (j + 1) * args.num_steps * int(getattr(args, "world_size", 1))
            if _is_main_process(args):
                extra = ""
                if bool(args.collect_during_train) and collect_ids is not None:
                    extra = f"  collect={sum(saved_counts.get(iid, 0) for iid in collect_ids)}"
                print(
                    f"[Iter {j:6d}] steps={total_steps:9d}  "
                    f"reward={np.mean(episode_rewards):7.1f}  "
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
