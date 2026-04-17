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
    parser.add_argument("--grid_resolution_m", type=float, default=2.0)
    parser.add_argument("--grid_padding_m", type=float, default=5.0)
    parser.add_argument("--grid_inflation_radius_m", type=float, default=4.0)
    parser.add_argument("--goal_success_radius_m", type=float, default=30.0,
                        help="Plan to a free proxy cell within this radius of target center.")
    parser.add_argument("--start_snap_radius_m", type=float, default=10.0,
                        help="If start cell is occupied, snap to a nearby free cell within this radius.")

    # Residual RL: action_env = action_guide(A*) + action_residual(PPO)
    parser.add_argument("--residual_frac", type=float, default=0.35,
                        help="Fraction of action limit reserved for PPO residual (rest used by A* guide).")
    parser.add_argument("--guide_lookahead_m", type=float, default=10.0,
                        help="Lookahead distance (meters) along the A* path for guide action.")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dist_backend", type=str, default=None,
                        help="DDP backend (default: nccl if cuda else gloo)")
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


def main() -> None:
    args = parse_args()

    from algo.ppo.policy import GaussianActorCritic  # noqa: E402
    from algo.ppo.ppo import PPO  # noqa: E402
    from algo.ppo.storage import DictRolloutStorage  # noqa: E402
    from algo.envs.uav_subgoal_env import UAVEnvParams, UAVSubgoalEnv  # noqa: E402
    from algo.common.distributed import barrier, broadcast_module, broadcast_object  # noqa: E402
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

    with open(args.initials_path) as f:
        initials = json.load(f)
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

    _iid, start_xy, goal_xy, subgoal_xy = sampler.sample()
    path_xy = sampler.get_path_xy(_iid)
    path_cum = _path_cumlen(path_xy)
    obs = env.reset(
        initial_pose=np.array([start_xy[0], start_xy[1], 0.0], dtype=np.float64),
        subgoal_xy=subgoal_xy,
        final_goal_xy=goal_xy,
    )

    for key in obs:
        rollouts.obs[key][0].copy_(obs[key])
    rollouts.to(device)

    episode_rewards = deque(maxlen=50)
    start_time = time.time()

    if _is_main_process(args):
        print(
            f"\nStarting PPO training (residual mode: guide_frac={guide_frac:.2f}, residual_frac={residual_frac:.2f}) "
            f"(max_iter={args.max_iter}, world_size={getattr(args,'world_size',1)})...\n"
        )

    for j in range(args.max_iter):
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

            # Residual RL: environment action = A* guide + PPO residual.
            env_action = guide_action + act_out.action

            next_obs, reward, done, infos = env.step(env_action)

            for info in infos:
                if "episode" in info:
                    episode_rewards.append(float(info["episode"]["r"]))

            if done[0]:
                _iid, start_xy, goal_xy, subgoal_xy = sampler.sample()
                path_xy = sampler.get_path_xy(_iid)
                path_cum = _path_cumlen(path_xy)
                next_obs = env.reset(
                    initial_pose=np.array([start_xy[0], start_xy[1], 0.0], dtype=np.float64),
                    subgoal_xy=subgoal_xy,
                    final_goal_xy=goal_xy,
                )

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
                print(
                    f"[Iter {j:6d}] steps={total_steps:9d}  "
                    f"reward={np.mean(episode_rewards):7.1f}  "
                    f"v_loss={stats.value_loss:.4f}  "
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
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
