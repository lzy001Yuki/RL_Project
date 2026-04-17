"""
Train SAC for the "planning + landmark/subgoal conditioned" method.

High level:
  - Use explicit planning (A* on an inflated occupancy grid) to sample a
    realistic intermediate subgoal along the start->goal route.
  - Train a low-level continuous controller with SAC to reach the subgoal.

Usage:
  python train_sac.py --pointcloud_path data/pointcloud_2d.npy \
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

    parser.add_argument("--max_env_steps", type=int, default=2_000_000)
    parser.add_argument("--max_steps", type=int, default=300)

    parser.add_argument("--replay_size", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--start_steps", type=int, default=20_000)
    parser.add_argument("--update_after", type=int, default=5_000)
    parser.add_argument("--updates_per_step", type=int, default=1)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)

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

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=2000)
    parser.add_argument("--save_interval", type=int, default=200_000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dist_backend", type=str, default=None,
                        help="DDP backend (default: nccl if cuda else gloo)")
    return parser.parse_args()


def _init_distributed(args: argparse.Namespace) -> torch.device:
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

    from algo.common.torch_utils import obs_to_tensor  # noqa: E402
    from algo.common.distributed import barrier, broadcast_module, broadcast_object, broadcast_tensor  # noqa: E402
    from algo.sac.replay_buffer import DictReplayBuffer  # noqa: E402
    from algo.sac.sac import SAC  # noqa: E402
    from algo.envs.uav_subgoal_env import UAVEnvParams, UAVSubgoalEnv  # noqa: E402
    from algo.planning.subgoal_sampler import (  # noqa: E402
        PlannedSubgoalSampler,
        SubgoalSamplerConfig,
    )

    device = _init_distributed(args)

    init_seed = int(args.seed)
    rank = int(getattr(args, "rank", 0))
    data_seed = init_seed + rank * 1000

    # Keep initialization identical across ranks; vary data collection / sampling.
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
        args.save_dir = os.path.join("saved_data", f"sac_{int(time.time())}")
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
    action_limit = tuple(float(x) for x in env.action_limit.tolist())

    agent = SAC(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_limit=action_limit,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        device=device,
    )

    # Safety: ensure exact parameter match across ranks.
    broadcast_module(agent.actor, src=0)
    broadcast_module(agent.critic, src=0)
    broadcast_module(agent.critic_target, src=0)
    broadcast_tensor(agent.log_alpha.data, src=0)
    barrier()

    # After init sync, re-seed torch RNG per-rank to decorrelate exploration.
    torch.manual_seed(data_seed)
    torch.cuda.manual_seed_all(data_seed)

    buffer = DictReplayBuffer(
        capacity=args.replay_size,
        obs_shapes=env.observation_shape,
        action_dim=action_dim,
    )

    _iid, start_xy, goal_xy, subgoal_xy = sampler.sample()
    obs = env.reset(
        initial_pose=np.array([start_xy[0], start_xy[1], 0.0], dtype=np.float64),
        subgoal_xy=subgoal_xy,
        final_goal_xy=goal_xy,
    )

    episode_rewards = deque(maxlen=50)
    start_time = time.time()

    if _is_main_process(args):
        print(
            f"\nStarting SAC training (max_env_steps={args.max_env_steps}, world_size={getattr(args,'world_size',1)})...\n"
        )

    for t in range(1, args.max_env_steps + 1):
        obs_t = obs_to_tensor(obs, device=device)

        if t < args.start_steps:
            action = (
                torch.empty((1, action_dim), device=device).uniform_(-1.0, 1.0)
                * torch.tensor(action_limit, device=device).unsqueeze(0)
            )
        else:
            with torch.no_grad():
                action = agent.act(obs_t, deterministic=False)

        next_obs, reward_t, done, infos = env.step(action)
        done_bool = bool(done[0])
        reward_float = float(reward_t.item()) if isinstance(reward_t, torch.Tensor) else float(reward_t)

        obs_np = {k: v.detach().cpu().numpy() for k, v in obs.items()}
        next_obs_np = {k: v.detach().cpu().numpy() for k, v in next_obs.items()}
        action_np = action.squeeze(0).detach().cpu().numpy()
        buffer.add(obs_np, action_np, reward_float, next_obs_np, done_bool)

        if done_bool:
            _iid, start_xy, goal_xy, subgoal_xy = sampler.sample()
            obs = env.reset(
                initial_pose=np.array([start_xy[0], start_xy[1], 0.0], dtype=np.float64),
                subgoal_xy=subgoal_xy,
                final_goal_xy=goal_xy,
            )
        else:
            obs = next_obs

        for info in infos:
            if "episode" in info:
                episode_rewards.append(float(info["episode"]["r"]))

        if t >= args.update_after and len(buffer) >= args.batch_size:
            for _ in range(args.updates_per_step):
                batch = buffer.sample(args.batch_size, device=device)
                stats = agent.update(batch)

        if t % args.log_interval == 0:
            elapsed = time.time() - start_time
            ep_r = float(np.mean(episode_rewards)) if len(episode_rewards) > 0 else float("nan")
            alpha = stats.alpha if "stats" in locals() else float("nan")
            if _is_main_process(args):
                global_steps = int(getattr(args, "world_size", 1)) * int(t)
                print(
                    f"[Step {global_steps:9d}] reward={ep_r:7.1f}  alpha={alpha:.3f}  elapsed={elapsed:.0f}s"
                )
                with open(os.path.join(args.save_dir, "train_log.txt"), "a") as f:
                    f.write(f"{global_steps}\t{ep_r:.4f}\t{alpha:.6f}\n")

        if t % args.save_interval == 0:
            if _is_main_process(args):
                torch.save(agent.actor.state_dict(), os.path.join(args.save_dir, "controllers", f"{t}_actor.pt"))
                torch.save(agent.critic.state_dict(), os.path.join(args.save_dir, "controllers", f"{t}_critic.pt"))

    if _is_main_process(args):
        torch.save(agent.actor.state_dict(), os.path.join(args.save_dir, "controllers", "final_actor.pt"))
        torch.save(agent.critic.state_dict(), os.path.join(args.save_dir, "controllers", "final_critic.pt"))
        print("\nTraining complete!")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
