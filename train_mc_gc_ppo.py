"""
MC-GC-PPO: Memory-Conditioned Goal-Conditioned PPO.

This script is designed for two stages:

  (A) Train + online collect:
      - Train a goal-conditioned PPO controller.
      - Maintain a per-initial external repulsion memory bank.
      - Optionally save every successful episode trajectory in baseline format
        while training (resume-friendly).

  (B) Stage-B collect only:
      - Load a trained policy.
      - For each initial, repeatedly rollout while updating external memory
        after each success, producing diverse successful trajectories.

Important:
  - Do NOT run training on this local machine (no env / no compute). This code is
    meant to be launched on your remote server (single GPU or torchrun multi-GPU).

Example (remote, multi-GPU training):
  torchrun --standalone --nproc_per_node=4 train_mc_gc_ppo.py \
    --mode train \
    --pointcloud_path data/pointcloud_2d.npy \
    --initials_path data/eval_initials_100.json \
    --save_dir saved_data/mc_gc_ppo_ddp

Example (stage-B collection, 20 initials × 100 trajs):
  torchrun --standalone --nproc_per_node=4 train_mc_gc_ppo.py \
    --mode collect \
    --pointcloud_path data/pointcloud_2d.npy \
    --initials_path data/eval_initials_100.json \
    --take_first_n 20 \
    --trajs_per_initial 100 \
    --output_dir submission/mc_gc_ppo \
    --policy_path saved_data/mc_gc_ppo_ddp/controllers/final_policy.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="train", choices=["train", "collect"])

    p.add_argument("--pointcloud_path", type=str, required=True)
    p.add_argument("--initials_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default=None)

    # Training config
    p.add_argument("--max_iter", type=int, default=30000)
    p.add_argument("--num_steps", type=int, default=256)
    p.add_argument("--max_steps", type=int, default=300)

    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--ppo_epoch", type=int, default=4)
    p.add_argument("--num_mini_batch", type=int, default=4)
    p.add_argument("--clip_param", type=float, default=0.1)
    p.add_argument("--entropy_coef", type=float, default=0.02)
    p.add_argument("--value_loss_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)

    # Env config (goal-conditioned)
    p.add_argument("--success_radius_m", type=float, default=30.0)
    p.add_argument("--collision_threshold_m", type=float, default=2.0)
    p.add_argument("--action_limit", type=float, nargs=2, default=(2.0, 2.0))
    p.add_argument("--step_penalty", type=float, default=0.5)
    p.add_argument("--success_bonus", type=float, default=200.0)
    p.add_argument("--collision_penalty", type=float, default=100.0)

    # Grid (for memory indexing)
    p.add_argument("--grid_resolution_m", type=float, default=2.0)
    p.add_argument("--grid_padding_m", type=float, default=5.0)
    p.add_argument("--grid_inflation_radius_m", type=float, default=4.0)
    p.add_argument("--start_snap_radius_m", type=float, default=10.0)

    # External memory update (per initial)
    p.add_argument("--memory_decay", type=float, default=0.99)
    p.add_argument("--memory_max", type=float, default=50.0)
    p.add_argument("--memory_radius_cells", type=int, default=2)
    p.add_argument("--memory_strength_success", type=float, default=2.0)
    p.add_argument("--memory_strength_failure", type=float, default=0.0)

    # Memory-conditioned obs + reward shaping
    p.add_argument("--memory_reward_weight", type=float, default=2.0)
    p.add_argument(
        "--memory_value_scale",
        type=float,
        default=None,
        help="Normalize memory values by this scale (default: memory_max).",
    )
    p.add_argument("--repulse_clip", type=float, default=10.0)
    p.add_argument("--fade_near_m", type=float, default=60.0)
    p.add_argument("--fade_far_frac", type=float, default=0.8)
    p.add_argument("--fade_w_min", type=float, default=0.1)

    # Optional latent-conditioning (helps multi-modal behaviors)
    p.add_argument("--latent_dim", type=int, default=0)

    # Common misc
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=5000)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--dist_backend",
        type=str,
        default=None,
        help="DDP backend (default: nccl if cuda else gloo)",
    )

    # Online collection during training (feature 1)
    p.add_argument("--collect_during_train", action="store_true")
    p.add_argument(
        "--collect_initials_path",
        type=str,
        default=None,
        help="Initials json for collection (default: initials_path).",
    )
    p.add_argument("--collect_take_first_n", type=int, default=20)
    p.add_argument("--collect_trajs_per_initial", type=int, default=100)
    p.add_argument(
        "--collect_output_dir",
        type=str,
        default=None,
        help="Baseline-format output dir (default: <save_dir>/baseline_trajs).",
    )
    p.add_argument("--stop_when_collected", action="store_true", help="Stop training once collection target is met.")

    # Stage-B collect-only (feature 2)
    p.add_argument("--policy_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="submission/mc_gc_ppo", help="Output dir in baseline_trajs format.")
    p.add_argument("--take_first_n", type=int, default=20)
    p.add_argument("--trajs_per_initial", type=int, default=100)
    p.add_argument("--max_attempts_per_traj", type=int, default=30)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


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


def _build_grid(
    pointcloud_path: str,
    initials: Sequence[dict],
    resolution_m: float,
    padding_m: float,
    inflation_radius_m: float,
):
    from algo.planning.grid_map import GridMap2D

    points = np.load(pointcloud_path).astype(np.float64)
    extra = []
    for init in initials:
        s, g = _extract_xy(init)
        extra.append(s)
        extra.append(g)
    extra_points_xy = np.stack(extra, axis=0) if len(extra) > 0 else None
    return GridMap2D.from_pointcloud(
        points_xy=points,
        resolution_m=float(resolution_m),
        padding_m=float(padding_m),
        inflation_radius_m=float(inflation_radius_m),
        extra_points_xy=extra_points_xy,
    )


def _rollout_one_episode(
    env,
    policy,
    device: torch.device,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    memory_map: np.ndarray,
    max_steps: int,
    deterministic: bool,
    start_snap_radius_m: float,
) -> Tuple[List[Tuple[float, float]], bool, Dict[str, object]]:
    _ = device
    obs = env.reset(
        initial_pose=np.array([float(start_xy[0]), float(start_xy[1]), 0.0], dtype=np.float64),
        goal_xy=goal_xy,
        memory_map=memory_map,
        snap_if_collision=True,
        start_snap_radius_m=float(start_snap_radius_m),
    )
    traj_xy: List[Tuple[float, float]] = [(float(env.curr_pose[0]), float(env.curr_pose[1]))]
    last_info: Dict[str, object] = {}
    won = False

    for _ in range(int(max_steps)):
        with torch.no_grad():
            act_out = policy.act(obs, deterministic=bool(deterministic))
        obs, _reward, done, infos = env.step(act_out.action)
        traj_xy.append((float(env.curr_pose[0]), float(env.curr_pose[1])))

        if done[0]:
            last_info = infos[0] if len(infos) > 0 else {}
            won = bool(last_info.get("won", False))
            break

    return traj_xy, won, last_info


def train(args: argparse.Namespace, device: torch.device) -> None:
    from algo.ppo.policy import GaussianActorCritic
    from algo.ppo.ppo import PPO
    from algo.ppo.storage import DictRolloutStorage
    from algo.common.distributed import all_reduce_tensor, barrier, broadcast_module, broadcast_object, get_world_size
    from algo.common.traj_io import ensure_at_least_two_points, next_traj_index, truncate_to_success, write_traj_txt
    from algo.envs.uav_subgoal_env import UAVEnvParams
    from algo.envs.uav_memory_goal_env import MemoryObsConfig, UAVMemoryGoalEnv
    from algo.memory.repulsion_memory import PerInitialRepulsionMemory, RepulsionMemoryConfig

    # Seeds:
    # - init_seed must be rank-invariant for identical initial weights.
    # - data_seed is rank-specific for env randomness (including latent z).
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
        print(f"Loaded {len(initials)} initials for training")

    grid = _build_grid(
        pointcloud_path=args.pointcloud_path,
        initials=initials,
        resolution_m=args.grid_resolution_m,
        padding_m=args.grid_padding_m,
        inflation_radius_m=args.grid_inflation_radius_m,
    )

    initial_ids = [_initial_id(init, i) for i, init in enumerate(initials)]
    mem_cfg = RepulsionMemoryConfig(
        decay=float(args.memory_decay),
        max_value=float(args.memory_max),
        radius_cells=int(args.memory_radius_cells),
        strength_success=float(args.memory_strength_success),
        strength_failure=float(args.memory_strength_failure),
    )
    memory = PerInitialRepulsionMemory(grid=grid, initial_ids=initial_ids, config=mem_cfg, seed=data_seed)

    value_scale = float(args.memory_value_scale) if args.memory_value_scale is not None else float(args.memory_max)
    obs_cfg = MemoryObsConfig(
        memory_value_scale=value_scale,
        repulse_clip=float(args.repulse_clip),
        reward_weight=float(args.memory_reward_weight),
        fade_near_m=float(args.fade_near_m),
        fade_far_frac=float(args.fade_far_frac),
        fade_w_min=float(args.fade_w_min),
    )

    env = UAVMemoryGoalEnv(
        pointcloud_path=args.pointcloud_path,
        grid=grid,
        params=UAVEnvParams(
            max_steps=int(args.max_steps),
            success_radius=float(args.success_radius_m),
            collision_threshold=float(args.collision_threshold_m),
            action_limit=(float(args.action_limit[0]), float(args.action_limit[1])),
            step_penalty=float(args.step_penalty),
            success_bonus=float(args.success_bonus),
            collision_penalty=float(args.collision_penalty),
        ),
        memory_cfg=obs_cfg,
        latent_dim=int(args.latent_dim),
        device=device,
    )

    obs_dim = int(sum(int(np.prod(shape)) for shape in env.observation_shape.values()))
    action_dim = int(np.prod(env.action_shape))
    action_limit = tuple(float(x) for x in env.action_limit.tolist())

    actor_critic = GaussianActorCritic(obs_dim=obs_dim, action_dim=action_dim, action_limit=action_limit).to(device)
    broadcast_module(actor_critic, src=0)
    barrier()

    # After model init sync, re-seed torch RNG per-rank to decorrelate exploration.
    torch.manual_seed(data_seed)
    torch.cuda.manual_seed_all(data_seed)

    agent = PPO(
        actor_critic=actor_critic,
        clip_param=float(args.clip_param),
        ppo_epoch=int(args.ppo_epoch),
        num_mini_batch=int(args.num_mini_batch),
        value_loss_coef=float(args.value_loss_coef),
        entropy_coef=float(args.entropy_coef),
        lr=float(args.lr),
        eps=1e-5,
        max_grad_norm=float(args.max_grad_norm),
    )

    # Save dir
    if args.save_dir is None:
        args.save_dir = os.path.join("saved_data", f"mc_gc_ppo_{int(time.time())}")
    args.save_dir = broadcast_object(args.save_dir, src=0)
    if _is_main_process(args):
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, "controllers"), exist_ok=True)
        with open(os.path.join(args.save_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        with open(os.path.join(args.save_dir, "mc_gc_ppo_config.json"), "w") as f:
            json.dump({"memory_obs": asdict(obs_cfg), "memory_update": asdict(mem_cfg)}, f, indent=2)
    barrier()

    # Optional online trajectory collection during training (rank0 only).
    collect_ids: Optional[set[int]] = None
    saved_counts: Dict[int, int] = {}
    collect_output_dir: Optional[str] = None
    if bool(args.collect_during_train):
        collect_path = args.collect_initials_path or args.initials_path
        collect_initials = _load_initials(collect_path, take_first_n=int(args.collect_take_first_n))
        collect_ids = {_initial_id(init, i) for i, init in enumerate(collect_initials)}
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
                    },
                    f,
                    indent=2,
                )
        barrier()

    def _done_collecting() -> bool:
        if not bool(args.collect_during_train):
            return False
        if collect_ids is None:
            return False
        target = int(args.collect_trajs_per_initial)
        if target <= 0:
            return False
        return all(int(saved_counts.get(iid, 0)) >= target for iid in collect_ids)

    rollouts = DictRolloutStorage(
        num_steps=int(args.num_steps),
        num_envs=1,
        obs_shapes=env.observation_shape,
        action_shape=env.action_shape,
        recurrent_hidden_state_size=actor_critic.recurrent_hidden_state_size,
    )

    rng = np.random.default_rng(int(data_seed))
    current_idx = int(rng.integers(0, len(initials)))
    current_init = initials[current_idx]
    current_iid = int(initial_ids[current_idx])
    start_xy, goal_xy = _extract_xy(current_init)
    obs = env.reset(
        initial_pose=np.array([float(start_xy[0]), float(start_xy[1]), 0.0], dtype=np.float64),
        goal_xy=goal_xy,
        memory_map=memory.get(current_iid),
        snap_if_collision=True,
        start_snap_radius_m=float(args.start_snap_radius_m),
    )
    for k in obs:
        rollouts.obs[k][0].copy_(obs[k])
    rollouts.to(device)

    traj_xy: List[Tuple[float, float]] = [(float(env.curr_pose[0]), float(env.curr_pose[1]))]
    episode_rewards = deque(maxlen=50)
    start_time = time.time()

    if _is_main_process(args):
        print(f"\nStarting MC-GC-PPO training (max_iter={args.max_iter}, world_size={getattr(args,'world_size',1)})...\n")

    for it in range(int(args.max_iter)):
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

        for step in range(int(args.num_steps)):
            with torch.no_grad():
                obs_step = {k: rollouts.obs[k][step] for k in rollouts.obs}
                act_out = actor_critic.act(obs_step)

            next_obs, reward, done, infos = env.step(act_out.action)
            traj_xy.append((float(env.curr_pose[0]), float(env.curr_pose[1])))

            if len(infos) > 0 and isinstance(infos[0], dict) and "episode" in infos[0]:
                episode_rewards.append(float(infos[0]["episode"]["r"]))

            if done[0]:
                info0 = infos[0] if len(infos) > 0 else {}
                won = bool(info0.get("won", False))

                memory.update_with_trajectory(current_iid, traj_xy, success=won)

                if (
                    bool(args.collect_during_train)
                    and _is_main_process(args)
                    and collect_ids is not None
                    and collect_output_dir is not None
                    and (current_iid in collect_ids)
                    and won
                ):
                    k = int(saved_counts.get(current_iid, 0))
                    if k < int(args.collect_trajs_per_initial):
                        out_path = os.path.join(collect_output_dir, f"initial_{current_iid}", f"traj_{k}.txt")
                        trimmed = truncate_to_success(traj_xy, goal_xy, success_radius_m=float(args.success_radius_m))
                        trimmed = ensure_at_least_two_points(trimmed)
                        write_traj_txt(out_path, trimmed)
                        saved_counts[current_iid] = k + 1

                current_idx = int(rng.integers(0, len(initials)))
                current_init = initials[current_idx]
                current_iid = int(initial_ids[current_idx])
                start_xy, goal_xy = _extract_xy(current_init)
                next_obs = env.reset(
                    initial_pose=np.array([float(start_xy[0]), float(start_xy[1]), 0.0], dtype=np.float64),
                    goal_xy=goal_xy,
                    memory_map=memory.get(current_iid),
                    snap_if_collision=True,
                    start_snap_radius_m=float(args.start_snap_radius_m),
                )
                traj_xy = [(float(env.curr_pose[0]), float(env.curr_pose[1]))]

            masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done]).to(device)
            bad_masks = torch.ones_like(masks)
            rhs = torch.zeros(1, actor_critic.recurrent_hidden_state_size, device=device)
            rollouts.insert(
                obs=next_obs,
                recurrent_hidden_states=rhs,
                actions=act_out.action,
                action_log_probs=act_out.action_log_prob,
                value_preds=act_out.value,
                rewards=reward,
                masks=masks,
                bad_masks=bad_masks,
            )

        with torch.no_grad():
            next_value = actor_critic.get_value({k: rollouts.obs[k][-1] for k in rollouts.obs}).detach()
        rollouts.compute_returns(next_value, use_gae=True, gamma=float(args.gamma), gae_lambda=float(args.gae_lambda))
        stats = agent.update(rollouts)
        rollouts.after_update()

        if it % int(args.log_interval) == 0 and len(episode_rewards) > 0 and _is_main_process(args):
            elapsed = time.time() - start_time
            total_steps = (it + 1) * int(args.num_steps) * int(getattr(args, "world_size", 1))
            extra = ""
            if bool(args.collect_during_train) and collect_ids is not None:
                extra = f"  collect={sum(saved_counts.get(iid, 0) for iid in collect_ids)}"
            print(
                f"[Iter {it:6d}] steps={total_steps:9d}  "
                f"reward={np.mean(episode_rewards):7.1f}  "
                f"v_loss={stats.value_loss:.4f}{extra}  "
                f"elapsed={elapsed:.0f}s"
            )
            with open(os.path.join(args.save_dir, "train_log.txt"), "a") as f:
                f.write(
                    f"{it}\t{float(np.mean(episode_rewards)):.4f}\t"
                    f"{stats.value_loss:.6f}\t{stats.action_loss:.6f}\t{stats.dist_entropy:.6f}\n"
                )

        if it % int(args.save_interval) == 0 and it > 0 and _is_main_process(args):
            torch.save(actor_critic.state_dict(), os.path.join(args.save_dir, "controllers", f"{it}_policy.pt"))

    if _is_main_process(args):
        torch.save(actor_critic.state_dict(), os.path.join(args.save_dir, "controllers", "final_policy.pt"))
        print("\nTraining complete!")
        if bool(args.collect_during_train) and collect_ids is not None:
            print(f"Collected counts (by initial_id): { {iid: saved_counts.get(iid, 0) for iid in sorted(collect_ids)} }")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def collect(args: argparse.Namespace, device: torch.device) -> None:
    from algo.ppo.policy import GaussianActorCritic
    from algo.common.distributed import barrier, broadcast_object
    from algo.common.traj_io import (
        ensure_at_least_two_points,
        ensure_dir,
        next_traj_index,
        reaches_target,
        truncate_to_success,
        write_traj_txt,
    )
    from algo.envs.uav_subgoal_env import UAVEnvParams
    from algo.envs.uav_memory_goal_env import MemoryObsConfig, UAVMemoryGoalEnv
    from algo.memory.repulsion_memory import PerInitialRepulsionMemory, RepulsionMemoryConfig

    if args.policy_path is None:
        raise ValueError("--policy_path is required in --mode collect")

    rank = int(getattr(args, "rank", 0))
    data_seed = int(args.seed) + rank * 1000
    np.random.seed(data_seed)
    random.seed(data_seed)
    torch.manual_seed(data_seed)
    torch.cuda.manual_seed_all(data_seed)
    torch.set_num_threads(1)

    initials = _load_initials(args.initials_path, take_first_n=int(args.take_first_n))
    all_ids = [_initial_id(init, i) for i, init in enumerate(initials)]

    world_size = int(getattr(args, "world_size", 1))
    my_pairs: List[Tuple[int, dict]] = []
    for idx, init in enumerate(initials):
        if (idx % world_size) == rank:
            my_pairs.append((int(all_ids[idx]), init))

    if _is_main_process(args):
        print(f"Collect-only: {len(initials)} initials total, world_size={world_size}")
    print(f"[rank{rank}] Assigned initials: {len(my_pairs)}")

    grid = _build_grid(
        pointcloud_path=args.pointcloud_path,
        initials=initials,
        resolution_m=args.grid_resolution_m,
        padding_m=args.grid_padding_m,
        inflation_radius_m=args.grid_inflation_radius_m,
    )

    mem_cfg = RepulsionMemoryConfig(
        decay=float(args.memory_decay),
        max_value=float(args.memory_max),
        radius_cells=int(args.memory_radius_cells),
        strength_success=float(args.memory_strength_success),
        strength_failure=float(args.memory_strength_failure),
    )
    memory = PerInitialRepulsionMemory(
        grid=grid,
        initial_ids=[iid for iid, _ in my_pairs],
        config=mem_cfg,
        seed=data_seed,
    )

    value_scale = float(args.memory_value_scale) if args.memory_value_scale is not None else float(args.memory_max)
    obs_cfg = MemoryObsConfig(
        memory_value_scale=value_scale,
        repulse_clip=float(args.repulse_clip),
        reward_weight=float(args.memory_reward_weight),
        fade_near_m=float(args.fade_near_m),
        fade_far_frac=float(args.fade_far_frac),
        fade_w_min=float(args.fade_w_min),
    )

    env = UAVMemoryGoalEnv(
        pointcloud_path=args.pointcloud_path,
        grid=grid,
        params=UAVEnvParams(
            max_steps=int(args.max_steps),
            success_radius=float(args.success_radius_m),
            collision_threshold=float(args.collision_threshold_m),
            action_limit=(float(args.action_limit[0]), float(args.action_limit[1])),
            step_penalty=float(args.step_penalty),
            success_bonus=float(args.success_bonus),
            collision_penalty=float(args.collision_penalty),
        ),
        memory_cfg=obs_cfg,
        latent_dim=int(args.latent_dim),
        device=device,
    )

    obs_dim = int(sum(int(np.prod(shape)) for shape in env.observation_shape.values()))
    action_dim = int(np.prod(env.action_shape))
    action_limit = tuple(float(x) for x in env.action_limit.tolist())
    policy = GaussianActorCritic(obs_dim=obs_dim, action_dim=action_dim, action_limit=action_limit).to(device)
    policy.load_state_dict(torch.load(args.policy_path, map_location=device))
    policy.eval()

    args.output_dir = broadcast_object(str(args.output_dir), src=0)
    if _is_main_process(args):
        if bool(args.overwrite) and os.path.exists(args.output_dir):
            names = []
            try:
                names = os.listdir(args.output_dir)
            except Exception:
                names = []
            if len(names) > 0 and not any(n.startswith("initial_") for n in names):
                raise RuntimeError(f"--overwrite refused: {args.output_dir} does not look like an output folder")
            import shutil

            shutil.rmtree(args.output_dir)
        ensure_dir(args.output_dir)
        with open(os.path.join(args.output_dir, "collect_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    barrier()

    for iid, init in my_pairs:
        start_xy, goal_xy = _extract_xy(init)
        init_dir = os.path.join(args.output_dir, f"initial_{iid}")
        ensure_dir(init_dir)

        saved_k = int(next_traj_index(init_dir))
        target_k = int(args.trajs_per_initial)
        if saved_k >= target_k:
            print(f"[rank{rank}] initial_{iid}: already has {saved_k} trajs, skipping")
            continue

        memory.reset(iid)
        if saved_k > 0:
            existing = []
            try:
                existing = os.listdir(init_dir)
            except Exception:
                existing = []
            indices = []
            for name in existing:
                if not (name.startswith("traj_") and name.endswith(".txt")):
                    continue
                try:
                    indices.append(int(name[len("traj_") : -len(".txt")]))
                except Exception:
                    continue
            for j in sorted(i for i in indices if 0 <= i < saved_k):
                path = os.path.join(init_dir, f"traj_{j}.txt")
                arr = np.loadtxt(path)
                if arr.ndim == 1:
                    arr = arr.reshape(1, 2)
                if arr.ndim != 2 or arr.shape[1] != 2:
                    raise RuntimeError(f"[rank{rank}] initial_{iid}: invalid existing trajectory file: {path}")
                traj = [(float(x), float(y)) for x, y in arr.tolist()]
                traj = ensure_at_least_two_points(traj)
                memory.update_with_trajectory(iid, traj, success=True)

        for k in range(saved_k, target_k):
            ok = False
            for attempt in range(int(args.max_attempts_per_traj)):
                traj_xy, won, info0 = _rollout_one_episode(
                    env=env,
                    policy=policy,
                    device=device,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                    memory_map=memory.get(iid),
                    max_steps=int(args.max_steps),
                    deterministic=bool(args.deterministic),
                    start_snap_radius_m=float(args.start_snap_radius_m),
                )
                _ = attempt

                if won:
                    trimmed = truncate_to_success(traj_xy, goal_xy, success_radius_m=float(args.success_radius_m))
                    if not reaches_target(trimmed, goal_xy, success_radius_m=float(args.success_radius_m)):
                        continue
                    trimmed = ensure_at_least_two_points(trimmed)
                    out_path = os.path.join(init_dir, f"traj_{k}.txt")
                    write_traj_txt(out_path, trimmed)
                    memory.update_with_trajectory(iid, trimmed, success=True)
                    ok = True
                    break

                memory.update_with_trajectory(iid, traj_xy, success=False)
                _ = info0

            if not ok:
                raise RuntimeError(
                    f"[rank{rank}] initial_{iid}: failed to collect traj_{k} "
                    f"after {args.max_attempts_per_traj} attempts"
                )

        print(f"[rank{rank}] initial_{iid}: saved {target_k} trajectories to {init_dir}")

    barrier()
    if _is_main_process(args):
        print("\nCollect-only complete.")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main() -> None:
    args = parse_args()
    device = _init_distributed(args)
    if args.mode == "train":
        train(args, device=device)
    else:
        collect(args, device=device)


if __name__ == "__main__":
    main()

