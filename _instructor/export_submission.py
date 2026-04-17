"""
Export trajectories in the required submission format:

  submission/
    initial_0/
      traj_0.txt
      traj_1.txt
      ...

Each traj file contains one "x y" per line (no headers).

This script runs a trained controller in UAVNavEnv (auto_reset=False) and
repeats rollouts until it collects enough *successful* trajectories per
initial.

Usage:
  cd _instructor
  python export_submission.py \
    --pointcloud_path ../data/pointcloud_2d.npy \
    --initials_path ../data/eval_initials_20.json \
    --controller_path saved_data/run_x/controllers/final_controller.pt \
    --out_dir ../submission \
    --trajs_per_initial 20
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import torch

from env.uav_env import UAVNavEnv
from model import Policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pointcloud_path", type=str, required=True)
    p.add_argument("--initials_path", type=str, required=True)
    p.add_argument("--controller_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--trajs_per_initial", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--success_radius", type=float, default=30.0)
    p.add_argument("--collision_threshold", type=float, default=2.0)
    p.add_argument("--action_limit", type=float, nargs=2, default=(2.0, 2.0))
    p.add_argument("--deterministic", action="store_true", help="Use mean action (less diversity)")
    p.add_argument(
        "--action_noise_std",
        type=float,
        default=0.0,
        help="Extra Gaussian noise added to actions at export time (for diversity).",
    )
    p.add_argument("--max_attempts_per_traj", type=int, default=50)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _rollout_one(
    env: UAVNavEnv,
    policy: Policy,
    *,
    initial_pose: np.ndarray,
    target_center: np.ndarray,
    max_steps: int,
    deterministic: bool,
    action_limit_t: torch.Tensor,
    action_noise_std: float,
) -> tuple[bool, np.ndarray]:
    obs = env.reset(initial_pose=initial_pose, target_center=target_center)
    traj: list[list[float]] = [[float(initial_pose[0]), float(initial_pose[1])]]

    for _ in range(max_steps):
        obs_batch = {"sensor": obs["sensor"].unsqueeze(0)}
        with torch.no_grad():
            _, action, _ = policy.act(obs_batch, deterministic=deterministic)

        action = action.squeeze(0)
        if action_noise_std > 0:
            action = action + torch.randn_like(action) * float(action_noise_std)

        action = torch.clamp(action, min=-action_limit_t, max=action_limit_t)

        obs, _, done, infos = env.step(action.unsqueeze(0))
        traj.append([float(env.curr_pose[0]), float(env.curr_pose[1])])

        if done[0]:
            return bool(infos[0].get("won", False)), np.asarray(traj, dtype=np.float64)

    return False, np.asarray(traj, dtype=np.float64)


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    with open(args.initials_path) as f:
        initials: list[dict[str, Any]] = json.load(f)
    print(f"Loaded {len(initials)} initials from {args.initials_path}")

    env_params = {
        "max_steps": args.max_steps,
        "success_radius": float(args.success_radius),
        "collision_threshold": float(args.collision_threshold),
        "action_limit": [float(args.action_limit[0]), float(args.action_limit[1])],
    }
    env = UAVNavEnv(
        pointcloud_path=args.pointcloud_path,
        env_params=env_params,
        save_dir=None,
        device=device,
        initials=initials,
        auto_reset=False,
    )

    policy = Policy(obs_dim=4, action_dim=2, action_limit=tuple(args.action_limit)).to(device)
    ckpt = torch.load(args.controller_path, map_location=device)
    policy.load_state_dict(ckpt, strict=True)
    policy.eval()
    print(f"Loaded controller from {args.controller_path} on {device}")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    action_limit = np.asarray(args.action_limit, dtype=np.float32)
    action_limit_t = torch.tensor(action_limit, dtype=torch.float32, device=device)

    total_written = 0
    for init in initials:
        iid = int(init["initial_id"])
        init_dir = os.path.join(out_dir, f"initial_{iid}")
        os.makedirs(init_dir, exist_ok=True)

        initial_pose = np.array([init["x_start"], init["y_start"], 0.0], dtype=np.float64)
        target_center = np.array([init["target_center_x"], init["target_center_y"]], dtype=np.float64)

        written = 0
        attempts = 0
        while written < args.trajs_per_initial:
            if attempts >= args.trajs_per_initial * args.max_attempts_per_traj:
                raise RuntimeError(
                    f"Too many failed attempts for initial_id={iid}: "
                    f"written={written}/{args.trajs_per_initial}"
                )

            # Make rollouts reproducible but different across (initial, traj, attempt).
            rollout_seed = args.seed + iid * 100000 + written * 1000 + attempts
            _set_seed(rollout_seed)

            ok, traj = _rollout_one(
                env,
                policy,
                initial_pose=initial_pose,
                target_center=target_center,
                max_steps=args.max_steps,
                deterministic=args.deterministic,
                action_limit_t=action_limit_t,
                action_noise_std=float(args.action_noise_std),
            )
            attempts += 1
            if not ok:
                continue

            traj_path = os.path.join(init_dir, f"traj_{written}.txt")
            with open(traj_path, "w") as f:
                for x, y in traj:
                    f.write(f"{x} {y}\n")

            written += 1
            total_written += 1

        print(f"initial_{iid}: wrote {written} trajectories (attempts={attempts})")

    print(f"Done. Wrote {total_written} trajectories to {out_dir}")


if __name__ == "__main__":
    main()
