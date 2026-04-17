from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np
import torch

from env.uav_env import UAVNavEnv


def _obs_to_numpy(obs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k, v in obs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
        else:
            out[k] = np.asarray(v)
    return out


def _worker(remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env_kwargs: dict[str, Any]):
    parent_remote.close()
    env = UAVNavEnv(**env_kwargs)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                obs = env.reset(initial_pose=data["initial_pose"], target_center=data["target_center"])
                remote.send(_obs_to_numpy(obs))
            elif cmd == "step":
                action_np = np.asarray(data, dtype=np.float32).reshape(1, 2)
                action = torch.from_numpy(action_np)
                obs, reward, done, infos = env.step(action)
                reward_f = float(reward.squeeze().item())
                done_b = bool(done[0])
                info = infos[0]
                remote.send((_obs_to_numpy(obs), reward_f, done_b, info))
            elif cmd == "close":
                remote.close()
                break
            else:
                raise RuntimeError(f"Unknown cmd: {cmd}")
    finally:
        try:
            remote.close()
        except Exception:
            pass


class UAVSubprocVecEnv:
    """
    A minimal sub-process vectorized env for UAVNavEnv.

    - Each worker runs its own UAVNavEnv on CPU.
    - Observations are returned as torch tensors on the main process device.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        pointcloud_path: str,
        env_params: dict[str, Any],
        initials: list[dict[str, Any]],
        device: torch.device,
        save_dir: str | None = None,
    ):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.device = device
        self.observation_shape = {"sensor": (4,)}
        self.action_shape = (2,)

        # Use spawn to avoid CUDA+fork issues when the main process uses GPU.
        ctx = mp.get_context("spawn")

        self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(self.num_envs)])
        self.processes: list[mp.Process] = []
        for wr, r in zip(self.work_remotes, self.remotes):
            env_kwargs = dict(
                pointcloud_path=pointcloud_path,
                env_params=env_params,
                save_dir=save_dir,
                device=torch.device("cpu"),
                initials=initials,
            )
            p = ctx.Process(target=_worker, args=(wr, r, env_kwargs), daemon=True)
            p.start()
            wr.close()
            self.processes.append(p)

    def reset(self, initial_poses: np.ndarray, target_centers: np.ndarray) -> dict[str, torch.Tensor]:
        initial_poses = np.asarray(initial_poses, dtype=np.float64)
        target_centers = np.asarray(target_centers, dtype=np.float64)
        if initial_poses.shape != (self.num_envs, 3):
            raise ValueError(f"initial_poses must be (num_envs, 3), got {initial_poses.shape}")
        if target_centers.shape != (self.num_envs, 2):
            raise ValueError(f"target_centers must be (num_envs, 2), got {target_centers.shape}")

        for remote, ip, tc in zip(self.remotes, initial_poses, target_centers):
            remote.send(("reset", {"initial_pose": ip, "target_center": tc}))

        obs_list = [remote.recv() for remote in self.remotes]
        sensor = np.stack([o["sensor"].reshape(-1) for o in obs_list], axis=0).astype(np.float32)
        return {"sensor": torch.tensor(sensor, device=self.device)}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[bool], list[dict[str, Any]]]:
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32)
        actions_np = actions.detach().cpu().numpy().astype(np.float32)
        if actions_np.shape != (self.num_envs, 2):
            raise ValueError(f"actions must be (num_envs, 2), got {actions_np.shape}")

        for remote, a in zip(self.remotes, actions_np):
            remote.send(("step", a))

        results = [remote.recv() for remote in self.remotes]
        obs_list, rewards, dones, infos = zip(*results)
        sensor = np.stack([o["sensor"].reshape(-1) for o in obs_list], axis=0).astype(np.float32)
        obs = {"sensor": torch.tensor(sensor, device=self.device)}
        rew = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        done_list = list(dones)
        info_list = list(infos)
        return obs, rew, done_list, info_list

    def close(self) -> None:
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for p in self.processes:
            try:
                p.join(timeout=1.0)
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

