from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from algo.envs.uav_subgoal_env import UAVEnvParams, UAVSubgoalEnv
from algo.planning.grid_map import GridMap2D


@dataclass(frozen=True)
class MemoryObsConfig:
    """Config for external-memory-conditioned observations + reward shaping.

    The external memory is a per-initial 2D scalar field (same shape as the grid)
    that accumulates repulsion around previously used trajectories.

    Observation key `memory` (shape=(4,)):
      [0:2] repulsion vector (x,y) ~ negative memory gradient (clipped to [-1,1])
      [2]   local memory value normalized to [0,1]
      [3]   fade weight in [fade_w_min, 1] (smaller near goal)
    """

    # Normalize memory values by this scale (typically equals memory_max).
    memory_value_scale: float = 50.0
    # Clip gradient-derived repulsion magnitude before normalizing to [-1,1].
    repulse_clip: float = 10.0

    # Reward shaping: subtract `reward_weight * fade_w * mem_value_norm`.
    reward_weight: float = 2.0

    # Fade (repulsion weaker near goal to preserve success rate)
    fade_near_m: float = 60.0
    fade_far_frac: float = 0.8
    fade_w_min: float = 0.1


def _fade_weight(dist_to_goal_m: float, start_to_goal_m: float, near_m: float, far_frac: float, w_min: float) -> float:
    near_m = float(near_m)
    far_m = max(near_m + 1.0, float(far_frac) * float(start_to_goal_m))
    w = (float(dist_to_goal_m) - near_m) / (far_m - near_m)
    w = float(np.clip(w, 0.0, 1.0))
    w = float(w_min) + (1.0 - float(w_min)) * w
    return float(w)


def _memory_features_at_xy(
    grid: GridMap2D,
    memory_map: Optional[np.ndarray],
    xy: np.ndarray,
    goal_xy: np.ndarray,
    start_to_goal_m: float,
    cfg: MemoryObsConfig,
) -> Tuple[np.ndarray, float, float]:
    """Return (repulse_xy, mem_value_norm, fade_w)."""
    if memory_map is None:
        return np.zeros((2,), dtype=np.float32), 0.0, 1.0

    r, c = grid.world_to_grid(float(xy[0]), float(xy[1]))
    if not grid.in_bounds(r, c):
        dist_to_goal = float(np.linalg.norm(goal_xy - xy))
        fade_w = _fade_weight(dist_to_goal, start_to_goal_m, cfg.fade_near_m, cfg.fade_far_frac, cfg.fade_w_min)
        return np.zeros((2,), dtype=np.float32), 0.0, fade_w

    mem = float(memory_map[r, c])
    scale = float(cfg.memory_value_scale)
    if scale > 0.0:
        mem_norm = float(np.clip(mem / scale, 0.0, 1.0))
    else:
        mem_norm = float(mem)

    def _mem_at(rr: int, cc: int) -> float:
        if 0 <= rr < memory_map.shape[0] and 0 <= cc < memory_map.shape[1]:
            return float(memory_map[rr, cc])
        return float(memory_map[r, c])

    # Finite-difference gradient in grid coordinates: +col == +x, +row == +y.
    gx = 0.5 * (_mem_at(r, c + 1) - _mem_at(r, c - 1))
    gy = 0.5 * (_mem_at(r + 1, c) - _mem_at(r - 1, c))
    if grid.resolution > 0:
        gx = gx / float(grid.resolution)
        gy = gy / float(grid.resolution)

    repulse = np.array([-gx, -gy], dtype=np.float32)
    clip = float(cfg.repulse_clip)
    if clip > 0.0:
        repulse = np.clip(repulse, -clip, clip) / clip

    dist_to_goal = float(np.linalg.norm(goal_xy - xy))
    fade_w = _fade_weight(dist_to_goal, start_to_goal_m, cfg.fade_near_m, cfg.fade_far_frac, cfg.fade_w_min)
    return repulse.astype(np.float32), mem_norm, fade_w


def _find_nearest_free_cell(grid: GridMap2D, xy: np.ndarray, max_radius_cells: int) -> Tuple[int, int]:
    """Snap to a nearby free cell if a reset position is in collision/occupied."""
    r0, c0 = grid.world_to_grid(float(xy[0]), float(xy[1]))
    if grid.is_free(r0, c0):
        return r0, c0

    for rad in range(1, int(max_radius_cells) + 1):
        r_min = r0 - rad
        r_max = r0 + rad
        c_min = c0 - rad
        c_max = c0 + rad
        for r in range(r_min, r_max + 1):
            for c in (c_min, c_max):
                if grid.is_free(r, c):
                    return r, c
        for c in range(c_min + 1, c_max):
            for r in (r_min, r_max):
                if grid.is_free(r, c):
                    return r, c

    raise RuntimeError("Could not find a free grid cell near requested reset pose")


class UAVMemoryGoalEnv:
    """Goal-conditioned UAV env with an external memory-conditioned observation/reward.

    This wraps `UAVSubgoalEnv` by setting `subgoal_xy == final_goal_xy` and adds:
      - `memory` observation key (see `MemoryObsConfig`)
      - memory-shaped reward penalty (repulsion fades near goal)

    The external memory map is supplied by the caller on each reset() (typically
    from a per-initial memory bank).
    """

    def __init__(
        self,
        pointcloud_path: str,
        grid: GridMap2D,
        params: UAVEnvParams,
        memory_cfg: MemoryObsConfig = MemoryObsConfig(),
        latent_dim: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        self._base = UAVSubgoalEnv(pointcloud_path=pointcloud_path, params=params, device=device)
        self.grid = grid
        self.memory_cfg = memory_cfg
        self.latent_dim = int(latent_dim)
        self.device = self._base.device

        self.observation_shape: Dict[str, Tuple[int, ...]] = dict(self._base.observation_shape)
        self.observation_shape["memory"] = (4,)
        if self.latent_dim > 0:
            self.observation_shape["z"] = (self.latent_dim,)

        self.action_shape = self._base.action_shape
        self.action_limit = self._base.action_limit

        self.curr_pose: Optional[np.ndarray] = None
        self.goal_xy: Optional[np.ndarray] = None
        self._start_to_goal_m: float = 1.0
        self._memory_map: Optional[np.ndarray] = None
        self._z: Optional[np.ndarray] = None

    def reset(
        self,
        initial_pose: np.ndarray,
        goal_xy: np.ndarray,
        memory_map: Optional[np.ndarray] = None,
        snap_if_collision: bool = True,
        start_snap_radius_m: float = 10.0,
        latent_z: Optional[np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        goal_xy = np.asarray(goal_xy, dtype=np.float64).copy()

        # Optional: if the start is *actually* in collision region, snap to a nearby free grid cell.
        pose = np.array(initial_pose, dtype=np.float64).copy()
        if snap_if_collision:
            xy0 = pose[:2].copy()
            nearest_rel, obstacle_dist = self._base._nearest_obstacle(xy0)  # noqa: SLF001
            _ = nearest_rel
            if float(obstacle_dist) <= float(self._base.collision_threshold):
                max_cells = int(math.ceil(float(start_snap_radius_m) / float(self.grid.resolution)))
                r, c = _find_nearest_free_cell(self.grid, xy0, max_radius_cells=max_cells)
                sx, sy = self.grid.grid_to_world(r, c)
                pose[0] = float(sx)
                pose[1] = float(sy)

        self.curr_pose = pose
        self.goal_xy = goal_xy
        self._memory_map = memory_map
        self._start_to_goal_m = float(np.linalg.norm(goal_xy - pose[:2])) + 1e-6

        if self.latent_dim > 0:
            if latent_z is None:
                self._z = np.random.normal(size=(self.latent_dim,)).astype(np.float32)
            else:
                z = np.asarray(latent_z, dtype=np.float32).reshape(-1)
                if z.shape[0] != self.latent_dim:
                    raise ValueError(f"latent_z must have shape ({self.latent_dim},), got {tuple(z.shape)}")
                self._z = z.copy()
        else:
            self._z = None

        obs = self._base.reset(initial_pose=pose, subgoal_xy=goal_xy, final_goal_xy=goal_xy)
        obs = dict(obs)
        obs["memory"] = self._memory_obs()
        if self.latent_dim > 0 and self._z is not None:
            obs["z"] = torch.tensor(self._z, device=self.device)
        return obs

    def step(self, action: torch.Tensor):
        obs, reward, done, infos = self._base.step(action)
        self.curr_pose = self._base.curr_pose.copy() if self._base.curr_pose is not None else None

        obs = dict(obs)
        obs["memory"] = self._memory_obs()
        if self.latent_dim > 0 and self._z is not None:
            obs["z"] = torch.tensor(self._z, device=self.device)

        mem_penalty = float(self._memory_reward_penalty())
        if mem_penalty != 0.0:
            reward = reward - torch.tensor([mem_penalty], device=self.device, dtype=reward.dtype)
            # Keep episode return consistent with shaped reward for logging/debugging.
            self._base.episode_reward -= float(mem_penalty)
            if done[0] and len(infos) > 0 and isinstance(infos[0], dict) and "episode" in infos[0]:
                try:
                    infos[0]["episode"]["r"] = float(self._base.episode_reward)
                except Exception:
                    pass

        if len(infos) > 0 and isinstance(infos[0], dict):
            infos[0]["mem_penalty"] = float(mem_penalty)

        return obs, reward, done, infos

    def _memory_obs(self) -> torch.Tensor:
        if self.curr_pose is None or self.goal_xy is None:
            return torch.zeros((4,), device=self.device)
        xy = np.array([float(self.curr_pose[0]), float(self.curr_pose[1])], dtype=np.float64)
        repulse_xy, mem_norm, fade_w = _memory_features_at_xy(
            grid=self.grid,
            memory_map=self._memory_map,
            xy=xy,
            goal_xy=self.goal_xy,
            start_to_goal_m=self._start_to_goal_m,
            cfg=self.memory_cfg,
        )
        feat = np.array([repulse_xy[0], repulse_xy[1], float(mem_norm), float(fade_w)], dtype=np.float32)
        return torch.tensor(feat, device=self.device)

    def _memory_reward_penalty(self) -> float:
        if self.curr_pose is None or self.goal_xy is None or self._memory_map is None:
            return 0.0
        xy = np.array([float(self.curr_pose[0]), float(self.curr_pose[1])], dtype=np.float64)
        _repulse_xy, mem_norm, fade_w = _memory_features_at_xy(
            grid=self.grid,
            memory_map=self._memory_map,
            xy=xy,
            goal_xy=self.goal_xy,
            start_to_goal_m=self._start_to_goal_m,
            cfg=self.memory_cfg,
        )
        return float(self.memory_cfg.reward_weight) * float(fade_w) * float(mem_norm)
