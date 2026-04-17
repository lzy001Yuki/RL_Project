from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover
    cKDTree = None


@dataclass(frozen=True)
class UAVEnvParams:
    max_steps: int = 300
    success_radius: float = 5.0
    collision_threshold: float = 2.0
    action_limit: Tuple[float, float] = (2.0, 2.0)
    step_penalty: float = 0.5
    success_bonus: float = 200.0
    collision_penalty: float = 100.0


class UAVSubgoalEnv:
    """A lightweight 2D UAV navigation environment with explicit subgoals.

    Differences vs `_instructor/env/uav_env.py`:
      - No auto-reset in `step()`. You call `reset()` explicitly.
      - Observation includes both current subgoal and final goal vectors.
      - Nearest obstacle query uses SciPy cKDTree if available (fast), otherwise brute force.

    Observation (`sensor`, shape = (6,)):
      [0:2]  relative offset to nearest obstacle point (dx, dy)
      [2:4]  relative offset to current subgoal (dx, dy)
      [4:6]  relative offset to final goal (dx, dy)

    Action:
      2D continuous delta (dx, dy) clipped to action_limit (m/step).
    """

    def __init__(
        self,
        pointcloud_path: str,
        params: UAVEnvParams = UAVEnvParams(),
        device: Optional[torch.device] = None,
    ) -> None:
        self.params = params
        self.device = device or torch.device("cpu")

        self.max_steps = int(params.max_steps)
        self.success_radius = float(params.success_radius)
        self.collision_threshold = float(params.collision_threshold)
        self.action_limit = np.asarray(params.action_limit, dtype=np.float64)

        self._step_penalty = float(params.step_penalty)
        self._success_bonus = float(params.success_bonus)
        self._collision_penalty = float(params.collision_penalty)

        self.points_xy = np.load(pointcloud_path).astype(np.float64)
        if self.points_xy.ndim != 2 or self.points_xy.shape[1] != 2:
            raise ValueError("pointcloud must be an array of shape (N,2)")

        if cKDTree is not None:
            self._kdtree = cKDTree(self.points_xy)
        else:
            self._kdtree = None

        self.observation_shape = {"sensor": (6,)}
        self.action_shape = (2,)

        self.curr_pose: Optional[np.ndarray] = None  # [x, y, yaw]
        self.subgoal_xy: Optional[np.ndarray] = None
        self.final_goal_xy: Optional[np.ndarray] = None

        self.step_count = 0
        self.episode_reward = 0.0
        self._prev_subgoal_dist: Optional[float] = None

    def reset(
        self,
        initial_pose: np.ndarray,
        subgoal_xy: np.ndarray,
        final_goal_xy: Optional[np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        self.curr_pose = np.array(initial_pose, dtype=np.float64).copy()
        self.subgoal_xy = np.array(subgoal_xy, dtype=np.float64).copy()
        self.final_goal_xy = (
            np.array(final_goal_xy, dtype=np.float64).copy()
            if final_goal_xy is not None
            else self.subgoal_xy.copy()
        )

        self.step_count = 0
        self.episode_reward = 0.0
        self._prev_subgoal_dist = None

        obs = self._get_obs()
        sensor = obs["sensor"].detach().cpu().numpy()
        self._prev_subgoal_dist = float(np.linalg.norm(sensor[2:4]))
        return obs

    def set_subgoal(self, subgoal_xy: np.ndarray) -> None:
        if self.subgoal_xy is None:
            raise RuntimeError("reset() must be called before set_subgoal()")
        self.subgoal_xy = np.array(subgoal_xy, dtype=np.float64).copy()
        self._prev_subgoal_dist = None

    def step(self, action: torch.Tensor):
        if self.curr_pose is None or self.subgoal_xy is None or self.final_goal_xy is None:
            raise RuntimeError("reset() must be called before step()")

        a = action
        if isinstance(a, torch.Tensor):
            a = a.detach()
        if a.ndim == 2 and a.shape[0] == 1:
            a = a.squeeze(0)
        if a.ndim != 1 or a.shape[0] != 2:
            raise ValueError(f"action must have shape (2,) or (1,2), got {tuple(a.shape)}")

        dx = float(a[0].item())
        dy = float(a[1].item())
        dx = float(np.clip(dx, -self.action_limit[0], self.action_limit[0]))
        dy = float(np.clip(dy, -self.action_limit[1], self.action_limit[1]))

        x, y, _yaw = self.curr_pose
        new_x = x + dx
        new_y = y + dy
        new_yaw = float(math.atan2(dy, dx)) if (abs(dx) + abs(dy)) > 1e-9 else float(_yaw)
        self.curr_pose = np.array([new_x, new_y, new_yaw], dtype=np.float64)

        obs = self._get_obs()
        sensor = obs["sensor"].detach().cpu().numpy()

        nearest_obs_rel = sensor[0:2]
        subgoal_rel = sensor[2:4]
        obstacle_dist = float(np.linalg.norm(nearest_obs_rel))
        subgoal_dist = float(np.linalg.norm(subgoal_rel))

        done = False
        info: Dict[str, object] = {}

        # Termination
        if obstacle_dist <= self.collision_threshold:
            done = True
            info["won"] = False
            info["term_reason"] = "collision"
        elif subgoal_dist <= self.success_radius:
            done = True
            info["won"] = True
            info["term_reason"] = "success"

        self.step_count += 1
        if self.step_count >= self.max_steps and not done:
            done = True
            info["won"] = False
            info["term_reason"] = "timeout"

        # Reward
        reward = 0.0
        if self._prev_subgoal_dist is not None:
            reward += (self._prev_subgoal_dist - subgoal_dist)

        if obstacle_dist <= self.collision_threshold:
            reward -= self._collision_penalty

        if info.get("won", False):
            reward += self._success_bonus

        reward -= self._step_penalty

        self._prev_subgoal_dist = subgoal_dist
        self.episode_reward += reward

        if done:
            info["episode"] = {"r": float(self.episode_reward)}
            info["final_pose"] = self.curr_pose.copy()
            info["subgoal_xy"] = self.subgoal_xy.copy()
            info["final_goal_xy"] = self.final_goal_xy.copy()

        return obs, torch.tensor([reward], device=self.device), [done], [info]

    def _nearest_obstacle(self, query_xy: np.ndarray) -> Tuple[np.ndarray, float]:
        if self._kdtree is not None:
            dist, idx = self._kdtree.query(query_xy, k=1)
            nearest = self.points_xy[int(idx)]
            rel = nearest - query_xy
            return rel.astype(np.float32), float(dist)

        # Fallback: brute force (slow but dependency-free).
        diffs = self.points_xy - query_xy.reshape(1, 2)
        dists = np.linalg.norm(diffs, axis=1)
        j = int(np.argmin(dists))
        return diffs[j].astype(np.float32), float(dists[j])

    def _get_obs(self) -> Dict[str, torch.Tensor]:
        assert self.curr_pose is not None
        assert self.subgoal_xy is not None
        assert self.final_goal_xy is not None

        xy = np.array([self.curr_pose[0], self.curr_pose[1]], dtype=np.float64)

        nearest_rel, _dist = self._nearest_obstacle(xy)
        subgoal_rel = (self.subgoal_xy - xy).astype(np.float32)
        goal_rel = (self.final_goal_xy - xy).astype(np.float32)

        sensor = np.concatenate([nearest_rel, subgoal_rel, goal_rel], axis=0).astype(np.float32)
        return {"sensor": torch.tensor(sensor, device=self.device)}

