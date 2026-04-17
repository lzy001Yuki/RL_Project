from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np
import torch


@dataclass
class ReplayBatch:
    obs: Dict[str, torch.Tensor]
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: Dict[str, torch.Tensor]
    dones: torch.Tensor


class DictReplayBuffer:
    """A minimal replay buffer for dict observations (CPU-backed numpy arrays)."""

    def __init__(
        self,
        capacity: int,
        obs_shapes: Mapping[str, Tuple[int, ...]],
        action_dim: int,
    ) -> None:
        self.capacity = int(capacity)
        self.action_dim = int(action_dim)

        self.obs_keys = list(obs_shapes.keys())
        self.obs = {k: np.zeros((self.capacity, *obs_shapes[k]), dtype=np.float32) for k in self.obs_keys}
        self.next_obs = {k: np.zeros((self.capacity, *obs_shapes[k]), dtype=np.float32) for k in self.obs_keys}

        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self.size = 0
        self.ptr = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: Mapping[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_obs: Mapping[str, np.ndarray],
        done: bool,
    ) -> None:
        index = self.ptr

        for key in self.obs_keys:
            self.obs[key][index] = np.asarray(obs[key], dtype=np.float32)
            self.next_obs[key][index] = np.asarray(next_obs[key], dtype=np.float32)

        self.actions[index] = np.asarray(action, dtype=np.float32)
        self.rewards[index, 0] = float(reward)
        self.dones[index, 0] = 1.0 if done else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        batch_size = min(int(batch_size), self.size)
        indices = np.random.randint(0, self.size, size=batch_size)

        obs = {k: torch.from_numpy(self.obs[k][indices]).to(device) for k in self.obs_keys}
        next_obs = {k: torch.from_numpy(self.next_obs[k][indices]).to(device) for k in self.obs_keys}
        actions = torch.from_numpy(self.actions[indices]).to(device)
        rewards = torch.from_numpy(self.rewards[indices]).to(device)
        dones = torch.from_numpy(self.dones[indices]).to(device)
        return ReplayBatch(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones)

