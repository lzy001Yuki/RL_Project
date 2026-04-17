from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import torch
import torch.nn as nn

from algo.common.tanh_normal import sample_tanh_normal, tanh_normal_log_prob
from algo.common.torch_utils import MLP, flatten_obs


class SACActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_limit: Tuple[float, float],
        hidden_dims=(256, 256),
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.register_buffer("action_scale", torch.tensor(action_limit, dtype=torch.float32).unsqueeze(0))

        self.trunk = MLP(obs_dim, hidden_dims, hidden_dims[-1], activation=nn.ReLU())
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(self, obs: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(flatten_obs(obs))
        mean = self.mean_head(feat)
        log_std = self.log_std_head(feat)
        return mean, log_std

    def sample(self, obs: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        sample = sample_tanh_normal(mean, log_std)
        # Scale to environment action range; see notes in SAC trainer on log_prob.
        action_env = sample.action * self.action_scale.to(sample.action.device)
        # Include constant scaling term so auto-alpha matches env-space entropy.
        log_det_scale = torch.log(self.action_scale.to(sample.action.device)).sum(dim=-1, keepdim=True)
        log_prob_env = sample.log_prob - log_det_scale
        return action_env, log_prob_env

    def act(self, obs: Mapping[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        if deterministic:
            action = torch.tanh(mean)
        else:
            action = sample_tanh_normal(mean, log_std).action
        return action * self.action_scale.to(action.device)

    def log_prob(self, obs: Mapping[str, torch.Tensor], action_env: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        action = action_env / self.action_scale.to(action_env.device)
        log_prob = tanh_normal_log_prob(mean, log_std, action)
        log_det_scale = torch.log(self.action_scale.to(action_env.device)).sum(dim=-1, keepdim=True)
        return log_prob - log_det_scale


class SACCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims=(256, 256),
    ) -> None:
        super().__init__()
        self.q1 = MLP(obs_dim + action_dim, hidden_dims, 1, activation=nn.ReLU())
        self.q2 = MLP(obs_dim + action_dim, hidden_dims, 1, activation=nn.ReLU())

    def forward(self, obs: Mapping[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([flatten_obs(obs), action], dim=-1)
        return self.q1(x), self.q2(x)

