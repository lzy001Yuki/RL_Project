from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from algo.common.torch_utils import MLP, flatten_obs


@dataclass
class ActOutput:
    value: torch.Tensor
    action: torch.Tensor
    action_log_prob: torch.Tensor


class GaussianActorCritic(nn.Module):
    """Simple actor-critic for continuous actions with tanh squashing."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_limit: Tuple[float, float],
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.register_buffer("action_scale", torch.tensor(action_limit, dtype=torch.float32).unsqueeze(0))

        self.backbone = MLP(obs_dim, [hidden_dim, hidden_dim, hidden_dim], hidden_dim, activation=nn.LeakyReLU(0.1))
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.constant_(self.actor_logstd, -0.5)

    def _features(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        flat = flatten_obs(obs)
        return self.backbone(flat)

    def act(self, obs: Mapping[str, torch.Tensor], deterministic: bool = False) -> ActOutput:
        feat = self._features(obs)
        mean = self.actor_mean(feat)
        std = self.actor_logstd.expand_as(mean).exp()

        dist = Normal(mean, std)
        raw_action = mean if deterministic else dist.sample()
        scaled_action = torch.tanh(raw_action) * self.action_scale.to(raw_action.device)
        value = self.critic(feat)

        action_log_prob = dist.log_prob(raw_action).sum(-1, keepdim=True)
        return ActOutput(value=value, action=scaled_action, action_log_prob=action_log_prob)

    def evaluate_actions(
        self,
        obs: Mapping[str, torch.Tensor],
        _rhs: Optional[torch.Tensor] = None,
        _masks: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ):
        if actions is None:
            raise ValueError("actions is required")

        feat = self._features(obs)
        mean = self.actor_mean(feat)
        std = self.actor_logstd.expand_as(mean).exp()

        # Invert scaling: a_scaled = tanh(a_raw) * scale.
        eps = 1e-6
        actions_unscaled = torch.clamp(
            actions / self.action_scale.to(actions.device), -1.0 + eps, 1.0 - eps
        )
        raw_action = torch.atanh(actions_unscaled)

        dist = Normal(mean, std)
        log_probs = dist.log_prob(raw_action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1).mean()
        value = self.critic(feat)
        return value, log_probs, entropy, None

    def get_value(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        feat = self._features(obs)
        return self.critic(feat)

    @property
    def is_recurrent(self) -> bool:
        return False

    @property
    def recurrent_hidden_state_size(self) -> int:
        return 1

