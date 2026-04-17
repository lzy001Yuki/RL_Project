from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algo.common.torch_utils import hard_update, soft_update
from algo.common.distributed import all_reduce_gradients, all_reduce_tensor, get_world_size
from .networks import SACActor, SACCritic
from .replay_buffer import DictReplayBuffer, ReplayBatch


@dataclass
class SACStats:
    critic_loss: float
    actor_loss: float
    alpha_loss: float
    alpha: float


class SAC:
    """Soft Actor-Critic for continuous control with dict observations."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_limit: Tuple[float, float],
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        target_entropy: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.gamma = float(gamma)
        self.tau = float(tau)

        self.actor = SACActor(obs_dim, action_dim, action_limit).to(self.device)
        self.critic = SACCritic(obs_dim, action_dim).to(self.device)
        self.critic_target = SACCritic(obs_dim, action_dim).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Automatic entropy tuning.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = float(target_entropy) if target_entropy is not None else -float(action_dim)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False) -> torch.Tensor:
        self.actor.eval()
        action = self.actor.act(obs, deterministic=deterministic)
        self.actor.train()
        return action

    def update(self, batch: ReplayBatch) -> SACStats:
        obs = batch.obs
        actions = batch.actions
        rewards = batch.rewards
        next_obs = batch.next_obs
        dones = batch.dones

        # ------------------------------------------------------------
        # Critic update
        # ------------------------------------------------------------
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_prob
            target_q = rewards + (1.0 - dones) * self.gamma * q_next

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if get_world_size() > 1:
            all_reduce_gradients(self.critic.parameters(), average=True)
        self.critic_opt.step()

        # ------------------------------------------------------------
        # Actor update
        # ------------------------------------------------------------
        new_actions, log_prob = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, new_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if get_world_size() > 1:
            all_reduce_gradients(self.actor.parameters(), average=True)
        self.actor_opt.step()

        # ------------------------------------------------------------
        # Alpha update
        # ------------------------------------------------------------
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        if get_world_size() > 1 and self.log_alpha.grad is not None:
            all_reduce_tensor(self.log_alpha.grad, average=True)
        self.alpha_opt.step()

        # ------------------------------------------------------------
        # Target update
        # ------------------------------------------------------------
        soft_update(self.critic_target, self.critic, tau=self.tau)

        return SACStats(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            alpha_loss=float(alpha_loss.item()),
            alpha=float(self.alpha.item()),
        )
