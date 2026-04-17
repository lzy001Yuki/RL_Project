from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class DictRolloutStorage:
    """Rollout buffer that stores dict-based observations.

    This is a lightly refactored version of the instructor baseline storage,
    kept intentionally simple and framework-free.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shapes: Mapping[str, Tuple[int, ...]],
        action_shape: Tuple[int, ...],
        recurrent_hidden_state_size: int,
    ) -> None:
        if not isinstance(obs_shapes, dict):
            raise TypeError("obs_shapes must be a dict")

        self.obs: Dict[str, torch.Tensor] = {}
        self.obs_keys = []
        for key, shape in obs_shapes.items():
            self.obs[key] = torch.zeros(num_steps + 1, num_envs, *shape)
            self.obs_keys.append(key)

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_envs, recurrent_hidden_state_size
        )
        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)
        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        self.actions = torch.zeros(num_steps, num_envs, *action_shape)
        self.masks = torch.ones(num_steps + 1, num_envs, 1)
        self.bad_masks = torch.ones(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device: torch.device) -> None:
        for key in self.obs_keys:
            self.obs[key] = self.obs[key].to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(
        self,
        obs: Mapping[str, torch.Tensor],
        recurrent_hidden_states: torch.Tensor,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        bad_masks: torch.Tensor,
    ) -> None:
        for key, value in obs.items():
            self.obs[key][self.step + 1].copy_(value)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self) -> None:
        for key in self.obs_keys:
            self.obs[key][0].copy_(self.obs[key][-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(
        self,
        next_value: torch.Tensor,
        use_gae: bool,
        gamma: float,
        gae_lambda: float,
        use_proper_time_limits: bool = True,
    ) -> None:
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0.0
            for step in reversed(range(self.rewards.size(0))):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                if use_proper_time_limits:
                    gae = gae * self.bad_masks[step + 1]
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def feed_forward_generator(self, advantages: torch.Tensor, num_mini_batch: int):
        num_steps, num_envs = self.rewards.size()[0:2]
        batch_size = num_envs * num_steps
        mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True
        )

        for indices in sampler:
            obs_batch = {}
            for key in self.obs_keys:
                flat = self.obs[key][:-1].view(-1, *self.obs[key].size()[2:])
                obs_batch[key] = flat[indices]

            rhs_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1)
            )[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield (
                obs_batch,
                rhs_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )

