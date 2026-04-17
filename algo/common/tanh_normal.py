from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.distributions import Normal


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def clamp_log_std(log_std: torch.Tensor) -> torch.Tensor:
    return torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)


@dataclass
class TanhNormalSample:
    action: torch.Tensor
    pre_tanh: torch.Tensor
    log_prob: torch.Tensor


def sample_tanh_normal(mean: torch.Tensor, log_std: torch.Tensor) -> TanhNormalSample:
    """Sample actions using a tanh-squashed Gaussian policy."""
    log_std = clamp_log_std(log_std)
    std = log_std.exp()
    dist = Normal(mean, std)
    pre_tanh = dist.rsample()
    action = torch.tanh(pre_tanh)

    # Change-of-variables correction.
    log_prob = dist.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
    # log(1 - tanh(x)^2) = log(1 - a^2); add eps for stability.
    eps = 1e-6
    log_prob = log_prob - torch.log(1.0 - action.pow(2) + eps).sum(dim=-1, keepdim=True)
    return TanhNormalSample(action=action, pre_tanh=pre_tanh, log_prob=log_prob)


def tanh_normal_log_prob(mean: torch.Tensor, log_std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Compute log π(a) for tanh-squashed Gaussian.

    Args:
        mean: (..., act_dim)
        log_std: (..., act_dim)
        action: (..., act_dim) in [-1, 1]
    """
    log_std = clamp_log_std(log_std)
    std = log_std.exp()
    dist = Normal(mean, std)
    eps = 1e-6
    action = torch.clamp(action, -1.0 + eps, 1.0 - eps)
    pre_tanh = torch.atanh(action)
    log_prob = dist.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
    log_prob = log_prob - torch.log(1.0 - action.pow(2) + eps).sum(dim=-1, keepdim=True)
    return log_prob

