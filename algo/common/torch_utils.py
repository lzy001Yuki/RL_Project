from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


TensorOrArray = Union[torch.Tensor, np.ndarray]
Obs = Union[torch.Tensor, Mapping[str, torch.Tensor]]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def to_tensor(x: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t.to(device=device)


def obs_to_tensor(obs: Any, device: torch.device) -> Dict[str, torch.Tensor]:
    """Normalize observations to a dict of tensors on device.

    The repo's instructor env uses dict obs: {"sensor": tensor([...])}.
    This helper keeps algorithms agnostic to whether env returns dict or array.
    """
    if isinstance(obs, dict):
        return {k: to_tensor(v, device=device) for k, v in obs.items()}
    return {"obs": to_tensor(obs, device=device)}


def flatten_obs(obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a dict observation into a single tensor feature vector.

    This uses key-sorted concatenation for determinism.
    """
    keys = sorted(obs.keys())
    parts = [obs[k].view(obs[k].shape[0], -1) if obs[k].ndim > 1 else obs[k].unsqueeze(0) for k in keys]
    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=-1)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Iterable[int],
        out_dim: int,
        activation: nn.Module = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation)
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        if out_activation is not None:
            layers.append(out_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

