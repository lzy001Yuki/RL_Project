from __future__ import annotations

from typing import Iterable, Optional

import torch


def is_dist_available() -> bool:
    return torch.distributed.is_available()


def is_dist_initialized() -> bool:
    return is_dist_available() and torch.distributed.is_initialized()


def get_world_size() -> int:
    return int(torch.distributed.get_world_size()) if is_dist_initialized() else 1


def get_rank() -> int:
    return int(torch.distributed.get_rank()) if is_dist_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist_initialized():
        torch.distributed.barrier()


def all_reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    if get_world_size() == 1:
        return tensor
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    if average:
        tensor.div_(get_world_size())
    return tensor


def all_reduce_gradients(
    parameters: Iterable[torch.nn.Parameter],
    average: bool = True,
) -> None:
    """Synchronize gradients across all processes.

    This is a minimal alternative to DDP. It works with arbitrary custom
    forward/evaluate methods because it only relies on `.grad` tensors.
    """
    world_size = get_world_size()
    if world_size == 1:
        return
    for p in parameters:
        if p.grad is None:
            continue
        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
        if average:
            p.grad.div_(world_size)


def broadcast_object(obj, src: int = 0):
    """Broadcast a picklable Python object from `src` to all ranks."""
    if get_world_size() == 1:
        return obj
    obj_list = [obj if get_rank() == src else None]
    torch.distributed.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast a tensor from `src` to all ranks (in-place)."""
    if get_world_size() == 1:
        return tensor
    torch.distributed.broadcast(tensor, src=src)
    return tensor


def broadcast_module(module: torch.nn.Module, src: int = 0) -> None:
    """Broadcast all parameters and buffers of a module from `src` to all ranks."""
    if get_world_size() == 1:
        return
    for p in module.parameters():
        torch.distributed.broadcast(p.data, src=src)
    for b in module.buffers():
        torch.distributed.broadcast(b.data, src=src)
