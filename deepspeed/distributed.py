from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist


def init_distributed(dist_backend: Optional[str] = None, auto_mpi_discovery: bool = True):
    del auto_mpi_discovery  # kept for DeepSpeed-compatible signature
    if dist.is_initialized():
        return
    if dist_backend is None:
        dist_backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=dist_backend)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)


def dist_env_requests_init() -> bool:
    if "RANK" in os.environ:
        return True
    if "WORLD_SIZE" in os.environ:
        try:
            return int(os.environ.get("WORLD_SIZE", "1")) > 1
        except ValueError:
            return True
    return ("LOCAL_RANK" in os.environ) and ("WORLD_SIZE" in os.environ)


def world(group=None) -> Tuple[int, int]:
    if not dist.is_initialized():
        return 1, 0
    if group is None:
        return dist.get_world_size(), dist.get_rank()
    return dist.get_world_size(group=group), dist.get_rank(group=group)
