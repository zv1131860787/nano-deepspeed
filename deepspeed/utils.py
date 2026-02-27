from __future__ import annotations

import math
from typing import Any, Dict, List

import torch
import torch.distributed as dist


def aligned_numel(numel: int, world_size: int, nccl_start_alignment_factor: int) -> int:
    align = int(nccl_start_alignment_factor) * int(world_size)
    return int(math.ceil(numel / align) * align)


def has_all_gather_into_tensor() -> bool:
    return hasattr(dist, "all_gather_into_tensor")


def flatten_params(params: List[torch.Tensor]) -> torch.Tensor:
    if len(params) == 1:
        return params[0].detach().contiguous().view(-1)
    return torch.cat([p.detach().contiguous().view(-1) for p in params], dim=0)


def supports_post_accumulate_grad_hook() -> bool:
    return hasattr(torch.nn.Parameter, "register_post_accumulate_grad_hook")


def register_post_acc_hook(p: torch.nn.Parameter, fn):
    if supports_post_accumulate_grad_hook():
        return p.register_post_accumulate_grad_hook(lambda *_: fn(p, p.grad))
    # Fallback path runs before `.grad` accumulation on older PyTorch; forward the hook grad
    # so callers can avoid reading a stale/None `p.grad`.
    return p.register_hook(lambda grad: (fn(p, grad), grad)[1])


def comm_dtype_from_cfg(zero_cfg: Dict[str, Any], fallback: torch.dtype) -> torch.dtype:
    s = str(zero_cfg.get("communication_data_type", "")).lower().strip()
    if s in ("fp32", "float32"):
        return torch.float32
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    return fallback
