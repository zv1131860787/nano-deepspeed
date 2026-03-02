from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class FlatGroup:
    group_idx: int
    params: List[torch.nn.Parameter]
    shapes: List[torch.Size]
    numels: List[int]
    offsets: List[int]
    total_numel: int
    aligned_total: int
    partition_size: int
    dtype: torch.dtype
    device: torch.device

    flat_param: torch.Tensor

    fp32_master_full: Optional[torch.Tensor] = None
    exp_avg_full: Optional[torch.Tensor] = None
    exp_avg_sq_full: Optional[torch.Tensor] = None

    fp32_master_shard: Optional[torch.Tensor] = None
    exp_avg_shard: Optional[torch.Tensor] = None
    exp_avg_sq_shard: Optional[torch.Tensor] = None

    grad_full_fp32: Optional[torch.Tensor] = None
    grad_partition_fp32: Optional[torch.Tensor] = None

    step: int = 0


@dataclass
class _IPGParamEntry:
    pidx: int
    global_off: int
    bucket_off: int
    numel: int
