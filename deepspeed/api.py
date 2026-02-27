from __future__ import annotations

import argparse
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from .config import load_config
from .distributed import dist_env_requests_init, init_distributed, world
from .engine import DeepSpeedEngine
from .fp16_scaler import DynamicLossScaler, FP16ScalerConfig
from .zero_optimizer import DeepSpeedZeroOptimizer


def add_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--deepspeed_config", type=str, default=None)
    return parser


def _maybe_init_dist(dist_init_required: Any):
    if dist.is_initialized():
        return
    if dist_init_required is False:
        return
    if dist_init_required is True:
        init_distributed()
        return
    if dist_env_requests_init():
        init_distributed()


def initialize(
    args: Any = None,
    model: Optional[nn.Module] = None,
    optimizer: Any = None,
    model_parameters: Optional[Iterable[torch.nn.Parameter]] = None,
    training_data: Any = None,
    lr_scheduler: Any = None,
    mpu: Any = None,
    dist_init_required: Any = None,
    collate_fn: Any = None,
    config: Any = None,
    config_params: Any = None,
    **kwargs,
):
    del mpu  # not used in this extracted v1
    if model is None:
        raise ValueError("deepspeed.initialize requires model=...")

    cfg = load_config(args=args, config=config, config_params=config_params)
    _maybe_init_dist(dist_init_required)

    zero_cfg = cfg.get("zero_optimization", {}) or {}
    stage = int(zero_cfg.get("stage", 0))
    if stage == 3:
        raise NotImplementedError("This extracted local version only implements ZeRO-1/2 (and stage0 helpers).")

    data_parallel_group = kwargs.get("data_parallel_group", None)
    sequence_data_parallel_group = kwargs.get("sequence_data_parallel_group", None)
    zero_param_parallel_group = kwargs.get("zero_param_parallel_group", None)
    if data_parallel_group is not None and sequence_data_parallel_group is not None:
        raise ValueError("Both data_parallel_group and sequence_data_parallel_group were specified. Provide only one.")
    dp_group = data_parallel_group if data_parallel_group is not None else sequence_data_parallel_group
    if zero_param_parallel_group is not None:
        if dp_group is not None and dp_group is not zero_param_parallel_group:
            raise NotImplementedError(
                "Different data_parallel_group and zero_param_parallel_group are not supported in this implementation."
            )
        dp_group = zero_param_parallel_group

    if model_parameters is None:
        model_parameters = list(model.parameters())
    else:
        model_parameters = list(model_parameters)

    if optimizer is not None:
        raise ValueError("This implementation builds optimizer from config; keep optimizer=None")

    fp16_cfg = cfg.get("fp16", {}) or {}
    bf16_cfg = cfg.get("bf16", {}) or {}
    fp16_enabled = bool(fp16_cfg.get("enabled", False))
    bf16_enabled = bool(bf16_cfg.get("enabled", False))
    if fp16_enabled and bf16_enabled:
        raise ValueError("fp16 and bf16 cannot both be enabled")

    scaler = DynamicLossScaler(
        FP16ScalerConfig(
            enabled=fp16_enabled,
            loss_scale=float(fp16_cfg.get("loss_scale", 0.0)),
            loss_scale_window=int(fp16_cfg.get("loss_scale_window", 1000)),
            hysteresis=int(fp16_cfg.get("hysteresis", 2)),
            min_loss_scale=float(fp16_cfg.get("min_loss_scale", 1.0)),
            scale_factor=float(fp16_cfg.get("scale_factor", 2.0)),
        )
    )

    opt_cfg = cfg.get("optimizer", {}) or {}
    opt_type = str(opt_cfg.get("type", "AdamW")).strip().lower()
    if opt_type not in ("adamw",):
        raise NotImplementedError(
            "This minimal implementation currently supports only optimizer.type='AdamW'. "
            f"Got optimizer.type={opt_cfg.get('type')!r}."
        )
    opt_params = opt_cfg.get("params", {}) or {}
    lr = float(opt_params.get("lr", 1e-4))
    betas = tuple(opt_params.get("betas", (0.9, 0.999)))
    eps = float(opt_params.get("eps", 1e-8))
    weight_decay = float(opt_params.get("weight_decay", 0.0))

    param_groups: List[Dict[str, Any]] = []
    if model_parameters and isinstance(model_parameters[0], dict):
        for g in model_parameters:
            params = [p for p in g.get("params", []) if p.requires_grad]
            if not params:
                continue
            merged = dict(g)
            merged["params"] = params
            merged.setdefault("lr", lr)
            merged.setdefault("betas", betas)
            merged.setdefault("eps", eps)
            merged.setdefault("weight_decay", weight_decay)
            param_groups.append(merged)
    else:
        param_groups.append(
            {
                "params": [p for p in model_parameters if p.requires_grad],
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            }
        )

    optim = DeepSpeedZeroOptimizer(
        module=model,
        param_groups=param_groups,
        zero_cfg=zero_cfg,
        fp16_scaler=scaler,
        stage=stage,
        dp_group=dp_group,
    )
    engine = DeepSpeedEngine(module=model, optimizer=optim, config=cfg, dp_group=dp_group)

    training_dataloader = None
    if training_data is not None:
        from torch.utils.data import DataLoader, DistributedSampler

        sampler = None
        if dist.is_initialized():
            dp_world_size, dp_rank = world(dp_group)
            sampler = DistributedSampler(
                training_data,
                num_replicas=dp_world_size,
                rank=dp_rank,
            )
        training_dataloader = DataLoader(
            training_data,
            batch_size=int(cfg.get("train_micro_batch_size_per_gpu", 1)),
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=collate_fn,
            drop_last=True,
        )

    return engine, optim, training_dataloader, lr_scheduler
