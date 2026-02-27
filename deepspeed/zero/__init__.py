from __future__ import annotations

import json
import os
from typing import Any, Dict


def _load_zero_cfg(config_dict_or_path: Any = None, config: Any = None) -> Dict[str, Any]:
    cfg_obj = config_dict_or_path if config_dict_or_path is not None else config
    if cfg_obj is None:
        return {}
    if isinstance(cfg_obj, dict):
        return cfg_obj
    if isinstance(cfg_obj, (str, os.PathLike)):
        with open(os.fspath(cfg_obj), "r", encoding="utf-8") as f:
            return json.load(f)
    raise TypeError(f"config must be dict or path str, got {type(cfg_obj)}")


class Init:
    """
    ZeRO-1/2-only local package compatibility stub for `deepspeed.zero.Init`.
    """

    def __init__(
        self,
        module=None,
        data_parallel_group=None,
        mem_efficient_linear=True,
        remote_device=None,
        pin_memory=False,
        config_dict_or_path=None,
        config=None,
        enabled=True,
        dtype=None,
        mpu=None,
        zero_param_parallel_group=None,
        zero_quantized_weights=False,
        zero_quantized_nontrainable_weights=False,
        sequence_data_parallel_group=None,
        param_swapper=None,
        tensor_overrides=None,
    ):
        del module, data_parallel_group, mem_efficient_linear, remote_device, pin_memory
        del dtype, mpu, zero_param_parallel_group, zero_quantized_weights
        del zero_quantized_nontrainable_weights, sequence_data_parallel_group, param_swapper, tensor_overrides
        self.enabled = bool(enabled)
        self.cfg = _load_zero_cfg(config_dict_or_path=config_dict_or_path, config=config)

    def __enter__(self):
        if not self.enabled:
            return self
        zero_cfg = self.cfg.get("zero_optimization", {}) or {}
        stage = int(zero_cfg.get("stage", 0))
        if stage == 3:
            raise NotImplementedError("This extracted local version only implements ZeRO-1/2 (and stage0 helpers).")
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
