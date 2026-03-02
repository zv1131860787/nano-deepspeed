from __future__ import annotations

import json
import os
from typing import Any, Dict


def load_config(args: Any = None, config: Any = None, config_params: Any = None) -> Dict[str, Any]:
    cfg = None
    if config is not None:
        cfg = config
    elif config_params is not None:
        cfg = config_params
    elif args is not None and getattr(args, "deepspeed_config", None):
        cfg = getattr(args, "deepspeed_config")

    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, (str, os.PathLike)):
        with open(os.fspath(cfg), "r", encoding="utf-8") as f:
            return json.load(f)
    raise TypeError(f"config must be dict or path str, got {type(cfg)}")
