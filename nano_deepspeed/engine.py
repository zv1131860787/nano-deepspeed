from __future__ import annotations

import os
from typing import Any, Dict

import torch
import torch.nn as nn

from .distributed import world
from .zero_optimizer import DeepSpeedZeroOptimizer


class DeepSpeedEngine(nn.Module):
    def __init__(self, module: nn.Module, optimizer: DeepSpeedZeroOptimizer, config: Dict[str, Any], dp_group=None):
        super().__init__()
        self.module = module
        self.optimizer = optimizer
        self.config = config
        self.dp_group = dp_group

        self.world_size, self.global_rank = world(self.dp_group)

        self.gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 1))
        self.clip_grad = float(config.get("gradient_clipping", 0.0))

        self.fp16_enabled = bool((config.get("fp16", {}) or {}).get("enabled", False))
        self.bf16_enabled = bool((config.get("bf16", {}) or {}).get("enabled", False))

        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.device = torch.device("cuda", local_rank)
        else:
            self.device = torch.device("cpu")

        self._micro_steps = 0

    def forward(self, *args, **kwargs):
        if torch.cuda.is_available() and (self.fp16_enabled or self.bf16_enabled):
            dtype = torch.float16 if self.fp16_enabled else torch.bfloat16
            with torch.autocast(device_type="cuda", dtype=dtype):
                return self.module(*args, **kwargs)
        return self.module(*args, **kwargs)

    def backward(self, loss: torch.Tensor, retain_graph: bool = False):
        loss = loss / float(self.gradient_accumulation_steps)
        if self.fp16_enabled:
            loss = self.optimizer.scaler.scale_loss(loss)
        loss.backward(retain_graph=retain_graph)

        self.optimizer.backward_epilogue()
        self._micro_steps += 1

    @torch.no_grad()
    def step(self, force: bool = False):
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self._micro_steps <= 0:
            return

        remainder = self._micro_steps % self.gradient_accumulation_steps
        at_boundary = remainder == 0
        if not at_boundary and not force:
            return
        if force and at_boundary:
            return

        if force and remainder > 0:
            self.optimizer.scale_pending_grads_(float(self.gradient_accumulation_steps) / float(remainder))

        self.optimizer.step(clip_grad=self.clip_grad, fp16=self.fp16_enabled)
        self.optimizer.zero_grad(set_to_none=True)

        if force and remainder > 0:
            self._micro_steps = 0

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        # Clearing grads mid-accumulation should also clear the local micro-step
        # boundary state, otherwise later `step()` calls can fire at the wrong time.
        self._micro_steps = 0
