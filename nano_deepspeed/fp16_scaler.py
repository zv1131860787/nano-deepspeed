from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FP16ScalerConfig:
    enabled: bool = False
    loss_scale: float = 0.0
    loss_scale_window: int = 1000
    hysteresis: int = 2
    min_loss_scale: float = 1.0
    scale_factor: float = 2.0


class DynamicLossScaler:
    def __init__(self, cfg: FP16ScalerConfig):
        self.cfg = cfg
        if not cfg.enabled:
            self.scale = 1.0
            self.dynamic = False
            return
        if cfg.loss_scale and cfg.loss_scale > 0:
            self.scale = float(cfg.loss_scale)
            self.dynamic = False
        else:
            self.scale = 2.0 ** 16
            self.dynamic = True
        self._stable_steps = 0
        self._hysteresis_left = cfg.hysteresis

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale if self.cfg.enabled else loss

    def unscale_(self, t: torch.Tensor):
        if self.cfg.enabled:
            t.div_(self.scale)

    def update(self, found_inf: bool):
        if not self.cfg.enabled or not self.dynamic:
            return
        if found_inf:
            if self._hysteresis_left > 1:
                self._hysteresis_left -= 1
            else:
                self.scale = max(self.scale / self.cfg.scale_factor, self.cfg.min_loss_scale)
                self._hysteresis_left = self.cfg.hysteresis
            self._stable_steps = 0
        else:
            self._stable_steps += 1
            if self._stable_steps % self.cfg.loss_scale_window == 0:
                self.scale *= self.cfg.scale_factor
