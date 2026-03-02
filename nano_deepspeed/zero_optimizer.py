from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

import torch
import torch.distributed as dist
import torch.nn as nn

from .distributed import world
from .fp16_scaler import DynamicLossScaler
from .utils import aligned_numel, flatten_params, has_all_gather_into_tensor
from .zero_reducer import Zero1or2Reducer
from .zero_types import FlatGroup


class DeepSpeedZeroOptimizer:
    """
    Extracted ZeRO optimizer for stage 0/1/2 only.
    """

    def __init__(
        self,
        module: nn.Module,
        param_groups: List[Dict[str, Any]],
        zero_cfg: Dict[str, Any],
        fp16_scaler: DynamicLossScaler,
        stage: int,
        dp_group=None,
    ):
        self.module = module
        self.param_groups = param_groups
        self.zero_cfg = zero_cfg
        self.scaler = fp16_scaler
        self.stage = int(stage)
        self.dp_group = dp_group

        self.world_size, self.rank = world(self.dp_group)
        self.nccl_start_alignment_factor = int(zero_cfg.get("nccl_start_alignment_factor", 2))

        self._flats: List[FlatGroup] = []
        self._reducers12: List[Zero1or2Reducer] = []

        if self.stage not in (0, 1, 2):
            raise NotImplementedError(
                f"ZeRO stage {self.stage} is not included in this extracted version. Only stage 0/1/2 are supported."
            )

        self._build_flats()
        if self.stage in (1, 2):
            self._reducers12 = [
                Zero1or2Reducer(fg, stage=self.stage, zero_cfg=zero_cfg, dp_group=self.dp_group)
                for fg in self._flats
            ]

    def _build_flats(self):
        for group_idx, pg in enumerate(self.param_groups):
            params = [p for p in pg["params"] if p.requires_grad]
            if not params:
                continue

            device = params[0].device
            dtype = params[0].dtype
            for pidx, p in enumerate(params[1:], start=1):
                if p.device != device:
                    raise ValueError(
                        "Parameters in the same optimizer group must be on the same device for flat ZeRO packing: "
                        f"group={group_idx}, param_index={pidx}, expected_device={device}, got={p.device}"
                    )
                if p.dtype != dtype:
                    raise ValueError(
                        "Parameters in the same optimizer group must share dtype for flat ZeRO packing: "
                        f"group={group_idx}, param_index={pidx}, expected_dtype={dtype}, got={p.dtype}"
                    )

            shapes = [p.data.shape for p in params]
            numels = [p.numel() for p in params]
            offsets = []
            cur = 0
            for n in numels:
                offsets.append(cur)
                cur += n
            total = cur

            aligned_total = aligned_numel(total, self.world_size, self.nccl_start_alignment_factor)
            P = aligned_total // self.world_size

            flat = flatten_params(params).to(device=device, dtype=dtype)
            if flat.numel() < aligned_total:
                pad = torch.zeros((aligned_total - flat.numel(),), device=device, dtype=dtype)
                flat = torch.cat([flat, pad], dim=0)
            flat = flat.contiguous()

            for p, off, n, shp in zip(params, offsets, numels, shapes):
                p.data = flat[off:off + n].view(shp)

            fg = FlatGroup(
                group_idx=group_idx,
                params=params,
                shapes=shapes,
                numels=numels,
                offsets=offsets,
                total_numel=total,
                aligned_total=aligned_total,
                partition_size=P,
                dtype=dtype,
                device=device,
                flat_param=flat,
            )

            if self.stage == 0:
                fg.fp32_master_full = flat.float().clone()
                fg.exp_avg_full = torch.zeros_like(fg.fp32_master_full)
                fg.exp_avg_sq_full = torch.zeros_like(fg.fp32_master_full)
            else:
                s = self.rank * P
                e = s + P
                fg.fp32_master_shard = flat[s:e].float().clone()
                fg.exp_avg_shard = torch.zeros_like(fg.fp32_master_shard)
                fg.exp_avg_sq_shard = torch.zeros_like(fg.fp32_master_shard)

                fg.grad_full_fp32 = torch.zeros((aligned_total,), device=device, dtype=torch.float32) if self.stage == 1 else None
                fg.grad_partition_fp32 = torch.zeros((P,), device=device, dtype=torch.float32) if self.stage == 2 else None

            self._flats.append(fg)

        if not self._flats:
            raise ValueError("No trainable parameters")

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True):
        for fg in self._flats:
            for p in fg.params:
                p.grad = None if set_to_none else (p.grad.zero_() if p.grad is not None else None)
            if self.stage == 1 and fg.grad_full_fp32 is not None:
                fg.grad_full_fp32.zero_()
            if self.stage == 2 and fg.grad_partition_fp32 is not None:
                fg.grad_partition_fp32.zero_()

    @torch.no_grad()
    def backward_epilogue(self):
        if self.stage in (1, 2):
            for reducer in self._reducers12:
                reducer.backward_epilogue()

    @torch.no_grad()
    def step(self, *, clip_grad: float = 0.0, fp16: bool = False):
        def _sync_found_inf(found_inf: bool, device: torch.device) -> bool:
            if dist.is_initialized() and self.world_size > 1:
                flag = torch.tensor(1.0 if found_inf else 0.0, device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self.dp_group)
                return bool(flag.item() > 0.0)
            return found_inf

        if not self._flats:
            return

        if self.stage == 0:
            full_grads: List[torch.Tensor] = []
            for fg in self._flats:
                grads = []
                for p, n in zip(fg.params, fg.numels):
                    if p.grad is None:
                        grads.append(torch.zeros((n,), device=fg.device, dtype=torch.float32))
                    else:
                        grads.append(p.grad.detach().contiguous().view(-1).float())
                flat_g = torch.cat(grads, dim=0)
                if flat_g.numel() < fg.aligned_total:
                    pad = torch.zeros((fg.aligned_total - flat_g.numel(),), device=fg.device, dtype=torch.float32)
                    flat_g = torch.cat([flat_g, pad], dim=0)
                if dist.is_initialized() and self.world_size > 1:
                    dist.all_reduce(flat_g, op=dist.ReduceOp.SUM, group=self.dp_group)
                    flat_g.div_(float(self.world_size))
                full_grads.append(flat_g)

            if fp16 and self.scaler.cfg.enabled:
                found_inf = any((not torch.isfinite(g).all().item()) for g in full_grads)
                found_inf = _sync_found_inf(found_inf, full_grads[0].device)
                if found_inf:
                    self.scaler.update(found_inf=True)
                    return
                for g in full_grads:
                    self.scaler.unscale_(g)

            clip_coef = 1.0
            if clip_grad and clip_grad > 0:
                local_sq = torch.zeros((), device=full_grads[0].device, dtype=torch.float32)
                for g in full_grads:
                    local_sq.add_((g * g).sum())
                # `full_grads` are already globally reduced/averaged above, so each rank has the
                # same full gradient vector. Reducing the norm again would over-scale clipping.
                gnorm = torch.sqrt(local_sq + 1e-12)
                if gnorm.item() > clip_grad:
                    clip_coef = clip_grad / gnorm.item()

            for fg, flat_g in zip(self._flats, full_grads):
                fg.step += 1
                pg = self.param_groups[fg.group_idx]
                lr = float(pg.get("lr", 1e-4))
                b1, b2 = map(float, pg.get("betas", (0.9, 0.999)))
                eps = float(pg.get("eps", 1e-8))
                wd = float(pg.get("weight_decay", 0.0))

                if clip_coef < 1.0:
                    flat_g.mul_(clip_coef)

                p = fg.fp32_master_full
                m = fg.exp_avg_full
                v = fg.exp_avg_sq_full
                assert p is not None and m is not None and v is not None

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                m.mul_(b1).add_(flat_g, alpha=(1.0 - b1))
                v.mul_(b2).addcmul_(flat_g, flat_g, value=(1.0 - b2))

                bc1 = 1.0 - (b1 ** fg.step)
                bc2 = 1.0 - (b2 ** fg.step)
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                step_size = lr / bc1
                p.addcdiv_(m, denom, value=-step_size)

                fg.flat_param.copy_(p.to(dtype=fg.dtype))
                for p_ in fg.params:
                    p_.grad = None

            if fp16 and self.scaler.cfg.enabled:
                self.scaler.update(found_inf=False)
            return

        if self.stage == 1:
            if fp16 and self.scaler.cfg.enabled:
                found_inf = False
                for fg in self._flats:
                    assert fg.grad_full_fp32 is not None
                    if not torch.isfinite(fg.grad_full_fp32).all().item():
                        found_inf = True
                        break
                found_inf = _sync_found_inf(found_inf, self._flats[0].device)
                if found_inf:
                    self.scaler.update(found_inf=True)
                    return
                for fg in self._flats:
                    assert fg.grad_full_fp32 is not None
                    self.scaler.unscale_(fg.grad_full_fp32)

            clip_coef = 1.0
            if clip_grad and clip_grad > 0:
                local_sq = torch.zeros((), device=self._flats[0].device, dtype=torch.float32)
                for fg in self._flats:
                    assert fg.grad_full_fp32 is not None
                    P = fg.partition_size
                    s = self.rank * P
                    e = s + P
                    local_sq.add_((fg.grad_full_fp32[s:e] * fg.grad_full_fp32[s:e]).sum())
                if dist.is_initialized() and self.world_size > 1:
                    dist.all_reduce(local_sq, op=dist.ReduceOp.SUM, group=self.dp_group)
                gnorm = torch.sqrt(local_sq + 1e-12)
                if gnorm.item() > clip_grad:
                    clip_coef = clip_grad / gnorm.item()

            for fg in self._flats:
                assert fg.grad_full_fp32 is not None
                flat_g_full = fg.grad_full_fp32
                fg.step += 1
                pg = self.param_groups[fg.group_idx]
                lr = float(pg.get("lr", 1e-4))
                b1, b2 = map(float, pg.get("betas", (0.9, 0.999)))
                eps = float(pg.get("eps", 1e-8))
                wd = float(pg.get("weight_decay", 0.0))
                P = fg.partition_size
                s = self.rank * P
                e = s + P

                if clip_coef < 1.0:
                    flat_g_full.mul_(clip_coef)
                g_shard = flat_g_full[s:e].contiguous()

                p_sh = fg.fp32_master_shard
                m_sh = fg.exp_avg_shard
                v_sh = fg.exp_avg_sq_shard
                assert p_sh is not None and m_sh is not None and v_sh is not None

                if wd != 0.0:
                    p_sh.mul_(1.0 - lr * wd)
                m_sh.mul_(b1).add_(g_shard, alpha=(1.0 - b1))
                v_sh.mul_(b2).addcmul_(g_shard, g_shard, value=(1.0 - b2))

                bc1 = 1.0 - (b1 ** fg.step)
                bc2 = 1.0 - (b2 ** fg.step)
                denom = (v_sh.sqrt() / math.sqrt(bc2)).add_(eps)
                step_size = lr / bc1
                p_sh.addcdiv_(m_sh, denom, value=-step_size)

                fg.flat_param[s:e].copy_(p_sh.to(dtype=fg.dtype))

                if dist.is_initialized() and self.world_size > 1:
                    if has_all_gather_into_tensor():
                        out = torch.empty((fg.aligned_total,), device=fg.device, dtype=fg.dtype)
                        dist.all_gather_into_tensor(out, fg.flat_param[s:e].contiguous(), group=self.dp_group)
                        fg.flat_param.copy_(out)
                    else:
                        parts = [torch.empty((P,), device=fg.device, dtype=fg.dtype) for _ in range(self.world_size)]
                        dist.all_gather(parts, fg.flat_param[s:e].contiguous(), group=self.dp_group)
                        fg.flat_param.copy_(torch.cat(parts, dim=0))
                fg.grad_full_fp32.zero_()

            if fp16 and self.scaler.cfg.enabled:
                self.scaler.update(found_inf=False)
            return

        if fp16 and self.scaler.cfg.enabled:
            found_inf = False
            for fg in self._flats:
                assert fg.grad_partition_fp32 is not None
                if not torch.isfinite(fg.grad_partition_fp32).all().item():
                    found_inf = True
                    break
            found_inf = _sync_found_inf(found_inf, self._flats[0].device)
            if found_inf:
                self.scaler.update(found_inf=True)
                return
            for fg in self._flats:
                assert fg.grad_partition_fp32 is not None
                self.scaler.unscale_(fg.grad_partition_fp32)

        clip_coef = 1.0
        if clip_grad and clip_grad > 0:
            local_sq = torch.zeros((), device=self._flats[0].device, dtype=torch.float32)
            for fg in self._flats:
                assert fg.grad_partition_fp32 is not None
                local_sq.add_((fg.grad_partition_fp32 * fg.grad_partition_fp32).sum())
            if dist.is_initialized() and self.world_size > 1:
                dist.all_reduce(local_sq, op=dist.ReduceOp.SUM, group=self.dp_group)
            gnorm = torch.sqrt(local_sq + 1e-12)
            if gnorm.item() > clip_grad:
                clip_coef = clip_grad / gnorm.item()

        for fg in self._flats:
            assert fg.grad_partition_fp32 is not None
            g_part = fg.grad_partition_fp32
            fg.step += 1
            pg = self.param_groups[fg.group_idx]
            lr = float(pg.get("lr", 1e-4))
            b1, b2 = map(float, pg.get("betas", (0.9, 0.999)))
            eps = float(pg.get("eps", 1e-8))
            wd = float(pg.get("weight_decay", 0.0))

            if clip_coef < 1.0:
                g_part.mul_(clip_coef)

            p_sh = fg.fp32_master_shard
            m_sh = fg.exp_avg_shard
            v_sh = fg.exp_avg_sq_shard
            assert p_sh is not None and m_sh is not None and v_sh is not None

            if wd != 0.0:
                p_sh.mul_(1.0 - lr * wd)
            m_sh.mul_(b1).add_(g_part, alpha=(1.0 - b1))
            v_sh.mul_(b2).addcmul_(g_part, g_part, value=(1.0 - b2))

            bc1 = 1.0 - (b1 ** fg.step)
            bc2 = 1.0 - (b2 ** fg.step)
            denom = (v_sh.sqrt() / math.sqrt(bc2)).add_(eps)
            step_size = lr / bc1
            p_sh.addcdiv_(m_sh, denom, value=-step_size)

            P = fg.partition_size
            s = self.rank * P
            e = s + P
            fg.flat_param[s:e].copy_(p_sh.to(dtype=fg.dtype))

            if dist.is_initialized() and self.world_size > 1:
                if has_all_gather_into_tensor():
                    out = torch.empty((fg.aligned_total,), device=fg.device, dtype=fg.dtype)
                    dist.all_gather_into_tensor(out, fg.flat_param[s:e].contiguous(), group=self.dp_group)
                    fg.flat_param.copy_(out)
                else:
                    parts = [torch.empty((P,), device=fg.device, dtype=fg.dtype) for _ in range(self.world_size)]
                    dist.all_gather(parts, fg.flat_param[s:e].contiguous(), group=self.dp_group)
                    fg.flat_param.copy_(torch.cat(parts, dim=0))
            fg.grad_partition_fp32.zero_()

        if fp16 and self.scaler.cfg.enabled:
            self.scaler.update(found_inf=False)

    @torch.no_grad()
    def scale_pending_grads_(self, scale: float):
        if scale == 1.0:
            return

        if self.stage == 0:
            for fg in self._flats:
                for p in fg.params:
                    if p.grad is not None:
                        p.grad.mul_(scale)
            return

        if self.stage == 1:
            for fg in self._flats:
                if fg.grad_full_fp32 is not None:
                    fg.grad_full_fp32.mul_(scale)
            return

        if self.stage == 2:
            for fg in self._flats:
                if fg.grad_partition_fp32 is not None:
                    fg.grad_partition_fp32.mul_(scale)
            return

    def state_dict(self) -> Dict[str, Any]:
        param_groups = []
        for pg in self.param_groups:
            snap: Dict[str, Any] = {}
            for k, v in pg.items():
                if k == "params":
                    snap["param_count"] = len(v)
                    continue
                snap[k] = list(v) if isinstance(v, tuple) else v
            param_groups.append(snap)

        out = {
            "zero_stage": self.stage,
            "world_size": self.world_size,
            "rank": self.rank,
            "param_groups": param_groups,
            "loss_scale": float(self.scaler.scale),
            "flats": [],
        }
        for fg in self._flats:
            g = {
                "group_idx": fg.group_idx,
                "step": fg.step,
                "aligned_total": fg.aligned_total,
                "partition_size": fg.partition_size,
            }
            if self.stage == 0:
                g["fp32_master_full"] = fg.fp32_master_full.detach().cpu()
                g["exp_avg_full"] = fg.exp_avg_full.detach().cpu()
                g["exp_avg_sq_full"] = fg.exp_avg_sq_full.detach().cpu()
            else:
                g["fp32_master_shard"] = fg.fp32_master_shard.detach().cpu()
                g["exp_avg_shard"] = fg.exp_avg_shard.detach().cpu()
                g["exp_avg_sq_shard"] = fg.exp_avg_sq_shard.detach().cpu()
            out["flats"].append(g)
        return out

    @torch.no_grad()
    def load_state_dict(self, state_dict: Dict[str, Any]):
        if not isinstance(state_dict, dict):
            raise TypeError(f"state_dict must be a dict, got {type(state_dict)}")

        ckpt_stage = int(state_dict.get("zero_stage", self.stage))
        if ckpt_stage != self.stage:
            raise ValueError(f"ZeRO stage mismatch: checkpoint={ckpt_stage}, current={self.stage}")

        ckpt_world_size = int(state_dict.get("world_size", self.world_size))
        if ckpt_world_size != self.world_size:
            raise ValueError(f"World size mismatch: checkpoint={ckpt_world_size}, current={self.world_size}")

        ckpt_rank = state_dict.get("rank", None)
        if self.stage in (1, 2) and ckpt_rank is not None and int(ckpt_rank) != self.rank:
            raise ValueError(f"Rank-local ZeRO-{self.stage} checkpoint mismatch: checkpoint rank={ckpt_rank}, current={self.rank}")

        saved_param_groups = state_dict.get("param_groups", None)
        if saved_param_groups is not None:
            if len(saved_param_groups) != len(self.param_groups):
                raise ValueError(
                    f"Param group count mismatch: checkpoint={len(saved_param_groups)}, current={len(self.param_groups)}"
                )
            for pg, saved_pg in zip(self.param_groups, saved_param_groups):
                expected_count = int(saved_pg.get("param_count", len(pg.get("params", []))))
                if expected_count != len(pg.get("params", [])):
                    raise ValueError(
                        "Param group parameter count mismatch: "
                        f"checkpoint={expected_count}, current={len(pg.get('params', []))}"
                    )
                for k, v in saved_pg.items():
                    if k == "param_count":
                        continue
                    pg[k] = tuple(v) if (k == "betas" and isinstance(v, list)) else v

        if "loss_scale" in state_dict:
            self.scaler.scale = float(state_dict["loss_scale"])
            if getattr(self.scaler, "dynamic", False):
                self.scaler._stable_steps = 0
                self.scaler._hysteresis_left = self.scaler.cfg.hysteresis

        saved_flats = state_dict.get("flats", None)
        if saved_flats is None:
            raise ValueError("Optimizer state_dict is missing 'flats'")
        if len(saved_flats) != len(self._flats):
            raise ValueError(f"Flat group count mismatch: checkpoint={len(saved_flats)}, current={len(self._flats)}")

        def _copy_tensor(dst: torch.Tensor, src: Any, *, name: str):
            if src is None:
                raise ValueError(f"Checkpoint is missing tensor '{name}'")
            if not isinstance(src, torch.Tensor):
                raise TypeError(f"Checkpoint tensor '{name}' must be a torch.Tensor, got {type(src)}")
            if src.numel() != dst.numel():
                raise ValueError(
                    f"Tensor size mismatch for '{name}': checkpoint={src.numel()} elements, current={dst.numel()} elements"
                )
            dst.copy_(src.to(device=dst.device, dtype=dst.dtype).view_as(dst))

        for fg, saved_fg in zip(self._flats, saved_flats):
            saved_group_idx = int(saved_fg.get("group_idx", fg.group_idx))
            if saved_group_idx != fg.group_idx:
                raise ValueError(f"Flat group index mismatch: checkpoint={saved_group_idx}, current={fg.group_idx}")

            saved_aligned_total = int(saved_fg.get("aligned_total", fg.aligned_total))
            if saved_aligned_total != fg.aligned_total:
                raise ValueError(
                    f"aligned_total mismatch for group {fg.group_idx}: checkpoint={saved_aligned_total}, current={fg.aligned_total}"
                )

            saved_partition_size = int(saved_fg.get("partition_size", fg.partition_size))
            if saved_partition_size != fg.partition_size:
                raise ValueError(
                    "partition_size mismatch for group "
                    f"{fg.group_idx}: checkpoint={saved_partition_size}, current={fg.partition_size}"
                )

            fg.step = int(saved_fg.get("step", 0))

            if self.stage == 0:
                assert fg.fp32_master_full is not None and fg.exp_avg_full is not None and fg.exp_avg_sq_full is not None
                _copy_tensor(fg.fp32_master_full, saved_fg.get("fp32_master_full"), name="fp32_master_full")
                _copy_tensor(fg.exp_avg_full, saved_fg.get("exp_avg_full"), name="exp_avg_full")
                _copy_tensor(fg.exp_avg_sq_full, saved_fg.get("exp_avg_sq_full"), name="exp_avg_sq_full")
                fg.flat_param.copy_(fg.fp32_master_full.to(dtype=fg.dtype))
                continue

            assert fg.fp32_master_shard is not None and fg.exp_avg_shard is not None and fg.exp_avg_sq_shard is not None
            _copy_tensor(fg.fp32_master_shard, saved_fg.get("fp32_master_shard"), name="fp32_master_shard")
            _copy_tensor(fg.exp_avg_shard, saved_fg.get("exp_avg_shard"), name="exp_avg_shard")
            _copy_tensor(fg.exp_avg_sq_shard, saved_fg.get("exp_avg_sq_shard"), name="exp_avg_sq_shard")

            P = fg.partition_size
            s = self.rank * P
            e = s + P
            fg.flat_param[s:e].copy_(fg.fp32_master_shard.to(dtype=fg.dtype))

            if dist.is_initialized() and self.world_size > 1:
                shard = fg.flat_param[s:e].contiguous()
                if has_all_gather_into_tensor():
                    out = torch.empty((fg.aligned_total,), device=fg.device, dtype=fg.dtype)
                    dist.all_gather_into_tensor(out, shard, group=self.dp_group)
                    fg.flat_param.copy_(out)
                else:
                    parts = [torch.empty((P,), device=fg.device, dtype=fg.dtype) for _ in range(self.world_size)]
                    dist.all_gather(parts, shard, group=self.dp_group)
                    fg.flat_param.copy_(torch.cat(parts, dim=0))

            if fg.grad_full_fp32 is not None:
                fg.grad_full_fp32.zero_()
            if fg.grad_partition_fp32 is not None:
                fg.grad_partition_fp32.zero_()

        self.zero_grad(set_to_none=True)
