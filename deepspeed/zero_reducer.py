from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from .distributed import world
from .utils import comm_dtype_from_cfg, register_post_acc_hook, supports_post_accumulate_grad_hook
from .zero_types import FlatGroup, _IPGParamEntry


class Zero1or2Reducer:
    """
    ZeRO Stage-1/2 reducer with hook-driven bucketed gradient communication.
    """

    def __init__(
        self,
        flat: FlatGroup,
        *,
        stage: int,
        zero_cfg: Dict[str, Any],
        dp_group=None,
    ):
        assert stage in (1, 2)
        self.flat = flat
        self.stage = stage
        self.zero_cfg = zero_cfg
        self.dp_group = dp_group

        self.world_size, self.rank = world(self.dp_group)
        self.device = flat.device

        self.overlap_comm = bool(zero_cfg.get("overlap_comm", True))
        self._supports_post_acc_hook = supports_post_accumulate_grad_hook()
        if not self._supports_post_acc_hook:
            # Without post-accumulate hooks, reading `.grad` inside Tensor hooks is not reliable.
            # Fall back to collecting gradients from `p.grad` in backward_epilogue().
            self.overlap_comm = False
        self.reduce_scatter = bool(zero_cfg.get("reduce_scatter", True))
        # DeepSpeed's upstream defaults can be very large; use a safer local default for this minimal implementation.
        self.reduce_bucket_size = int(zero_cfg.get("reduce_bucket_size", 5e6))
        self.nccl_start_alignment_factor = int(zero_cfg.get("nccl_start_alignment_factor", 2))
        self.stage1_partition_aware_grad_reduce = bool(zero_cfg.get("stage1_partition_aware_grad_reduce", True))
        self.use_multi_rank_bucket_allreduce = bool(zero_cfg.get("use_multi_rank_bucket_allreduce", True))
        if not self.use_multi_rank_bucket_allreduce:
            raise NotImplementedError(
                "use_multi_rank_bucket_allreduce=False is not implemented in this minimal version yet."
            )
        self.stage2_comm_strategy = str(zero_cfg.get("stage2_comm_strategy", "official")).strip().lower()
        if self.stage2_comm_strategy in ("auto",):
            self.stage2_comm_strategy = "official"
        if self.stage2_comm_strategy != "official":
            raise ValueError(
                f"Unsupported stage2_comm_strategy={self.stage2_comm_strategy!r}, only 'official' is supported."
            )

        self.comm_dtype = comm_dtype_from_cfg(zero_cfg, fallback=flat.dtype)
        self.ignore_unused_parameters = bool(zero_cfg.get("ignore_unused_parameters", False))

        self.comm_stream = (
            torch.cuda.Stream()
            if (torch.cuda.is_available() and self.overlap_comm and self.world_size > 1)
            else None
        )

        self._stage2_param_partitions = self._build_stage2_param_partitions()

        self._param_order = [i for i, p in enumerate(self.flat.params) if p.requires_grad]
        self._param_order.reverse()
        self._ready_flags = [False for _ in self.flat.params]
        self._order_pos = 0

        self.reduce_bucket_size = max(1, int(self.reduce_bucket_size))
        self.reduce_bucket_size = min(self.reduce_bucket_size, max(1, int(self.flat.aligned_total)))
        nbuf = 2 if (self.overlap_comm and self.world_size > 1) else 1
        self._ipg_buffers = [
            torch.empty((self.reduce_bucket_size,), device=self.device, dtype=self.comm_dtype)
            for _ in range(nbuf)
        ]
        self._ipg_active_idx = 0
        self._ipg_elements = 0
        self._ipg_entries: List[_IPGParamEntry] = []
        self._busy_buffers = set()

        self._pending: List[Dict[str, Any]] = []

        self._hooks = []
        self.register_hooks()

    def _build_stage2_param_partitions(self) -> Dict[int, List[Tuple[int, int, int]]]:
        P = int(self.flat.partition_size)
        out: Dict[int, List[Tuple[int, int, int]]] = {}

        for pidx, (off, n) in enumerate(zip(self.flat.offsets, self.flat.numels)):
            end = int(off + n)
            if n <= 0:
                out[pidx] = []
                continue

            first_pid = max(0, int(off) // P)
            last_pid = min(self.world_size - 1, (end - 1) // P)
            parts: List[Tuple[int, int, int]] = []

            for pid in range(first_pid, last_pid + 1):
                part_s = pid * P
                part_e = part_s + P
                is_ = max(int(off), part_s)
                ie_ = min(end, part_e)
                L = ie_ - is_
                if L > 0:
                    parts.append((pid, is_ - int(off), L))

            out[pidx] = parts
        return out

    def register_hooks(self):
        if not self._supports_post_acc_hook:
            return
        for pidx, p in enumerate(self.flat.params):
            if not p.requires_grad:
                continue

            def _fn(param, grad, pidx=pidx):
                self.on_grad_ready(pidx, grad)

            h = register_post_acc_hook(p, _fn)
            self._hooks.append(h)

    def on_grad_ready(self, pidx: int, grad: Optional[torch.Tensor]):
        del grad
        self._ready_flags[pidx] = True
        self._drain_ready_prefix(force_zeros=False)

    def _use_async_collective(self) -> bool:
        return bool(self.overlap_comm and dist.is_initialized() and self.world_size > 1)

    def _sync_from_comm_stream(self):
        if self.comm_stream is not None and torch.cuda.is_available():
            torch.cuda.current_stream().wait_stream(self.comm_stream)

    def _launch_all_reduce(self, tensor: torch.Tensor, *, async_op: bool):
        if not (dist.is_initialized() and self.world_size > 1):
            return None
        if self.comm_stream is not None and torch.cuda.is_available():
            self.comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.comm_stream):
                return dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.dp_group, async_op=async_op)
        return dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.dp_group, async_op=async_op)

    def _stage2_build_rank_and_offsets(self, entries: List[_IPGParamEntry]) -> List[Tuple[int, int, int, int]]:
        P = int(self.flat.partition_size)
        raw: List[Tuple[int, int, int, int]] = []

        for ent in entries:
            for pid, off_in_param, n in self._stage2_param_partitions.get(ent.pidx, []):
                g_off = ent.global_off + int(off_in_param)
                part_off = g_off - pid * P
                raw.append((pid, ent.bucket_off + int(off_in_param), int(n), int(part_off)))

        if not raw:
            return []

        merged: List[Tuple[int, int, int, int]] = []
        for pid, boff, n, part_off in raw:
            if merged:
                m_pid, m_boff, m_n, m_part_off = merged[-1]
                if m_pid == pid and (m_boff + m_n == boff) and (m_part_off + m_n == part_off):
                    merged[-1] = (m_pid, m_boff, m_n + n, m_part_off)
                    continue
            merged.append((pid, boff, n, part_off))
        return merged

    def _drain_ready_prefix(self, *, force_zeros: bool):
        while self._order_pos < len(self._param_order):
            pidx = self._param_order[self._order_pos]
            is_ready = self._ready_flags[pidx]
            if not is_ready:
                if not force_zeros:
                    break
                if not self.ignore_unused_parameters:
                    p = self.flat.params[pidx]
                    raise RuntimeError(
                        "Unused parameter detected during ZeRO gradient reduction "
                        f"(stage={self.stage}, param_index={pidx}, shape={tuple(p.shape)}). "
                        "Set zero_optimization.ignore_unused_parameters=true to treat missing grads as zero."
                    )
            self._consume_param_grad(pidx, force_zero=(not is_ready))
            self._ready_flags[pidx] = False
            self._order_pos += 1

    @torch.no_grad()
    def _consume_param_grad(self, pidx: int, *, force_zero: bool):
        p = self.flat.params[pidx]
        n = int(self.flat.numels[pidx])

        if n <= 0:
            p.grad = None
            return

        grad_vec = None
        if (not force_zero) and (p.grad is not None):
            g = p.grad.detach()
            if g.is_sparse:
                raise ValueError("Sparse grads not supported in this minimal implementation.")
            grad_vec = g.contiguous().view(-1)
            if int(grad_vec.numel()) != n:
                raise RuntimeError(f"Gradient size mismatch for param index {pidx}: expected {n}, got {grad_vec.numel()}")

        p.grad = None

        if n > self.reduce_bucket_size:
            self._flush_active_bucket()

            single = torch.zeros((n,), device=self.device, dtype=self.comm_dtype)
            if grad_vec is not None:
                single.copy_(grad_vec.to(dtype=self.comm_dtype), non_blocking=True)

            ent = _IPGParamEntry(
                pidx=pidx,
                global_off=int(self.flat.offsets[pidx]),
                bucket_off=0,
                numel=n,
            )
            self._launch_bucket(single, [ent], buffer_idx=None)
            return

        if self._ipg_elements + n > self.reduce_bucket_size:
            self._flush_active_bucket()

        self._ensure_active_buffer_available()
        buf = self._ipg_buffers[self._ipg_active_idx]

        dst = buf.narrow(0, self._ipg_elements, n)
        if grad_vec is None:
            dst.zero_()
        else:
            dst.copy_(grad_vec.to(dtype=self.comm_dtype), non_blocking=True)

        self._ipg_entries.append(
            _IPGParamEntry(
                pidx=pidx,
                global_off=int(self.flat.offsets[pidx]),
                bucket_off=int(self._ipg_elements),
                numel=n,
            )
        )
        self._ipg_elements += n

        if self._ipg_elements >= self.reduce_bucket_size:
            self._flush_active_bucket()

    def _ensure_active_buffer_available(self):
        if self._ipg_active_idx in self._busy_buffers:
            self._select_next_active_buffer()

    def _select_next_active_buffer(self):
        for i in range(len(self._ipg_buffers)):
            if i not in self._busy_buffers:
                self._ipg_active_idx = i
                return
        self._finalize_oldest_pending()
        for i in range(len(self._ipg_buffers)):
            if i not in self._busy_buffers:
                self._ipg_active_idx = i
                return
        raise RuntimeError("No available IPG buffer after finalizing pending communication.")

    @torch.no_grad()
    def _flush_active_bucket(self):
        if self._ipg_elements == 0:
            return

        idx = self._ipg_active_idx
        buf = self._ipg_buffers[idx].narrow(0, 0, self._ipg_elements)
        entries = list(self._ipg_entries)

        self._launch_bucket(buf, entries, buffer_idx=idx)

        self._ipg_entries = []
        self._ipg_elements = 0
        self._select_next_active_buffer()

    def _launch_bucket(self, bucket_tensor: torch.Tensor, entries: List[_IPGParamEntry], buffer_idx: Optional[int]):
        if self.stage == 1:
            self._launch_stage1_bucket(bucket_tensor, entries, buffer_idx)
        else:
            self._launch_stage2_bucket(bucket_tensor, entries, buffer_idx)

    def _launch_stage1_bucket(self, bucket_tensor: torch.Tensor, entries: List[_IPGParamEntry], buffer_idx: Optional[int]):
        async_flag = self._use_async_collective()

        use_partition_aware = bool(
            self.stage1_partition_aware_grad_reduce
            and self.reduce_scatter
            and dist.is_initialized()
            and self.world_size > 1
        )

        if use_partition_aware:
            pieces = self._stage2_build_rank_and_offsets(entries)
            chunks: List[Dict[str, Any]] = []

            small: List[Tuple[int, int, int, int]] = []
            numel = 0

            def _flush_small():
                nonlocal small, numel
                if not small:
                    return

                total = sum(x[2] for x in small)
                packed = torch.empty((total,), device=self.device, dtype=self.comm_dtype)
                meta = []
                off = 0
                for dst, bucket_off, n, part_off in small:
                    packed[off:off + n].copy_(bucket_tensor.narrow(0, int(bucket_off), int(n)), non_blocking=True)
                    meta.append((int(dst), int(off), int(n), int(part_off)))
                    off += n

                work = self._launch_all_reduce(packed, async_op=async_flag)
                chunks.append({"tensor": packed, "work": work, "meta": meta})

                small = []
                numel = 0

            for dst, bucket_off, n, part_off in pieces:
                small.append((dst, bucket_off, n, part_off))
                numel += int(n)
                if numel >= self.reduce_bucket_size:
                    _flush_small()
            _flush_small()

            item = {
                "kind": "stage1",
                "buffer_idx": buffer_idx,
                "holds_buffer": False,
                "entries": entries,
                "partition_aware": True,
                "pieces": pieces,
                "chunks": chunks,
            }
        else:
            work = self._launch_all_reduce(bucket_tensor, async_op=async_flag)
            item = {
                "kind": "stage1",
                "buffer_idx": buffer_idx,
                "holds_buffer": True,
                "tensor": bucket_tensor,
                "entries": entries,
                "work": work,
                "partition_aware": False,
            }

        if async_flag and buffer_idx is not None and item["holds_buffer"]:
            self._busy_buffers.add(buffer_idx)

        self._pending.append(item)
        if not async_flag:
            self._finalize_oldest_pending()

    def _launch_stage2_bucket(self, bucket_tensor: torch.Tensor, entries: List[_IPGParamEntry], buffer_idx: Optional[int]):
        pieces = self._stage2_build_rank_and_offsets(entries)
        async_flag = self._use_async_collective()
        chunks: List[Dict[str, Any]] = []
        holds_buffer = True

        if self.reduce_scatter and dist.is_initialized() and self.world_size > 1:
            holds_buffer = False
            small: List[Tuple[int, int, int, int]] = []
            numel = 0

            def _flush_small():
                nonlocal small, numel
                if not small:
                    return
                total = sum(x[2] for x in small)
                packed = torch.empty((total,), device=self.device, dtype=self.comm_dtype)
                meta = []
                off = 0
                for dst, bucket_off, n, part_off in small:
                    packed[off:off + n].copy_(bucket_tensor.narrow(0, int(bucket_off), int(n)), non_blocking=True)
                    meta.append((int(dst), int(off), int(n), int(part_off)))
                    off += n
                work = self._launch_all_reduce(packed, async_op=async_flag)
                chunks.append({"tensor": packed, "work": work, "meta": meta})
                small = []
                numel = 0

            for dst, bucket_off, n, part_off in pieces:
                small.append((dst, bucket_off, n, part_off))
                numel += int(n)
                if numel >= self.reduce_bucket_size:
                    _flush_small()
            _flush_small()
        else:
            work = self._launch_all_reduce(bucket_tensor, async_op=async_flag)
            chunks.append({"tensor": bucket_tensor, "work": work, "meta": None})

        item = {
            "kind": "stage2",
            "buffer_idx": buffer_idx,
            "holds_buffer": holds_buffer,
            "chunks": chunks,
            "pieces": pieces,
        }
        if async_flag and buffer_idx is not None and item["holds_buffer"]:
            self._busy_buffers.add(buffer_idx)
        self._pending.append(item)
        if not async_flag:
            self._finalize_oldest_pending()

    def _finalize_stage1_item(self, item: Dict[str, Any]):
        if item.get("partition_aware", False):
            chunks = item.get("chunks", [])
            for ch in chunks:
                w = ch.get("work", None)
                if w is not None:
                    w.wait()
            self._sync_from_comm_stream()

            assert self.flat.grad_full_fp32 is not None
            if dist.is_initialized() and self.world_size > 1:
                for ch in chunks:
                    ch["tensor"].div_(float(self.world_size))

            part_start = int(self.rank * self.flat.partition_size)
            for ch in chunks:
                meta = ch.get("meta", None)
                if meta is None:
                    continue
                t = ch["tensor"]
                for dst, off, n, part_off in meta:
                    if int(dst) != self.rank:
                        continue
                    g_off = part_start + int(part_off)
                    self.flat.grad_full_fp32[g_off:g_off + int(n)].add_(
                        t.narrow(0, int(off), int(n)).to(torch.float32)
                    )
            return

        work = item.get("work", None)
        if work is not None:
            work.wait()
        self._sync_from_comm_stream()

        t = item["tensor"]
        if dist.is_initialized() and self.world_size > 1:
            t.div_(float(self.world_size))

        assert self.flat.grad_full_fp32 is not None
        for ent in item["entries"]:
            src = t.narrow(0, int(ent.bucket_off), int(ent.numel))
            g_off = int(ent.global_off)
            self.flat.grad_full_fp32[g_off:g_off + int(ent.numel)].add_(src.to(torch.float32))

    def _finalize_stage2_item(self, item: Dict[str, Any]):
        chunks = item["chunks"]
        for ch in chunks:
            w = ch.get("work", None)
            if w is not None:
                w.wait()
        self._sync_from_comm_stream()

        assert self.flat.grad_partition_fp32 is not None

        if dist.is_initialized() and self.world_size > 1:
            for ch in chunks:
                ch["tensor"].div_(float(self.world_size))

        used_meta = False
        for ch in chunks:
            meta = ch.get("meta", None)
            if meta is None:
                continue
            used_meta = True
            t = ch["tensor"]
            for dst, off, n, part_off in meta:
                if int(dst) != self.rank:
                    continue
                self.flat.grad_partition_fp32[int(part_off):int(part_off) + int(n)].add_(
                    t.narrow(0, int(off), int(n)).to(torch.float32)
                )

        if used_meta:
            return

        if not chunks:
            return
        bucket_tensor = chunks[0]["tensor"]
        for dst, bucket_off, n, part_off in item["pieces"]:
            if int(dst) != self.rank:
                continue
            self.flat.grad_partition_fp32[int(part_off):int(part_off) + int(n)].add_(
                bucket_tensor.narrow(0, int(bucket_off), int(n)).to(torch.float32)
            )

    def _finalize_oldest_pending(self):
        if not self._pending:
            return
        item = self._pending.pop(0)
        kind = item.get("kind", "")
        if kind == "stage1":
            self._finalize_stage1_item(item)
        elif kind == "stage2":
            self._finalize_stage2_item(item)
        else:
            raise RuntimeError(f"Unknown pending item kind: {kind}")

        buf_idx = item.get("buffer_idx", None)
        if item.get("holds_buffer", False) and buf_idx is not None and buf_idx in self._busy_buffers:
            self._busy_buffers.remove(buf_idx)

    def _finalize_all_pending(self):
        while self._pending:
            self._finalize_oldest_pending()

    @torch.no_grad()
    def backward_epilogue(self):
        if not self._supports_post_acc_hook:
            for pidx, p in enumerate(self.flat.params):
                if p.requires_grad and p.grad is not None:
                    self._ready_flags[pidx] = True
        self._drain_ready_prefix(force_zeros=True)
        self._flush_active_bucket()
        self._finalize_all_pending()

        self._ready_flags = [False for _ in self.flat.params]
        self._order_pos = 0
