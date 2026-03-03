import argparse
import json
import os
from pathlib import Path

import deepspeed as ds
import torch
import torch.distributed as dist


def _dist_required() -> bool:
    if "RANK" in os.environ:
        return True
    if "WORLD_SIZE" in os.environ:
        try:
            return int(os.environ.get("WORLD_SIZE", "1")) > 1
        except ValueError:
            return True
    return False


def _resolve_user_path(path_value: str, script_dir: Path) -> str:
    p = Path(path_value)
    if p.is_absolute():
        return str(p)

    candidates = [
        Path.cwd() / p,
        script_dir / p,
        script_dir.parent / p,
    ]
    if p.parts and p.parts[0] == script_dir.name:
        if len(p.parts) > 1:
            candidates.append(script_dir / Path(*p.parts[1:]))
        candidates.append(script_dir.parent / Path(*p.parts))

    seen = set()
    for cand in candidates:
        c = cand.resolve()
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists():
            return key

    return str((script_dir / p).resolve())


def _resolve_model_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    name = str(dtype_name).strip().lower()
    if name == "float32":
        return torch.float32
    if name == "float16":
        if device.type != "cuda":
            raise RuntimeError("float16 requires CUDA.")
        return torch.float16
    if name == "bfloat16":
        if device.type != "cuda":
            raise RuntimeError("bfloat16 requires CUDA.")
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Requested bfloat16 but CUDA device does not support bf16.")
        return torch.bfloat16
    raise ValueError(f"Unsupported --model-dtype={dtype_name!r}")


def _configure_deepspeed_precision(ds_cfg: dict, model_dtype: torch.dtype, device: torch.device) -> str:
    ds_cfg.pop("fp16", None)
    ds_cfg.pop("bf16", None)

    if device.type != "cuda":
        return "fp32"

    if model_dtype == torch.bfloat16:
        ds_cfg["bf16"] = {"enabled": True}
        return "bf16"

    if model_dtype == torch.float16:
        ds_cfg["fp16"] = {"enabled": True, "loss_scale": 0}
        return "fp16"

    return "fp32"


def _normalize_role(raw_role: str):
    role = str(raw_role).strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "model"}:
        return "assistant"
    if role == "system":
        return "system"
    return None


def _render_chat_fallback(messages, add_generation_prompt: bool) -> str:
    lines = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            prefix = "System"
        elif role == "assistant":
            prefix = "Assistant"
        else:
            prefix = "User"
        lines.append(f"{prefix}: {msg['content']}")
    if add_generation_prompt:
        lines.append("Assistant:")
    return "\n".join(lines)


def _normalize_token_ids(token_ids):
    if isinstance(token_ids, dict):
        token_ids = token_ids.get("input_ids")

    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)

    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], (list, tuple)):
        token_ids = list(token_ids[0])

    if not isinstance(token_ids, list):
        raise TypeError(f"Unsupported token ids type: {type(token_ids)!r}")

    return [int(x) for x in token_ids]


def _tokenize_chat(tokenizer, messages, seq_len: int, add_generation_prompt: bool):
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                truncation=True,
                max_length=seq_len,
            )
            return _normalize_token_ids(token_ids)
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            encoded = tokenizer(text, truncation=True, max_length=seq_len, add_special_tokens=False)
            return _normalize_token_ids(encoded["input_ids"])

    text = _render_chat_fallback(messages, add_generation_prompt=add_generation_prompt)
    encoded = tokenizer(text, truncation=True, max_length=seq_len, add_special_tokens=True)
    return _normalize_token_ids(encoded["input_ids"])


def _infer_prompt_len(full_ids, prompt_ids) -> int:
    if not full_ids or not prompt_ids:
        return 0
    if len(full_ids) >= len(prompt_ids) and full_ids[: len(prompt_ids)] == prompt_ids:
        return len(prompt_ids)

    common = 0
    upper = min(len(full_ids), len(prompt_ids))
    while common < upper and full_ids[common] == prompt_ids[common]:
        common += 1
    return common


def _count_shift_valid_labels(labels) -> int:
    if len(labels) <= 1:
        return 0
    return sum(1 for x in labels[1:] if x != -100)


def _load_sft_samples(
    dataset_path: str,
    tokenizer,
    seq_len: int,
    max_samples: int = 0,
    min_target_tokens: int = 2,
):
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(f"Dataset {dataset_path} must be a JSON list.")

    samples = []
    skipped = 0
    stats = {
        "items_total": len(raw_data),
        "assistant_turns": 0,
        "skipped_all_masked": 0,
        "skipped_short_target": 0,
    }

    for item in raw_data:
        if not isinstance(item, dict):
            continue
        turns = item.get("conversations")
        if not isinstance(turns, list):
            continue

        messages = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = _normalize_role(turn.get("from"))
            content = turn.get("value")
            if role is None or not isinstance(content, str):
                continue
            content = content.strip()
            if not content:
                continue
            messages.append({"role": role, "content": content})

        for idx, msg in enumerate(messages):
            if msg["role"] != "assistant" or idx == 0:
                continue

            stats["assistant_turns"] += 1

            prompt_messages = messages[:idx]
            full_messages = messages[: idx + 1]

            full_ids = _tokenize_chat(tokenizer, full_messages, seq_len=seq_len, add_generation_prompt=False)
            prompt_ids = _tokenize_chat(tokenizer, prompt_messages, seq_len=seq_len, add_generation_prompt=True)

            if not full_ids:
                skipped += 1
                continue

            prompt_len = _infer_prompt_len(full_ids, prompt_ids)
            labels = list(full_ids)
            for i in range(prompt_len):
                labels[i] = -100

            if not any(x != -100 for x in labels):
                skipped += 1
                stats["skipped_all_masked"] += 1
                continue

            target_tokens = _count_shift_valid_labels(labels)
            if target_tokens < int(min_target_tokens):
                skipped += 1
                stats["skipped_short_target"] += 1
                continue

            samples.append({"input_ids": full_ids, "labels": labels})
            if max_samples > 0 and len(samples) >= max_samples:
                break

        if max_samples > 0 and len(samples) >= max_samples:
            break

    if not samples:
        raise RuntimeError(
            f"No valid SFT samples built from {dataset_path}. "
            f"assistant_turns={stats['assistant_turns']} "
            f"skipped_all_masked={stats['skipped_all_masked']} "
            f"skipped_short_target={stats['skipped_short_target']}"
        )

    return samples, skipped, stats


def _build_sft_batch(samples, batch_size: int, step: int, rank: int, world_size: int, pad_token_id: int, device):
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    start = step * batch_size * world_size + rank * batch_size
    total = len(samples)

    batch = []
    for offset in range(batch_size):
        idx = (start + offset) % total
        batch.append(samples[idx])

    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.full((batch_size, max_len), int(pad_token_id), dtype=torch.long, device=device)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for row, sample in enumerate(batch):
        ids = torch.tensor(sample["input_ids"], dtype=torch.long, device=device)
        lbs = torch.tensor(sample["labels"], dtype=torch.long, device=device)
        n = int(ids.shape[0])
        input_ids[row, :n] = ids
        labels[row, :n] = lbs
        attention_mask[row, :n] = 1

    return input_ids, attention_mask, labels


def _extract_loss(outputs) -> torch.Tensor:
    if hasattr(outputs, "loss") and outputs.loss is not None:
        return outputs.loss
    if isinstance(outputs, dict) and "loss" in outputs:
        return outputs["loss"]
    if isinstance(outputs, tuple) and outputs and isinstance(outputs[0], torch.Tensor):
        return outputs[0]
    raise RuntimeError("Model outputs do not contain a loss tensor.")


def _any_true_across_ranks(flag: bool, device: torch.device) -> bool:
    if not dist.is_initialized():
        return bool(flag)
    t = torch.tensor(1 if flag else 0, device=device, dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item() > 0)


def _mean_tensor_across_ranks(value: torch.Tensor, device: torch.device) -> float:
    t = value.detach()
    if t.ndim != 0:
        t = t.mean()
    t = t.to(device=device, dtype=torch.float32)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return float(t.item())


def _max_float_across_ranks(value: float, device: torch.device) -> float:
    t = torch.tensor(float(value), device=device, dtype=torch.float32)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _cuda_mem_stats(device: torch.device):
    if device.type != "cuda":
        return None
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / (1024.0 * 1024.0),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1024.0 * 1024.0),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0),
    }


def _engine_zero_grad(engine):
    try:
        engine.zero_grad(set_to_none=True)
    except TypeError:
        engine.zero_grad()


def _has_nonfinite_grads(module) -> bool:
    for p in module.parameters():
        g = getattr(p, "grad", None)
        if g is None:
            continue
        if not bool(torch.isfinite(g).all()):
            return True
    return False


def _repair_nonfinite_params(module) -> int:
    fixed = 0
    for p in module.parameters():
        finite = torch.isfinite(p.data)
        if bool(finite.all()):
            continue
        torch.nan_to_num(p.data, nan=0.0, posinf=1e4, neginf=-1e4, out=p.data)
        fixed += 1
    return fixed


def _init_wandb(args, rank: int, dataset_size: int):
    if rank != 0:
        return None

    try:
        import wandb
    except Exception:
        return None

    try:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model_name": args.model_name,
                "steps": int(args.steps),
                "seq_len": int(args.seq_len),
                "batch_size": int(args.batch_size),
                "dataset_path": str(args.dataset_path),
                "dataset_size": int(dataset_size),
                "attn_impl": args.attn_impl,
            },
        )
        print(f"[wandb] enabled project={args.wandb_project}")
        return run
    except Exception:
        return None


def _load_model_and_tokenizer(args, device: torch.device):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Please install transformers first: pip install transformers") from exc

    model_dtype = _resolve_model_dtype(args.model_dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise RuntimeError("Tokenizer has no pad_token/eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = "right"

    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "torch_dtype": model_dtype,
    }

    attn_impl = args.attn_impl
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            attn_implementation=attn_impl,
            **model_kwargs,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
        attn_impl = "default"

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    model.to(device=device)
    model.train()
    return model, tokenizer, model_dtype, attn_impl


def main():
    parser = argparse.ArgumentParser()
    parser = ds.add_config_arguments(parser)

    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--attn-impl", type=str, default="eager", choices=["eager", "sdpa"])

    parser.add_argument("--dataset-path", type=str, default="data/lima.json")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--min-target-tokens", type=int, default=8)

    parser.add_argument("--zero-stage", type=int, default=None, choices=[0, 1, 2])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--wandb-project", type=str, default="nano-deepspeed-sft")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.deepspeed_config is None:
        args.deepspeed_config = str(script_dir / "ds_config_zero2_official.json")
    else:
        args.deepspeed_config = _resolve_user_path(args.deepspeed_config, script_dir)

    args.dataset_path = _resolve_user_path(args.dataset_path, script_dir)

    dist_required = _dist_required()
    if dist_required:
        ds.init_distributed()

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    device = (
        torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(args.seed)

    with open(args.deepspeed_config, "r", encoding="utf-8") as f:
        ds_cfg = json.load(f)
    if args.zero_stage is not None:
        ds_cfg.setdefault("zero_optimization", {})
        ds_cfg["zero_optimization"]["stage"] = int(args.zero_stage)

    model, tokenizer, model_dtype, attn_impl = _load_model_and_tokenizer(args, device)
    param_dtype = next(model.parameters()).dtype
    ds_precision = _configure_deepspeed_precision(ds_cfg, model_dtype, device)

    samples, skipped_samples, stats = _load_sft_samples(
        args.dataset_path,
        tokenizer,
        seq_len=args.seq_len,
        max_samples=args.max_samples,
        min_target_tokens=args.min_target_tokens,
    )

    if rank == 0:
        print(f"[script] path={Path(__file__).resolve()}")
        print(f"[deepspeed] impl=official module={Path(getattr(ds, '__file__', 'unknown')).resolve()}")
        zero_cfg = ds_cfg.get("zero_optimization", {}) or {}
        print(
            f"[zero] stage={zero_cfg.get('stage', 'n/a')} "
            f"reduce_scatter={zero_cfg.get('reduce_scatter', 'n/a')} "
            f"reduce_bucket_size={zero_cfg.get('reduce_bucket_size', 'n/a')} "
            f"allgather_bucket_size={zero_cfg.get('allgather_bucket_size', 'n/a')} "
            f"communication_data_type={zero_cfg.get('communication_data_type', 'auto')}"
        )
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            total_gb = props.total_memory / (1024.0 * 1024.0 * 1024.0)
            print(f"[cuda] device={props.name} total_mem_gb={total_gb:.2f}")
        print(
            f"[model] {args.model_name} param_dtype={param_dtype} "
            f"compute_dtype={model_dtype} ds_precision={ds_precision} attention={attn_impl}"
        )
        print(f"[data] path={args.dataset_path} samples={len(samples)} skipped={skipped_samples}")
        print(
            f"[data] items_total={stats['items_total']} assistant_turns={stats['assistant_turns']} "
            f"skipped_all_masked={stats['skipped_all_masked']} "
            f"skipped_short_target={stats['skipped_short_target']} "
            f"min_target_tokens={args.min_target_tokens}"
        )

    args.deepspeed_config = None
    engine, _, _, _ = ds.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=ds_cfg,
        dist_init_required=dist_required,
    )

    grad_accum = int(ds_cfg.get("gradient_accumulation_steps", 1))
    _engine_zero_grad(engine)

    wandb_run = _init_wandb(args, rank=rank, dataset_size=len(samples))

    opt_loss_sum = 0.0
    opt_micro_count = 0

    for step in range(args.steps):
        input_ids, attention_mask, labels = _build_sft_batch(
            samples,
            batch_size=args.batch_size,
            step=step,
            rank=rank,
            world_size=world_size,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
        )

        outputs = engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = _extract_loss(outputs)

        non_finite_loss = _any_true_across_ranks(not bool(torch.isfinite(loss.detach()).all()), device)
        if non_finite_loss:
            fixed = _repair_nonfinite_params(engine.module)
            _engine_zero_grad(engine)
            if rank == 0:
                print(f"[warn] non-finite loss at step={step + 1}; repaired_params={fixed}; step skipped")
            continue

        engine.backward(loss)

        non_finite_grad = _any_true_across_ranks(_has_nonfinite_grads(engine.module), device)
        if non_finite_grad:
            _engine_zero_grad(engine)
            if rank == 0:
                print(f"[warn] non-finite gradients at step={step + 1}; optimizer step skipped")
            continue

        reduced_loss = _mean_tensor_across_ranks(loss, device)
        opt_loss_sum += reduced_loss
        opt_micro_count += 1

        engine.step()

        if (step + 1) % grad_accum == 0:
            opt_step = (step + 1) // grad_accum
            opt_loss = opt_loss_sum / max(opt_micro_count, 1)
            mem = _cuda_mem_stats(device)
            if mem is not None:
                alloc_mb_max = _max_float_across_ranks(mem["allocated_mb"], device)
                reserved_mb_max = _max_float_across_ranks(mem["reserved_mb"], device)
                peak_alloc_mb_max = _max_float_across_ranks(mem["peak_allocated_mb"], device)
                peak_reserved_mb_max = _max_float_across_ranks(mem["peak_reserved_mb"], device)
            else:
                alloc_mb_max = None
                reserved_mb_max = None
                peak_alloc_mb_max = None
                peak_reserved_mb_max = None
            if rank == 0:
                if alloc_mb_max is None:
                    print(f"[opt_step {opt_step}] loss={opt_loss:.4f}")
                else:
                    print(
                        f"[opt_step {opt_step}] loss={opt_loss:.4f} "
                        f"cuda_alloc_max_mb={alloc_mb_max:.1f} "
                        f"cuda_reserved_max_mb={reserved_mb_max:.1f} "
                        f"cuda_peak_alloc_max_mb={peak_alloc_mb_max:.1f} "
                        f"cuda_peak_reserved_max_mb={peak_reserved_mb_max:.1f}"
                    )
                if wandb_run is not None:
                    payload = {"train/loss": opt_loss, "train/opt_step": opt_step}
                    if alloc_mb_max is not None:
                        payload.update(
                            {
                                "system/cuda_alloc_max_mb": alloc_mb_max,
                                "system/cuda_reserved_max_mb": reserved_mb_max,
                                "system/cuda_peak_alloc_max_mb": peak_alloc_mb_max,
                                "system/cuda_peak_reserved_max_mb": peak_reserved_mb_max,
                            }
                        )
                    wandb_run.log(payload, step=opt_step)
            opt_loss_sum = 0.0
            opt_micro_count = 0

    if rank == 0 and wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
