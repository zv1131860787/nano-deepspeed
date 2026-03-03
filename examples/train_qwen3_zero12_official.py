import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

import deepspeed as ds


def _dist_required() -> bool:
    if "RANK" in os.environ:
        return True
    if "WORLD_SIZE" in os.environ:
        try:
            return int(os.environ.get("WORLD_SIZE", "1")) > 1
        except ValueError:
            return True
    return False


def _resolve_model_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    def _cuda_bf16_supported() -> bool:
        if device.type != "cuda":
            return False
        if hasattr(torch.cuda, "is_bf16_supported"):
            try:
                return bool(torch.cuda.is_bf16_supported())
            except Exception:
                return False
        return False

    name = str(dtype_name).strip().lower()
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        if device.type == "cuda" and not _cuda_bf16_supported():
            raise RuntimeError(
                "Requested --model-dtype=bfloat16 but current CUDA device does not support bf16."
            )
        return torch.bfloat16
    if name != "auto":
        raise ValueError(f"Unsupported --model-dtype={dtype_name!r}")

    if device.type != "cuda":
        return torch.float32

    return torch.bfloat16 if _cuda_bf16_supported() else torch.float16


def _configure_deepspeed_precision(ds_cfg, *, model_dtype: torch.dtype, device: torch.device) -> str:
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


def _load_qwen3(args, device: torch.device):
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "This example requires `transformers`. Install it first, e.g. `pip install transformers`."
        ) from exc

    model_dtype = _resolve_model_dtype(args.model_dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer has no pad_token/eos_token; please provide a compatible tokenizer or set one manually."
            )
        tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    attn_impl = "default"
    flash_fallback_reason = None
    if device.type == "cuda":
        try:
            model = AutoModelForCausalLM.from_config(
                model_config,
                trust_remote_code=args.trust_remote_code,
                attn_implementation="flash_attention_2",
            )
            attn_impl = "flash_attention_2"
        except Exception as exc:
            flash_fallback_reason = str(exc).strip().splitlines()[0]
            model = AutoModelForCausalLM.from_config(
                model_config,
                trust_remote_code=args.trust_remote_code,
            )
    else:
        model = AutoModelForCausalLM.from_config(
            model_config,
            trust_remote_code=args.trust_remote_code,
        )
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    # Keep parameters in fp32 for stability; let DeepSpeed mixed-precision control compute dtype.
    model.to(device=device)
    model.train()
    return model, tokenizer, model_dtype, attn_impl, flash_fallback_reason


def _normalize_role(raw_role: str):
    role = str(raw_role).strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "model"}:
        return "assistant"
    if role == "system":
        return "system"
    return None


def _render_chat_fallback(messages, *, add_generation_prompt: bool) -> str:
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
        if "input_ids" not in token_ids:
            raise TypeError("chat template output dict has no 'input_ids'")
        token_ids = token_ids["input_ids"]

    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    elif hasattr(token_ids, "tolist") and not isinstance(token_ids, list):
        try:
            token_ids = token_ids.tolist()
        except Exception:
            pass

    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)

    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], (list, tuple)):
        token_ids = list(token_ids[0])

    if not isinstance(token_ids, list):
        raise TypeError(f"Unsupported token ids type: {type(token_ids)!r}")

    try:
        return [int(x) for x in token_ids]
    except Exception as exc:
        raise TypeError(f"Token ids are not integer-like: {type(token_ids)!r}") from exc


def _tokenize_chat(tokenizer, messages, *, seq_len: int, add_generation_prompt: bool):
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
            chat_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            encoded = tokenizer(
                chat_text,
                truncation=True,
                max_length=seq_len,
                add_special_tokens=False,
            )
            return _normalize_token_ids(encoded["input_ids"])

    chat_text = _render_chat_fallback(messages, add_generation_prompt=add_generation_prompt)
    encoded = tokenizer(
        chat_text,
        truncation=True,
        max_length=seq_len,
        add_special_tokens=True,
    )
    return _normalize_token_ids(encoded["input_ids"])


def _infer_prompt_len(full_ids, prompt_ids):
    if not full_ids or not prompt_ids:
        return 0, "empty"
    if len(full_ids) >= len(prompt_ids) and full_ids[: len(prompt_ids)] == prompt_ids:
        return len(prompt_ids), "prefix"

    # Fallback when tokenizer truncation/template behavior breaks strict prefix matching.
    common = 0
    upper = min(len(full_ids), len(prompt_ids))
    while common < upper and full_ids[common] == prompt_ids[common]:
        common += 1
    return common, "common_prefix"


def _load_sft_samples(
    dataset_path: str,
    tokenizer,
    *,
    seq_len: int,
    max_samples: int = 0,
    return_stats: bool = False,
):
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(f"Dataset {dataset_path} must be a JSON list.")

    samples = []
    skipped = 0
    stats = {
        "items_total": len(raw_data),
        "items_kept": 0,
        "messages_kept": 0,
        "assistant_turns": 0,
        "prompt_alignment_common_prefix": 0,
        "skipped_empty_full_ids": 0,
        "skipped_all_masked": 0,
    }
    for item in raw_data:
        if not isinstance(item, dict):
            continue
        turns = item.get("conversations")
        if not isinstance(turns, list):
            continue
        stats["items_kept"] += 1

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
        stats["messages_kept"] += len(messages)

        for idx, msg in enumerate(messages):
            if msg["role"] != "assistant" or idx == 0:
                continue
            stats["assistant_turns"] += 1

            prompt_messages = messages[:idx]
            full_messages = messages[: idx + 1]

            full_ids = _tokenize_chat(
                tokenizer,
                full_messages,
                seq_len=seq_len,
                add_generation_prompt=False,
            )
            prompt_ids = _tokenize_chat(
                tokenizer,
                prompt_messages,
                seq_len=seq_len,
                add_generation_prompt=True,
            )

            if not full_ids:
                skipped += 1
                stats["skipped_empty_full_ids"] += 1
                continue

            prompt_len, align_mode = _infer_prompt_len(full_ids, prompt_ids)
            if align_mode == "common_prefix":
                stats["prompt_alignment_common_prefix"] += 1
            labels = list(full_ids)
            for pos in range(prompt_len):
                labels[pos] = -100

            if not any(token != -100 for token in labels):
                skipped += 1
                stats["skipped_all_masked"] += 1
                continue

            samples.append({"input_ids": full_ids, "labels": labels})
            if max_samples > 0 and len(samples) >= max_samples:
                if return_stats:
                    return samples, skipped, stats
                return samples, skipped

    if not samples:
        detail = (
            f"items_total={stats['items_total']} "
            f"items_kept={stats['items_kept']} "
            f"messages_kept={stats['messages_kept']} "
            f"assistant_turns={stats['assistant_turns']} "
            f"skipped_empty_full_ids={stats['skipped_empty_full_ids']} "
            f"skipped_all_masked={stats['skipped_all_masked']} "
            f"prompt_alignment_common_prefix={stats['prompt_alignment_common_prefix']}"
        )
        raise RuntimeError(
            f"No valid SFT samples were built from {dataset_path}. "
            "Check dataset schema or increase --seq-len. "
            f"SFT stats: {detail}"
        )
    if return_stats:
        return samples, skipped, stats
    return samples, skipped


def _build_sft_batch(
    samples,
    *,
    batch_size: int,
    step: int,
    rank: int,
    world_size: int,
    pad_token_id: int,
    device: torch.device,
):
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if not samples:
        raise ValueError("SFT samples are empty.")

    start = step * batch_size * world_size + rank * batch_size
    total = len(samples)
    batch_samples = []
    for offset in range(batch_size):
        sample_idx = (start + offset) % total
        batch_samples.append(samples[sample_idx])

    max_len = max(len(sample["input_ids"]) for sample in batch_samples)
    input_ids = torch.full(
        (batch_size, max_len),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)

    for row, sample in enumerate(batch_samples):
        ids = torch.tensor(sample["input_ids"], dtype=torch.long, device=device)
        lbs = torch.tensor(sample["labels"], dtype=torch.long, device=device)
        sample_len = int(ids.shape[0])
        input_ids[row, :sample_len] = ids
        attention_mask[row, :sample_len] = 1
        labels[row, :sample_len] = lbs

    return input_ids, attention_mask, labels


def _extract_loss(outputs) -> torch.Tensor:
    if hasattr(outputs, "loss") and outputs.loss is not None:
        return outputs.loss
    if isinstance(outputs, dict) and "loss" in outputs:
        return outputs["loss"]
    if isinstance(outputs, tuple) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
        return outputs[0]
    raise RuntimeError("Model outputs do not contain a loss tensor. Ensure labels are passed to the model.")


def _max_across_ranks(value: float, device: torch.device) -> float:
    if not dist.is_initialized():
        return float(value)
    t = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _mean_tensor_across_ranks(value: torch.Tensor, device: torch.device) -> float:
    t = value.detach()
    if t.ndim != 0:
        t = t.mean()
    t = t.to(device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return float(t.item())


def _engine_zero_grad_compat(engine):
    try:
        engine.zero_grad(set_to_none=True)
    except TypeError:
        engine.zero_grad()


def _engine_force_step_compat(engine) -> bool:
    try:
        engine.step(force=True)
        return True
    except TypeError:
        return False


def _init_wandb_if_available(args, *, rank: int, ds_impl: str, ds_cfg: dict, dataset_size: int):
    if rank != 0:
        return None

    try:
        import wandb
    except ImportError:
        print("[wandb] package not found; skip logging.")
        return None

    try:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "ds_impl": ds_impl,
                "model_name": args.model_name,
                "model_dtype": args.model_dtype,
                "steps": int(args.steps),
                "seq_len": int(args.seq_len),
                "batch_size": int(args.batch_size),
                "grad_accum": int(ds_cfg.get("gradient_accumulation_steps", 1)),
                "zero_stage": int((ds_cfg.get("zero_optimization", {}) or {}).get("stage", 0)),
                "dataset_path": str(args.dataset_path),
                "dataset_size": int(dataset_size),
            },
        )
        print(f"[wandb] enabled project={args.wandb_project}")
        return run
    except Exception as exc:
        print(f"[wandb] init failed, skip logging: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser = ds.add_config_arguments(parser)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero-stage", type=int, default=None, choices=[0, 1, 2])
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--model-dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--dataset-path", type=str, default="data/lima.json")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="nano-deepspeed-sft")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.deepspeed_config is None:
        args.deepspeed_config = os.path.join(script_dir, "ds_config_zero2.json")
    elif not os.path.isabs(args.deepspeed_config) and not os.path.exists(args.deepspeed_config):
        args.deepspeed_config = os.path.join(script_dir, args.deepspeed_config)
    if not os.path.isabs(args.dataset_path) and not os.path.exists(args.dataset_path):
        args.dataset_path = os.path.join(script_dir, args.dataset_path)

    dist_required = _dist_required()
    if dist_required:
        ds.init_distributed()

    torch.manual_seed(args.seed)

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = (
        torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    with open(args.deepspeed_config, "r", encoding="utf-8") as f:
        ds_cfg = json.load(f)
    if args.zero_stage is not None:
        ds_cfg.setdefault("zero_optimization", {})
        ds_cfg["zero_optimization"]["stage"] = int(args.zero_stage)

    model, tokenizer, model_dtype, attn_impl, flash_fallback_reason = _load_qwen3(args, device)
    ds_precision_mode = _configure_deepspeed_precision(ds_cfg, model_dtype=model_dtype, device=device)
    param_dtype = next(model.parameters()).dtype
    samples, skipped_samples, sft_stats = _load_sft_samples(
        args.dataset_path,
        tokenizer,
        seq_len=args.seq_len,
        max_samples=args.max_samples,
        return_stats=True,
    )
    if rank == 0:
        ds_file = str(Path(getattr(ds, "__file__", "unknown")).resolve())
        print(f"[deepspeed] impl=official module={ds_file}")
        print(
            f"[model] initialized {args.model_name} from config on {device} "
            f"(param_dtype={param_dtype}, compute_dtype={model_dtype}, ds_precision={ds_precision_mode}, attention={attn_impl})"
        )
        if flash_fallback_reason is not None:
            print(f"[model] flash_attention_2 unavailable, fallback to default attention: {flash_fallback_reason}")
        print(f"[data] path={args.dataset_path} samples={len(samples)} skipped={skipped_samples}")
        print(
            "[data] sft_stats "
            f"items_total={sft_stats['items_total']} "
            f"items_kept={sft_stats['items_kept']} "
            f"messages_kept={sft_stats['messages_kept']} "
            f"assistant_turns={sft_stats['assistant_turns']} "
            f"skipped_empty_full_ids={sft_stats['skipped_empty_full_ids']} "
            f"skipped_all_masked={sft_stats['skipped_all_masked']} "
            f"prompt_alignment_common_prefix={sft_stats['prompt_alignment_common_prefix']}"
        )

    # Avoid passing DeepSpeed config from two sources simultaneously.
    args.deepspeed_config = None
    engine, _, _, _ = ds.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=ds_cfg,
        dist_init_required=dist_required,
    )

    grad_accum = int(ds_cfg.get("gradient_accumulation_steps", 1))
    _engine_zero_grad_compat(engine)
    last_loss = None
    logged_loss = None
    opt_loss_sum = 0.0
    opt_micro_count = 0
    wandb_run = _init_wandb_if_available(
        args,
        rank=rank,
        ds_impl="official",
        ds_cfg=ds_cfg,
        dataset_size=len(samples),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

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

        outputs = engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = _extract_loss(outputs)
        if not bool(torch.isfinite(loss.detach()).all()):
            valid_label_tokens = int((labels != -100).sum().item())
            max_input_id = int(input_ids.max().item()) if input_ids.numel() > 0 else -1
            raise RuntimeError(
                "Non-finite loss detected before backward. "
                f"step={step + 1} valid_label_tokens={valid_label_tokens} max_input_id={max_input_id} "
                f"attention={attn_impl} compute_dtype={model_dtype} ds_precision={ds_precision_mode}"
            )
        reduced_loss = _mean_tensor_across_ranks(loss, device=device)
        last_loss = reduced_loss
        opt_loss_sum += reduced_loss
        opt_micro_count += 1

        engine.backward(loss)
        engine.step()

        if (step + 1) % grad_accum == 0:
            opt_step = (step + 1) // grad_accum
            opt_loss = opt_loss_sum / max(opt_micro_count, 1)
            logged_loss = opt_loss
            if rank == 0:
                print(f"[opt_step {opt_step}] loss={opt_loss:.4f}")
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": opt_loss,
                            "train/opt_step": opt_step,
                            "train/micro_step": step + 1,
                        },
                        step=opt_step,
                    )
            opt_loss_sum = 0.0
            opt_micro_count = 0

    tail_micro = getattr(engine, "_micro_steps", 0) % grad_accum
    if tail_micro != 0:
        did_tail_flush = _engine_force_step_compat(engine)
        if did_tail_flush and opt_micro_count > 0:
            opt_step = (args.steps + grad_accum - 1) // grad_accum
            opt_loss = opt_loss_sum / opt_micro_count
            logged_loss = opt_loss
            if rank == 0:
                print(f"[opt_step {opt_step}] loss={opt_loss:.4f} (tail_flush {tail_micro}/{grad_accum})")
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": opt_loss,
                            "train/opt_step": opt_step,
                            "train/micro_step": args.steps,
                        },
                        step=opt_step,
                    )
            opt_loss_sum = 0.0
            opt_micro_count = 0
        elif rank == 0:
            print("[warn] engine.step(force=True) is not supported by this DeepSpeed version; tail flush skipped.")

    peak_mem_mb = 0.0
    peak_reserved_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = float(torch.cuda.max_memory_allocated(device=device) / (1024.0 * 1024.0))
        peak_reserved_mb = float(torch.cuda.max_memory_reserved(device=device) / (1024.0 * 1024.0))
    peak_mem_mb = _max_across_ranks(peak_mem_mb, device=device)
    peak_reserved_mb = _max_across_ranks(peak_reserved_mb, device=device)
    final_loss = logged_loss if logged_loss is not None else last_loss
    if rank == 0:
        metrics = {
            "ds_impl": "official",
            "steps": int(args.steps),
            "zero_stage": int((ds_cfg.get("zero_optimization", {}) or {}).get("stage", 0)),
            "peak_mem_mb_max_rank": round(peak_mem_mb, 2),
            "peak_reserved_mb_max_rank": round(peak_reserved_mb, 2),
            "dataset_samples": int(len(samples)),
            "skipped_samples": int(skipped_samples),
            "last_loss": None if final_loss is None else float(final_loss),
        }
        print("[metrics] " + json.dumps(metrics, sort_keys=True))
        if wandb_run is not None:
            wandb_run.summary["peak_mem_mb_max_rank"] = round(peak_mem_mb, 2)
            wandb_run.summary["peak_reserved_mb_max_rank"] = round(peak_reserved_mb, 2)
            if final_loss is not None:
                wandb_run.summary["last_loss"] = float(final_loss)
            wandb_run.finish()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
