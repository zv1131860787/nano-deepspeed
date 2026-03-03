import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

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


def _resolve_model_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    name = str(dtype_name).strip().lower()
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name != "auto":
        raise ValueError(f"Unsupported --model-dtype={dtype_name!r}")

    if device.type != "cuda":
        return torch.float32

    bf16_supported = False
    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            bf16_supported = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16_supported = False
    return torch.bfloat16 if bf16_supported else torch.float16


def _load_qwen3(args, device: torch.device):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
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
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer has no pad_token/eos_token; please provide a compatible tokenizer or set one manually."
            )
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        torch_dtype=model_dtype,
    )
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device)
    model.train()
    return model, tokenizer, model_dtype


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
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            return list(token_ids)
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
            return list(encoded["input_ids"])

    chat_text = _render_chat_fallback(messages, add_generation_prompt=add_generation_prompt)
    encoded = tokenizer(
        chat_text,
        truncation=True,
        max_length=seq_len,
        add_special_tokens=True,
    )
    return list(encoded["input_ids"])


def _load_sft_samples(dataset_path: str, tokenizer, *, seq_len: int, max_samples: int = 0):
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(f"Dataset {dataset_path} must be a JSON list.")

    samples = []
    skipped = 0
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
                continue

            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = list(full_ids)
            for pos in range(prompt_len):
                labels[pos] = -100

            if not any(token != -100 for token in labels):
                skipped += 1
                continue

            samples.append({"input_ids": full_ids, "labels": labels})
            if max_samples > 0 and len(samples) >= max_samples:
                return samples, skipped

    if not samples:
        raise RuntimeError(
            f"No valid SFT samples were built from {dataset_path}. "
            "Check dataset schema or increase --seq-len."
        )
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


def _load_nano_deepspeed():
    repo_root = Path(__file__).resolve().parents[1]
    alias = "nano_deepspeed"
    if alias in sys.modules:
        return sys.modules[alias]

    pkg_dir = repo_root / "nano_deepspeed"
    init_py = pkg_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        alias,
        init_py,
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load local package from {init_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


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
    ds = _load_nano_deepspeed()

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
        default="auto",
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

    model, tokenizer, model_dtype = _load_qwen3(args, device)
    samples, skipped_samples = _load_sft_samples(
        args.dataset_path,
        tokenizer,
        seq_len=args.seq_len,
        max_samples=args.max_samples,
    )
    if rank == 0:
        ds_file = str(Path(getattr(ds, "__file__", "unknown")).resolve())
        print(f"[deepspeed] impl=nano module={ds_file}")
        print(f"[model] loaded {args.model_name} on {device} (param_dtype={model_dtype})")
        print(f"[data] path={args.dataset_path} samples={len(samples)} skipped={skipped_samples}")

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
        ds_impl="nano",
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
            "ds_impl": "nano",
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
