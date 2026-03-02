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


def _build_batch(tokenizer, *, batch_size: int, seq_len: int, step: int, rank: int, device: torch.device, prompt: str):
    prompts = [
        f"{prompt}\n[rank={rank} step={step} sample={i}]"
        for i in range(batch_size)
    ]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    labels = input_ids.clone()
    if attention_mask is not None:
        labels.masked_fill_(attention_mask == 0, -100)

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
    parser.add_argument("--prompt", type=str, default="Write a short note about ZeRO optimizer states.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.deepspeed_config is None:
        args.deepspeed_config = os.path.join(script_dir, "ds_config_zero2.json")
    elif not os.path.isabs(args.deepspeed_config) and not os.path.exists(args.deepspeed_config):
        args.deepspeed_config = os.path.join(script_dir, args.deepspeed_config)

    dist_required = _dist_required()
    if dist_required:
        ds.init_distributed()

    torch.manual_seed(args.seed)

    rank = dist.get_rank() if dist.is_initialized() else 0
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
    if rank == 0:
        ds_file = str(Path(getattr(ds, "__file__", "unknown")).resolve())
        print(f"[deepspeed] impl=nano module={ds_file}")
        print(f"[model] loaded {args.model_name} on {device} (param_dtype={model_dtype})")

    engine, _, _, _ = ds.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=ds_cfg,
        dist_init_required=dist_required,
    )

    data_seed = int(args.seed) + 1000 + int(rank)
    torch.manual_seed(data_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(data_seed)

    grad_accum = int(ds_cfg.get("gradient_accumulation_steps", 1))
    engine.zero_grad(set_to_none=True)
    last_loss = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    for step in range(args.steps):
        input_ids, attention_mask, labels = _build_batch(
            tokenizer,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            step=step,
            rank=rank,
            device=device,
            prompt=args.prompt,
        )

        outputs = engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = _extract_loss(outputs)
        last_loss = loss

        engine.backward(loss)
        engine.step()

        if rank == 0 and ((step + 1) % grad_accum == 0):
            print(f"[opt_step {(step + 1) // grad_accum}] loss={loss.item():.4f}")

    tail_micro = getattr(engine, "_micro_steps", 0) % grad_accum
    if tail_micro != 0:
        engine.step(force=True)
        if rank == 0 and last_loss is not None:
            opt_step = (args.steps + grad_accum - 1) // grad_accum
            print(f"[opt_step {opt_step}] loss={last_loss.item():.4f} (tail_flush {tail_micro}/{grad_accum})")

    peak_mem_mb = 0.0
    peak_reserved_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = float(torch.cuda.max_memory_allocated(device=device) / (1024.0 * 1024.0))
        peak_reserved_mb = float(torch.cuda.max_memory_reserved(device=device) / (1024.0 * 1024.0))
    peak_mem_mb = _max_across_ranks(peak_mem_mb, device=device)
    peak_reserved_mb = _max_across_ranks(peak_reserved_mb, device=device)
    if rank == 0:
        metrics = {
            "ds_impl": "nano",
            "steps": int(args.steps),
            "zero_stage": int((ds_cfg.get("zero_optimization", {}) or {}).get("stage", 0)),
            "peak_mem_mb_max_rank": round(peak_mem_mb, 2),
            "peak_reserved_mb_max_rank": round(peak_reserved_mb, 2),
            "last_loss": None if last_loss is None else float(last_loss.item()),
        }
        print("[metrics] " + json.dumps(metrics, sort_keys=True))

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
