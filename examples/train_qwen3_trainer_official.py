import argparse
import json
import os
from typing import Optional

import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def _normalize_role(raw_role: str):
    role = str(raw_role).strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "model"}:
        return "assistant"
    if role == "system":
        return "system"
    return None


def _render_chat_fallback(messages) -> str:
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
    return "\n".join(lines)


def _render_chat_text(tokenizer, messages) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if isinstance(text, str):
                return text
        except TypeError:
            pass
    return _render_chat_fallback(messages)


def _load_sft_text_dataset(dataset_path: str, tokenizer, *, max_samples: int = 0):
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(f"Dataset {dataset_path} must be a JSON list.")

    rows = []
    skipped = 0
    for item in raw_data:
        if not isinstance(item, dict):
            skipped += 1
            continue
        turns = item.get("conversations")
        if not isinstance(turns, list):
            skipped += 1
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

        # keep only samples with at least one assistant target turn
        has_target = any(msg["role"] == "assistant" for msg in messages[1:])
        if not has_target:
            skipped += 1
            continue

        text = _render_chat_text(tokenizer, messages)
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue

        rows.append({"text": text})
        if max_samples > 0 and len(rows) >= max_samples:
            break

    if not rows:
        raise RuntimeError(
            f"No valid samples were built from {dataset_path}. "
            "Check dataset schema."
        )

    return Dataset.from_list(rows), skipped


def _resolve_precision(precision: str, device: torch.device):
    name = str(precision).strip().lower()
    if name not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"Unsupported --precision={precision!r}")
    if name == "bf16" and device.type == "cuda":
        supported = False
        if hasattr(torch.cuda, "is_bf16_supported"):
            try:
                supported = bool(torch.cuda.is_bf16_supported())
            except Exception:
                supported = False
        if not supported:
            raise RuntimeError("Requested --precision=bf16 but current CUDA device does not support bf16.")
    if name in {"bf16", "fp16"} and device.type != "cuda":
        raise RuntimeError(f"Requested --precision={name} but CUDA is not available.")
    return name


def _build_model(args, device: torch.device):
    model_kwargs = {}
    if args.local_files_only:
        model_kwargs["local_files_only"] = True
    model_kwargs["trust_remote_code"] = args.trust_remote_code

    def _create(attn_impl: Optional[str]):
        create_kwargs = dict(model_kwargs)
        if attn_impl:
            create_kwargs["attn_implementation"] = attn_impl

        if args.init_mode == "config":
            cfg = AutoConfig.from_pretrained(args.model_name, **model_kwargs)
            return AutoModelForCausalLM.from_config(cfg, **create_kwargs)
        return AutoModelForCausalLM.from_pretrained(args.model_name, **create_kwargs)

    attn_impl = "default"
    flash_fallback_reason = None
    use_flash = device.type == "cuda" and (not args.no_flash_attention)
    if use_flash:
        try:
            model = _create("flash_attention_2")
            attn_impl = "flash_attention_2"
        except Exception as exc:
            flash_fallback_reason = str(exc).strip().splitlines()[0]
            model = _create(None)
    else:
        model = _create(None)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, attn_impl, flash_fallback_reason


def _json_safe_metrics(metrics: dict):
    safe = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            safe[k] = v
        else:
            try:
                safe[k] = float(v)
            except Exception:
                safe[k] = str(v)
    return safe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspeed-config", type=str, default="ds_config_zero2_official.json")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset-path", type=str, default="data/lima.json")
    parser.add_argument("--output-dir", type=str, default="outputs/qwen3_trainer_ds")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--init-mode", type=str, default="config", choices=["config", "pretrained"])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-flash-attention", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="nano-deepspeed-sft")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.deepspeed_config) and not os.path.exists(args.deepspeed_config):
        args.deepspeed_config = os.path.join(script_dir, args.deepspeed_config)
    if not os.path.isabs(args.dataset_path) and not os.path.exists(args.dataset_path):
        args.dataset_path = os.path.join(script_dir, args.dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = _resolve_precision(args.precision, device)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has no pad_token/eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    model, attn_impl, flash_fallback_reason = _build_model(args, device)
    param_dtype = next(model.parameters()).dtype

    text_ds, skipped_samples = _load_sft_text_dataset(
        args.dataset_path,
        tokenizer,
        max_samples=args.max_samples,
    )

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.seq_len,
            add_special_tokens=True,
        )

    tokenized = text_ds.map(
        _tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)
    if len(tokenized) == 0:
        raise RuntimeError("Tokenized dataset is empty after filtering.")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    report_to = []
    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        report_to = ["wandb"]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="no",
        evaluation_strategy="no",
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
        deepspeed=args.deepspeed_config,
        report_to=report_to,
        run_name=args.wandb_run_name,
        optim="adamw_torch",
    )

    if training_args.world_size > 1:
        print(f"[dist] world_size={training_args.world_size}")
    print(
        f"[model] mode={args.init_mode} name={args.model_name} param_dtype={param_dtype} "
        f"precision={precision} attention={attn_impl}"
    )
    if flash_fallback_reason is not None:
        print(f"[model] flash_attention_2 unavailable, fallback to default attention: {flash_fallback_reason}")
    print(f"[data] path={args.dataset_path} samples={len(tokenized)} skipped={skipped_samples}")
    print(f"[deepspeed] config={args.deepspeed_config}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    train_result = trainer.train()
    metrics = _json_safe_metrics(train_result.metrics)
    metrics["dataset_samples"] = int(len(tokenized))
    metrics["skipped_samples"] = int(skipped_samples)
    print("[metrics] " + json.dumps(metrics, ensure_ascii=True, sort_keys=True))
    trainer.save_state()


if __name__ == "__main__":
    main()
