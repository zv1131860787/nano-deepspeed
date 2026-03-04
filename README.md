# nano-deepspeed (Teaching Edition)

English is the default documentation for this repository.

For Chinese documentation, see [README.zh-CN.md](README.zh-CN.md).

This project is a teaching-oriented re-implementation of DeepSpeed ZeRO. The goal is to help you understand ZeRO data flow and communication behavior, not to replace official DeepSpeed in production.

## 1. Project Scope

- Positioning: teaching, source-code reading, mechanism verification.
- Goal: readable code, explainable behavior, and easy small-scale comparison with official DeepSpeed.
- Non-goal: full feature parity and full engineering optimization of official DeepSpeed.

In one sentence: a ZeRO teaching implementation that you can run, read, and compare.

## 2. Current Feature Range

- Supported ZeRO stages: `0/1/2`
- Not supported: `stage 3`
- Optimizer: `AdamW` only
- Precision: FP16 dynamic loss scaling and bf16/autocast path
- Communication: stage2 teaching implementation (`packed + all_reduce + local scatter-back`)
- API compatibility entry points:
  - `nano_deepspeed.initialize(...)`
  - `init_distributed(...)`
  - `add_config_arguments(...)`

Core files:

- `nano_deepspeed/api.py`
- `nano_deepspeed/engine.py`
- `nano_deepspeed/zero_optimizer.py`
- `nano_deepspeed/zero_reducer.py`
- `nano_deepspeed/fp16_scaler.py`

## 3. Repository Layout

```text
.
├── nano_deepspeed/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── distributed.py
│   ├── engine.py
│   ├── fp16_scaler.py
│   ├── utils.py
│   ├── zero_optimizer.py
│   ├── zero_reducer.py
│   ├── zero_types.py
│   └── zero/
│       └── __init__.py
└── examples/
    ├── ds_config_zero2.json
    ├── ds_config_zero2_official.json
    ├── train_qwen3_zero12_nano.py
    └── train_qwen3_zero12_official.py
```

## 4. Requirements

- Python 3.9+
- PyTorch (CUDA build recommended)
- transformers (for Qwen example)
- official DeepSpeed (required only for `train_qwen3_zero12_official.py`)

Example installation:

```bash
pip install torch transformers
pip install deepspeed
```

## 5. Scripts and Configs

Training scripts:

- `examples/train_qwen3_zero12_nano.py`: runs this repo's `nano_deepspeed`
- `examples/train_qwen3_zero12_official.py`: runs pip-installed official `deepspeed`

Config files:

- `examples/ds_config_zero2.json`: default teaching config for nano (contains nano-specific fields)
- `examples/ds_config_zero2_official.json`: official-compatible config (nano-only fields removed)

Important:

- Use `ds_config_zero2_official.json` for the official script, otherwise you may hit `ValidationError: extra_forbidden`.

## 6. Quick Start

Single GPU (nano):

```bash
python3 examples/train_qwen3_zero12_nano.py \
  --deepspeed_config examples/ds_config_zero2.json \
  --steps 50 --batch-size 1 --seq-len 512
```

2 GPUs (nano):

```bash
torchrun --standalone --nproc_per_node=2 examples/train_qwen3_zero12_nano.py \
  --deepspeed_config examples/ds_config_zero2.json \
  --steps 50 --seq-len 512
```

2 GPUs (official):

```bash
torchrun --standalone --nproc_per_node=2 examples/train_qwen3_zero12_official.py \
  --deepspeed_config examples/ds_config_zero2_official.json \
  --steps 50 --seq-len 512
```

## 7. Recommended Side-by-Side Commands

```bash
export CUDA_VISIBLE_DEVICES=0,1
NANO_CONF=examples/ds_config_zero2.json
OFFICIAL_CONF=examples/ds_config_zero2_official.json
MODEL=/root/autodl-tmp/pretrained_models/Qwen3-0.6B

# Optional warmup to reduce first-time compile/JIT effects
torchrun --standalone --nproc_per_node=2 examples/train_qwen3_zero12_nano.py \
  --deepspeed_config $NANO_CONF --model-name $MODEL --model-dtype bfloat16 --seed 42 --steps 2 >/dev/null

torchrun --standalone --nproc_per_node=2 examples/train_qwen3_zero12_official.py \
  --deepspeed_config $OFFICIAL_CONF --model-name $MODEL --model-dtype bfloat16 --seed 42 --steps 2 >/dev/null

# Main run
torchrun --standalone --nproc_per_node=2 examples/train_qwen3_zero12_nano.py \
  --deepspeed_config $NANO_CONF --model-name $MODEL --model-dtype bfloat16 --seed 42 \
  --batch-size 1 --seq-len 512 --steps 50 | tee nano.log

torchrun --standalone --nproc_per_node=2 examples/train_qwen3_zero12_official.py \
  --deepspeed_config $OFFICIAL_CONF --model-name $MODEL --model-dtype bfloat16 --seed 42 \
  --batch-size 1 --seq-len 512 --steps 50 | tee official.log

grep "\\[opt_step" nano.log official.log
```

Before 8-GPU runs, verify visible GPU count first:

```bash
python3 - <<'PY'
import torch
print(torch.cuda.device_count())
PY
```

`--nproc_per_node` must be less than or equal to visible GPU count, otherwise `invalid device ordinal` will occur.

## 8. Log Field Reference

The scripts currently output these key fields:

- `[zero]`: ZeRO runtime config (`stage/reduce_scatter/bucket/communication_data_type`)
- `[model]`: precision info (`param_dtype/compute_dtype/ds_precision`)
- `[opt_step k]`: training stats (`loss/cuda_alloc_max_mb/cuda_reserved_max_mb/cuda_peak_alloc_max_mb/cuda_peak_reserved_max_mb`)

Memory relationship:

- Usually `reserved >= allocated`
- `reserved - allocated` approximates memory kept in CUDA caching allocator but not actively occupied by tensors

## 9. Experimental Results (from `log/`, 2026-03-04)

Source logs:

- `log/nano-node2.log`
- `log/official-node2.log`
- `log/nano-node8.log`
- `log/official-node8.log`

Shared conditions (read from log headers):

- GPU: NVIDIA GeForce RTX 4090 D
- `param_dtype=torch.bfloat16`
- `compute_dtype=torch.bfloat16`
- `opt_step=50` in logs (equivalent to 100 micro steps with `gradient_accumulation_steps=2`)
- Config differences in this run:
  - nano: `reduce_scatter=False` + `communication_data_type=bf16`
  - official: `reduce_scatter=True` + `communication_data_type=auto`

2 GPUs (node2):

| impl | last_loss | alloc_max_mb | reserved_max_mb | peak_alloc_max_mb | peak_reserved_max_mb |
|---|---:|---:|---:|---:|---:|
| nano | 2.7722 | 7337.0 | 13718.0 | 10264.2 | 13718.0 |
| official | 2.7738 | 5901.5 | 12750.0 | 9377.1 | 12750.0 |

2-GPU delta (nano - official):

- `alloc_max_mb`: `+1435.5 MB` (about `+24.3%`)
- `peak_alloc_max_mb`: `+887.1 MB` (about `+9.46%`)
- `reserved_max_mb`: `+968.0 MB` (about `+7.59%`)
- `last_loss` delta: `-0.0016`

8 GPUs (node8):

| impl | last_loss | alloc_max_mb | reserved_max_mb | peak_alloc_max_mb | peak_reserved_max_mb |
|---|---:|---:|---:|---:|---:|
| nano | 2.6393 | 3034.2 | 8376.0 | 5724.8 | 8376.0 |
| official | 2.6398 | 2673.9 | 7104.0 | 5663.4 | 7104.0 |

8-GPU delta (nano - official):

- `alloc_max_mb`: `+360.3 MB` (about `+13.5%`)
- `peak_alloc_max_mb`: `+61.4 MB` (about `+1.08%`)
- `reserved_max_mb`: `+1272.0 MB` (about `+17.9%`)
- `last_loss` delta: `-0.0005`

Conclusions:

- All four runs completed stably (no `warn/non-finite/traceback`).
- Under this configuration, nano still uses more memory (`allocated` and `reserved`) than official.
- Final losses are very close, indicating both runs are stable and in a similar convergence band.

## 10. Why Official Uses Less Memory

Main reasons are implementation strategy differences:

- nano stage2 is still a teaching path, not the full official engineering path
- temporary buffer and communication packing are more conservative, which increases peaks
- official implementation includes more optimization in communication, memory reuse, and scheduling

## 11. Gaps vs Official DeepSpeed

- No ZeRO-3
- No offload (optimizer/parameter)
- Missing many official ecosystem capabilities (MoE, full pipeline/tensor-parallel integration, AIO, etc.)
- Simpler engineering robustness (fault tolerance, performance tuning, extreme-scale stability)

## 12. Roadmap

- v0.2: keep improving ZeRO-2 path, add observability and comparison tooling
- v0.3: first release of teaching ZeRO-3 minimal runnable path
- v0.4: first release of teaching offload path
- v0.5: improve tests, checkpoint consistency, and documentation consistency

## 13. Common Errors

`ValidationError: extra_forbidden` (official script):

- Cause: using nano-only fields in config
- Fix: use `examples/ds_config_zero2_official.json`

`CUDA error: invalid device ordinal`:

- Cause: `--nproc_per_node` is larger than visible GPU count
- Fix: reduce `--nproc_per_node` or set `CUDA_VISIBLE_DEVICES` correctly

`DeepSpeedEngine.zero_grad() got an unexpected keyword argument 'set_to_none'`:

- Cause: API differences across official DeepSpeed versions
- Current script already has compatibility fallback

## 14. Disclaimer

This project is for learning and research. For production workloads, use official DeepSpeed first.
