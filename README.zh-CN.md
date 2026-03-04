# nano-deepspeed (Teaching Edition)

这是一个教学版 DeepSpeed 复刻项目，目标是帮助你理解 ZeRO 训练的核心数据流和通信流程，而不是替代官方 DeepSpeed 做生产训练。

## 1. 项目定位

- 定位：教学、源码阅读、机制验证。
- 目标：代码可读、行为可解释、便于和官方做小规模对比。
- 非目标：覆盖官方 DeepSpeed 全部能力和工程优化。

一句话：这是一个“能跑、能看懂、能对比”的 ZeRO 教学实现。

## 2. 当前功能范围

- 支持 ZeRO stage：`0/1/2`
- 不支持：`stage 3`
- 优化器：仅 `AdamW`
- 精度：支持 FP16 动态 loss scaler；支持 bf16/autocast 路径
- 通信：stage2 为教学实现（`packed + all_reduce + local scatter-back`）
- 入口兼容：提供 `nano_deepspeed.initialize(...)`、`init_distributed(...)`、`add_config_arguments(...)`

核心代码：

- `nano_deepspeed/api.py`
- `nano_deepspeed/engine.py`
- `nano_deepspeed/zero_optimizer.py`
- `nano_deepspeed/zero_reducer.py`
- `nano_deepspeed/fp16_scaler.py`

## 3. 目录结构

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

## 4. 环境要求

- Python 3.9+
- PyTorch（建议 CUDA 版本）
- transformers（用于 Qwen 示例）
- datasets（用于 `Trainer` 极简脚本）
- 官方 DeepSpeed（仅在运行 `train_qwen3_zero12_official.py` 时需要）

示例安装：

```bash
pip install torch transformers
pip install deepspeed
```

## 5. 脚本与配置

训练脚本：

- `examples/train_qwen3_zero12_nano.py`：跑本仓库 `nano_deepspeed`
- `examples/train_qwen3_zero12_official.py`：跑 pip 安装的官方 `deepspeed`

配置文件：

- `examples/ds_config_zero2.json`：nano 默认教学配置（含 nano 自定义字段）
- `examples/ds_config_zero2_official.json`：官方可接受配置（去掉官方不接受的字段）

重要说明：

- 官方脚本请优先传 `ds_config_zero2_official.json`，否则可能报 `ValidationError: extra_forbidden`。

## 6. 快速开始

单卡（nano）：

```bash
python3 examples/train_qwen3_zero12_nano.py \
  --deepspeed_config examples/ds_config_zero2.json \
  --steps 50 --batch-size 1 --seq-len 512
```

2 卡（nano）：

```bash
torchrun --standalone --nproc_per_node=2 examples/train_qwen3_zero12_nano.py \
  --deepspeed_config examples/ds_config_zero2.json \
  --steps 50 --seq-len 512
```

2 卡（official）：

```bash
torchrun --standalone --nproc_per_node=2 examples/train_qwen3_zero12_official.py \
  --deepspeed_config examples/ds_config_zero2_official.json \
  --steps 50 --seq-len 512
```

## 7. 严格对齐对比命令（推荐）

```bash
export CUDA_VISIBLE_DEVICES=0,1
NANO_CONF=examples/ds_config_zero2.json
OFFICIAL_CONF=examples/ds_config_zero2_official.json
MODEL=/root/autodl-tmp/pretrained_models/Qwen3-0.6B

# 可选 warmup，减少首次编译/JIT 影响
torchrun --standalone --nproc_per_node=2 examples/train_qwen3_zero12_nano.py \
  --deepspeed_config $NANO_CONF --model-name $MODEL --model-dtype bfloat16 --seed 42 --steps 2 >/dev/null

torchrun --standalone --nproc_per_node=2 examples/train_qwen3_zero12_official.py \
  --deepspeed_config $OFFICIAL_CONF --model-name $MODEL --model-dtype bfloat16 --seed 42 --steps 2 >/dev/null

# 正式对比
torchrun --standalone --nproc_per_node=2 examples/train_qwen3_zero12_nano.py \
  --deepspeed_config $NANO_CONF --model-name $MODEL --model-dtype bfloat16 --seed 42 \
  --batch-size 1 --seq-len 512 --steps 50 | tee nano.log

torchrun --standalone --nproc_per_node=2 examples/train_qwen3_zero12_official.py \
  --deepspeed_config $OFFICIAL_CONF --model-name $MODEL --model-dtype bfloat16 --seed 42 \
  --batch-size 1 --seq-len 512 --steps 50 | tee official.log

grep "\\[metrics\\]" nano.log official.log
```

8 卡运行前建议先确认可见卡数：

```bash
python3 - <<'PY'
import torch
print(torch.cuda.device_count())
PY
```

`--nproc_per_node` 必须小于等于可见 GPU 数，否则会报 `invalid device ordinal`。

## 8. 指标解释

当前脚本会输出以下关键信息：

- `[zero]`：ZeRO 关键配置（`stage/reduce_scatter/bucket/communication_data_type`）
- `[model]`：模型精度信息（`param_dtype/compute_dtype/ds_precision`）
- `[opt_step k]`：训练指标（`loss/cuda_alloc_max_mb/cuda_reserved_max_mb/cuda_peak_alloc_max_mb/cuda_peak_reserved_max_mb`）

关系：

- 一般 `reserved >= allocated`
- `reserved - allocated` 近似是缓存池中暂未被 tensor 使用但未归还驱动的空间

## 9. 实测结果（来自 `log/`，2026-03-04）

日志文件：

- `log/nano-node2.log`
- `log/official-node2.log`
- `log/nano-node8.log`
- `log/official-node8.log`

共同条件（从日志头部读取）：

- GPU: NVIDIA GeForce RTX 4090 D
- `param_dtype=torch.bfloat16`
- `compute_dtype=torch.bfloat16`
- 日志共 `opt_step=50`（对应 micro step 100，`gradient_accumulation_steps=2`）
- 本轮配置差异：nano 使用 `reduce_scatter=False` + `communication_data_type=bf16`；official 使用 `reduce_scatter=True` + `communication_data_type=auto`

2 卡（node2）：

| impl | last_loss | alloc_max_mb | reserved_max_mb | peak_alloc_max_mb | peak_reserved_max_mb |
|---|---:|---:|---:|---:|---:|
| nano | 2.7722 | 7337.0 | 13718.0 | 10264.2 | 13718.0 |
| official | 2.7738 | 5901.5 | 12750.0 | 9377.1 | 12750.0 |

2 卡差值（nano - official）：

- `alloc_max_mb`：`+1435.5 MB`（约 `+24.3%`）
- `peak_alloc_max_mb`：`+887.1 MB`（约 `+9.46%`）
- `reserved_max_mb`：`+968.0 MB`（约 `+7.59%`）
- `last_loss` 差值：`-0.0016`

8 卡（node8）：

| impl | last_loss | alloc_max_mb | reserved_max_mb | peak_alloc_max_mb | peak_reserved_max_mb |
|---|---:|---:|---:|---:|---:|
| nano | 2.6393 | 3034.2 | 8376.0 | 5724.8 | 8376.0 |
| official | 2.6398 | 2673.9 | 7104.0 | 5663.4 | 7104.0 |

8 卡差值（nano - official）：

- `alloc_max_mb`：`+360.3 MB`（约 `+13.5%`）
- `peak_alloc_max_mb`：`+61.4 MB`（约 `+1.08%`）
- `reserved_max_mb`：`+1272.0 MB`（约 `+17.9%`）
- `last_loss` 差值：`-0.0005`

结论：

- 四份日志都稳定完成（未出现 `warn/non-finite/traceback`）。
- 在本轮配置下，nano 的 `allocated` 和 `reserved` 仍高于 official。
- `last_loss` 基本一致，说明两者都能稳定训练并收敛到相近区间。

## 10. 为什么官方显存更低

主要原因是实现策略差异：

- nano 的 stage2 当前是教学路径，不是官方完整工程路径
- 临时缓冲与通信打包策略更保守，峰值更容易抬高
- 官方在通信、内存复用、调度上有更多工程优化

## 11. 与官方 DeepSpeed 的差距

- 不支持 ZeRO-3
- 不支持 offload（optimizer/parameter）
- 缺少大量官方生态能力（MoE、pipeline/tensor 并行全链路、AIO 等）
- 工程化能力简化（容错、性能调优、极端规模稳定性）

## 12. 路线图

- v0.2：ZeRO-2 路径继续收敛，补齐可观测性与对比工具
- v0.3：ZeRO-3 教学版首发（最小可运行主线）
- v0.4：offload 教学版首发
- v0.5：测试、checkpoint、文档一致性与差异矩阵收敛

## 13. 常见报错

`ValidationError: extra_forbidden`（官方脚本）：

- 原因：用了 nano 配置里的自定义字段
- 解决：使用 `examples/ds_config_zero2_official.json`

`CUDA error: invalid device ordinal`：

- 原因：`--nproc_per_node` 大于可见 GPU 数
- 解决：把 `--nproc_per_node` 调整为可见卡数，或正确设置 `CUDA_VISIBLE_DEVICES`

`DeepSpeedEngine.zero_grad() got an unexpected keyword argument 'set_to_none'`：

- 这是不同官方版本 API 差异导致
- 当前 `train_qwen3_zero12_official.py` 已做兼容回退处理

## 14. 声明

本项目用于学习与研究。生产环境请优先使用官方 DeepSpeed。
