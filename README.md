# nano-deepspeed (Teaching Edition)

这是一个**教学版** DeepSpeed 复刻项目，目标是帮助你从源码层面理解 ZeRO 的核心流程，而不是直接替代官方 DeepSpeed 用于生产训练。

项目重点是把 ZeRO-1/2 的关键路径拆清楚：参数扁平化、梯度 hook、分桶通信、分片更新、参数回收（all-gather）。

## 1. 项目定位

- 面向对象：希望深入理解 DeepSpeed ZeRO 机制的学习者/研究者。
- 设计目标：代码尽量短、逻辑可追踪、变量语义清晰。
- 非目标：覆盖官方 DeepSpeed 的全部功能与工程优化。

一句话概括：这是一个“能跑、能对比、能读懂”的教学实现，并且会持续迭代更新。

## 2. 当前已复刻的功能

### 2.1 训练入口与 API 兼容层

- `nano_deepspeed.add_config_arguments(parser)`
- `nano_deepspeed.init_distributed(...)`
- `nano_deepspeed.initialize(...)`
- `nano_deepspeed.zero.Init`（兼容 stub，支持 ZeRO-1/2 场景下的基本进入/退出）

对应文件：

- `nano_deepspeed/api.py`
- `nano_deepspeed/distributed.py`
- `nano_deepspeed/zero/__init__.py`

### 2.2 ZeRO 优化器核心能力

- ZeRO Stage 0 / 1 / 2
- 参数 flatten + 对齐填充（`aligned_total`、`partition_size`）
- AdamW 更新（从 config 构建）
- FP16 动态 loss scaler（含 `loss_scale_window`、`hysteresis`、`min_loss_scale`）
- 梯度裁剪
- rank-local state_dict 保存/加载（stage1/2 会校验 rank/world_size）

对应文件：

- `nano_deepspeed/zero_optimizer.py`
- `nano_deepspeed/fp16_scaler.py`
- `nano_deepspeed/zero_types.py`

### 2.3 ZeRO-1/2 梯度规约器（重点）

- 基于参数 hook 的梯度就绪触发
- 梯度分桶（IPG bucket）+ 异步通信
- stage1 的 partition-aware 路径
- stage2 的分片映射与回填（写入 `grad_partition_fp32`）
- comm stream 与 default stream 的同步（`wait_stream`）
- pending 队列与 oldest finalize 策略（限制异步 in-flight 项）

对应文件：

- `nano_deepspeed/zero_reducer.py`
- `nano_deepspeed/utils.py`

### 2.4 可用于对比实验的示例脚本

推荐使用两个独立入口脚本：

- `examples/train_qwen3_zero12_nano.py`：加载本仓库教学版 `nano_deepspeed`
- `examples/train_qwen3_zero12_official.py`：加载已安装的官方 `deepspeed`

兼容入口：

- `examples/train_qwen3_zero12.py` 仍支持 `--ds-impl nano|official` 切换。

脚本会输出：

- 当前实现来源（module 路径）
- loss 日志
- `peak_mem_mb_max_rank`（多卡下跨 rank 取最大峰值显存）

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
    ├── train_qwen3_zero12.py
    ├── train_qwen3_zero12_nano.py
    └── train_qwen3_zero12_official.py
```

## 4. 环境要求

- Python 3.9+
- PyTorch（建议 CUDA 版本）
- transformers（用于 Qwen 示例）
- 官方 DeepSpeed（仅在你要跑 `--ds-impl official` 时需要）

示例安装：

```bash
pip install torch transformers
pip install deepspeed
```

说明：

- 如果你只测试教学版 `nano`，理论上不强依赖官方 `deepspeed` 包。
- 如果你要做“官方 vs 教学版”对比，需安装官方 `deepspeed`。

## 5. 快速开始

### 5.1 单卡快速跑通（教学版）

```bash
python3 examples/train_qwen3_zero12_nano.py \
  --deepspeed_config examples/ds_config_zero2.json \
  --steps 5 \
  --batch-size 1 \
  --seq-len 128
```

### 5.2 多卡运行（推荐用 torchrun）

```bash
torchrun --standalone --nproc_per_node=2 examples/train_qwen3_zero12_nano.py \
  --deepspeed_config examples/ds_config_zero2.json \
  --steps 20
```

## 6. 显存峰值对比（官方 vs 教学版）

确保两次实验使用**相同**模型、batch、seq_len、steps、zero config。

### 6.1 跑教学版

```bash
torchrun --standalone --nproc_per_node=2 examples/train_qwen3_zero12_nano.py \
  --deepspeed_config examples/ds_config_zero2.json \
  --steps 20
```

### 6.2 跑官方版

```bash
torchrun --standalone --nproc_per_node=2 examples/train_qwen3_zero12_official.py \
  --deepspeed_config examples/ds_config_zero2.json \
  --steps 20
```

### 6.3 看结果

关注日志中的 `[metrics]` 行，例如：

```text
[metrics] {"ds_impl":"nano","last_loss":...,"peak_mem_mb_max_rank":...,"steps":20,"zero_stage":2}
```

对比 `peak_mem_mb_max_rank` 即可。

## 7. 教学实现的主线心智模型

你可以按这条主线阅读 ZeRO reducer：

`ready -> consume -> bucket -> launch -> pending -> finalize -> reset`

对应含义：

1. `ready`：参数梯度就绪（hook 触发）
2. `consume`：读取/整理该参数梯度并清空 `p.grad`
3. `bucket`：写入活动桶，桶满则 flush
4. `launch`：发起通信（all-reduce 路径）
5. `pending`：异步任务入队等待 finalize
6. `finalize`：等待通信完成并把本 rank 需要的片段回填
7. `reset`：清空标记，准备下一轮 backward

## 8. 配置说明（当前项目常用）

`examples/ds_config_zero2.json` 中常见项：

- `zero_optimization.stage`：`0/1/2`
- `overlap_comm`：是否开启通信重叠
- `reduce_scatter`：是否走分片感知路径（教学实现里用于策略开关）
- `reduce_bucket_size`：分桶大小
- `communication_data_type`：通信 dtype（fp16/fp32/bf16）
- `stage1_partition_aware_grad_reduce`：stage1 是否按分片感知规约
- `ignore_unused_parameters`：是否把未产 grad 参数按 0 处理

## 9. 与官方 DeepSpeed 的差距（重点）

下面这些是当前教学版和官方之间的关键差距，也是你在结果解读时必须考虑的点。

### 9.1 功能覆盖差距

- 仅支持 ZeRO-0/1/2，不支持 ZeRO-3。
- 优化器仅实现 `AdamW`。
- 大量官方生态能力未覆盖：offload、AIO、activation checkpoint 全家桶集成、MoE、pipeline/tensor 并行全链路等。

### 9.2 通信实现差距

- stage2 目前核心路径是 `packed + all_reduce + 本地回填` 的教学实现。
- 官方在 stage2 上有更成熟的 `reduce_scatter/allgather` 组合与大量细节优化。
- 因此峰值显存和吞吐可能“接近但不等同”官方，不应直接视为官方行为复现。

### 9.3 工程化差距

- 错误处理、可观测性、容错、性能调优、不同硬件后端适配都比官方简化很多。
- 在极端配置（超大模型、复杂并行组合、长时训练）下稳定性与性能上限不等同官方。

## 10. 如何正确使用这个项目

推荐使用方式：

- 用它理解 ZeRO 内核机制与数据流。
- 用它做小规模实验、可解释性验证、单点逻辑调试。
- 用它和官方做“定性 + 小规模定量”对比。

不推荐直接作为生产训练替代品。

## 11. 常见问题

### Q1：为什么官方脚本可能导入失败？

确保环境里已安装官方 `deepspeed`，并确认没有把其他同名本地包放到 `PYTHONPATH` 前面。若你使用兼容入口 `train_qwen3_zero12.py --ds-impl official`，同样要避免同名覆盖。

### Q2：为什么会报 `Unused parameter detected during ZeRO gradient reduction`？

说明某些参数在这轮 backward 没有梯度。如果这是预期行为，可在 zero 配置里设：

```json
"ignore_unused_parameters": true
```

### Q3：为什么我看到的显存峰值和官方不一致？

这是正常现象。教学版的通信与调度策略是“可读性优先”，不是“完全工程等价”。

## 12. 许可与声明

本项目用于学习与研究目的。若用于生产，请优先评估并使用官方 DeepSpeed。

## 13. 持续更新计划

本项目会持续更新，当前重点方向如下：

- 补齐 ZeRO-3 核心训练路径（参数分片生命周期、通信与状态管理）。
- 增加 offload 能力（优先考虑 optimizer/parameter offload 的教学实现）。

说明：

- 在上述能力稳定前，仓库仍以“教学可解释性优先”为原则推进，而不是追求与官方完全工程等价。

## 14. 版本路线图（Roadmap）

### v0.2（当前阶段后的首个里程碑）

- 目标：
- 把 ZeRO-2 路径进一步稳定，完善教学文档与对比基线。

- 计划交付：
- stage2 通信与 finalize 路径补充更多注释与验证点。
- 增加显存/吞吐对比脚本模板（nano vs official）。
- README 增补“常见配置组合 + 预期行为”章节。

- 完成判据：
- 在固定模型和配置下，能够稳定复现实验并输出可比较指标。

### v0.3（ZeRO-3 教学版首发）

- 目标：
- 提供可运行、可讲解的 ZeRO-3 最小实现主线。

- 计划交付：
- 参数分片持有与按需 gather 的核心路径。
- forward/backward/step 阶段的参数生命周期管理。
- ZeRO-3 基础配置开关与最小示例脚本。

- 完成判据：
- 小模型场景可稳定训练，代码路径可被文档完整串讲。

### v0.4（Offload 教学版首发）

- 目标：
- 提供 offload 机制的教学实现与性能/显存权衡示例。

- 计划交付：
- optimizer state offload（优先）。
- parameter offload（次优先，先最小可运行版本）。
- 关键数据搬运路径与同步点说明（CPU/GPU）。

- 完成判据：
- 在显存受限场景可运行训练，并能清晰展示“显存下降 vs 性能开销”。

### v0.5（收敛与对齐增强）

- 目标：
- 在保持教学可读性的前提下，提升行为稳定性与可对比性。

- 计划交付：
- 更完善的测试样例（stage0/1/2/3 与 offload 关键路径）。
- checkpoint 与恢复流程增强。
- 文档补齐“与官方差异矩阵（能力、性能、工程化）”。

- 完成判据：
- 关键路径具备可重复验证结果，文档与实现版本一致。
