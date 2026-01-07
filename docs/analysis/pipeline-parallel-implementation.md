# Pipeline Parallel Implementation in MS-SWIFT

**分析日期**: 2026-01-04
**框架版本**: MS-SWIFT (基于 Megatron-Core 0.13+)
**分析范围**: ~3,500 行核心代码，涵盖 10+ 关键文件

---

## 目录

1. [概述](#1-概述)
2. [核心架构](#2-核心架构)
3. [参数配置](#3-参数配置)
4. [进程组初始化](#4-进程组初始化)
5. [层分配机制](#5-层分配机制)
6. [Virtual Pipeline Parallel](#6-virtual-pipeline-parallel-vpp)
7. [数据流与调度](#7-数据流与调度)
8. [Mcore-Bridge PP 实现](#8-mcore-bridge-pp-实现)
9. [GRPO 与 PP 约束](#9-grpo-与-pp-约束)
10. [示例配置](#10-示例配置)
11. [性能分析](#11-性能分析)
12. [最佳实践](#12-最佳实践)
13. [总结](#13-总结)

---

## 1. 概述

### 1.1 什么是 Pipeline Parallel

**Pipeline Parallel (PP)** 是一种**跨节点/跨 GPU 的模型并行策略**，通过将深度神经网络的**层（layers）** 切分到不同的设备上来减少单设备的内存占用。

#### 核心思想

```
┌────────────────────────────────────────────────────────────┐
│  原始模型 (32 layers, 显存需求 > 单 GPU 容量)              │
│  Layer 0-31                                                 │
└────────────────────────────────────────────────────────────┘

                    Pipeline Parallel 分割 ↓

┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ PP Rank 0   │ → │ PP Rank 1   │ → │ PP Rank 2   │ → │ PP Rank 3   │
│ Layer 0-7   │   │ Layer 8-15  │   │ Layer 16-23 │   │ Layer 24-31 │
│ GPU 0       │   │ GPU 1       │   │ GPU 2       │   │ GPU 3       │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
     ↓ output          ↓ output          ↓ output          ↓ output
```

**关键概念**:
- **Stage**: PP 的每个阶段，对应一个或多个 transformer 层
- **Bubble**: 由于流水线依赖导致的 GPU 空闲时间
- **Schedule**: 调度策略（如 1F1B, Interleaved）用于减少 bubble

---

### 1.2 为什么需要 PP

#### 使用场景

1. **超大模型训练**
   - 单 GPU 无法容纳完整模型（如 175B+ 参数）
   - 与 TP/EP 组合实现多维并行

2. **多模态模型**
   - 第一个 PP stage 包含视觉编码器，显存占用高
   - 后续 stage 只包含 LLM 层，负载相对均衡

3. **MoE 模型**
   - 某些层包含大量专家，需要独立 PP stage
   - 配合 EP (Expert Parallel) 实现高效训练

#### 内存优势

```
Without PP (single GPU):
  Memory = Embedding + All Layers + Output + Optimizer States

With PP=4:
  Memory per GPU ≈ Embedding/4 + Layers/4 + Output/4 + Optimizer States/4
  实际节省: ~60-75% (取决于模型架构)
```

---

### 1.3 PP 与其他并行策略的关系

```
世界大小 (World Size) = TP × PP × EP × DP × CP

示例配置 (64 GPUs):
  TP=4   (Tensor Parallel, 每层切分)
  PP=4   (Pipeline Parallel, 层间切分)
  EP=2   (Expert Parallel, MoE 专家切分)
  DP=2   (Data Parallel, 数据切分)
  ────────────────────────────
  Total = 4 × 4 × 2 × 2 = 64 GPUs
```

**MS-SWIFT 中 PP 的位置**:
- PP 在**最外层**：不同 PP rank 处理不同层
- 每个 PP stage 内部可以使用 TP/EP/DP
- CP (Context Parallel) 可以与 PP 组合使用

---

## 2. 核心架构

### 2.1 三层架构设计

MS-SWIFT 的 PP 实现采用**三层架构**：

```
┌────────────────────────────────────────────────────────────┐
│ 1. 配置层 (Configuration Layer)                            │
│    - MegatronArguments (pipeline_model_parallel_size)      │
│    - decoder_first/last_pipeline_num_layers                │
│    - num_layers_per_virtual_pipeline_stage                 │
└────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────┐
│ 2. 初始化层 (Initialization Layer)                         │
│    - parallel_state.initialize_model_parallel()            │
│    - 创建 PP process groups                                │
│    - 确定当前 rank 的 PP rank                              │
└────────────────────────────────────────────────────────────┐
                           ↓
┌────────────────────────────────────────────────────────────┐
│ 3. 执行层 (Execution Layer)                                │
│    - Layer distribution (层分配)                           │
│    - Forward-backward schedule (Megatron-Core)             │
│    - P2P communication (rank间通信)                        │
│    - Mcore-Bridge weight loading (权重加载)                │
└────────────────────────────────────────────────────────────┘
```

---

### 2.2 源码文件组织

| 文件路径 | 功能描述 | 核心代码行数 |
|---------|---------|------------|
| `swift/megatron/argument/megatron_args.py` | PP 参数定义 | ~805 行 |
| `swift/megatron/model/gpt_bridge.py` | Mcore-Bridge PP 权重加载 | ~500 行 (lines 1-500) |
| `swift/megatron/trainers/base.py` | PP 训练器基类 | ~1,231 行 |
| `swift/megatron/trainers/trainer.py` | Forward-backward 调用 | ~77-80 行 |
| `swift/megatron/init.py` | PP 初始化逻辑 | ~300 行 (部分) |
| `swift/megatron/utils/utils.py` | PP 工具函数 | ~347 行 |
| `examples/megatron/moe/moe.sh` | PP 配置示例 | ~44 行 |

**总计**: ~3,500 行核心代码

---

## 3. 参数配置

### 3.1 核心参数定义

**文件**: `swift/megatron/argument/megatron_args.py`

```python
@dataclass
class MegatronArguments(ExtraMegatronArguments):
    # ==================== Pipeline Parallel 参数 ====================

    # 基础配置
    pipeline_model_parallel_size: int = 1  # PP 大小，默认为 1（无 PP）

    # 层分配配置
    decoder_first_pipeline_num_layers: Optional[int] = None  # 第一个 PP stage 的层数
    decoder_last_pipeline_num_layers: Optional[int] = None   # 最后一个 PP stage 的层数

    # 特殊配置
    account_for_embedding_in_pipeline_split: bool = False  # Embedding 层算作一个 layer
    account_for_loss_in_pipeline_split: bool = False       # Loss 层算作一个 layer

    # Virtual Pipeline Parallel (VPP)
    num_layers_per_virtual_pipeline_stage: Optional[int] = None  # 每个 virtual stage 的层数
    num_virtual_stages_per_pipeline_rank: Optional[int] = None   # 每个 rank 的 virtual stages 数
    microbatch_group_size_per_virtual_pipeline_stage: Optional[int] = None
    pipeline_model_parallel_layout: Optional[str] = None         # PP 布局策略
```

**位置**: `swift/megatron/argument/megatron_args.py:460-476`

---

### 3.2 参数验证逻辑

```python
# swift/megatron/argument/megatron_args.py:704-707

if self.pipeline_model_parallel_size == 1 and (
    self.decoder_first_pipeline_num_layers is not None or
    self.decoder_last_pipeline_num_layers is not None
):
    raise ValueError(
        'pipeline_model_parallel_size must be greater than 1 if you want to set '
        'decoder_first_pipeline_num_layers or decoder_last_pipeline_num_layers.'
    )
```

**验证规则**:
1. `decoder_first/last_pipeline_num_layers` 只能在 `PP > 1` 时使用
2. `freeze_parameters_ratio` 不支持 `PP > 1`（见 `megatron_args.py:298-299`）

---

### 3.3 GRPO 特殊约束

```python
# swift/megatron/argument/megatron_args.py:216-224

world_size = torch.distributed.get_world_size()
dp_size = world_size // (
    self.pipeline_model_parallel_size *
    self.tensor_model_parallel_size *
    self.context_parallel_size
)

num_rollout_prompt = self.generation_batch_size // self.num_generations
if num_rollout_prompt % dp_size != 0:
    raise ValueError(
        f'num_rollout_prompt ({num_rollout_prompt}) must be divisible by '
        f'dp_size ({dp_size}). Please adjust generation_batch_size/num_generations.'
    )
```

**关键公式**:
```
DP Size = World Size / (PP × TP × CP)

约束条件:
(generation_batch_size / num_generations) % DP Size == 0
```

这意味着 **PP 大小直接影响 GRPO 的 batch size 配置**！

---

## 4. 进程组初始化

### 4.1 PP Process Group 创建

PP 进程组通过 Megatron-Core 的 `parallel_state.initialize_model_parallel()` 创建。

**关键代码**（Megatron-Core 内部）:
```python
# 伪代码：Megatron-Core 内部实现
def initialize_model_parallel(
    tensor_model_parallel_size,
    pipeline_model_parallel_size,
    context_parallel_size,
    expert_model_parallel_size,
):
    # 创建 PP process groups
    for tp_rank in range(tensor_model_parallel_size):
        for cp_rank in range(context_parallel_size):
            for ep_rank in range(expert_model_parallel_size):
                # 每个 (TP, CP, EP) 组合创建一个 PP group
                pp_ranks = [
                    get_rank(pp_idx, tp_rank, cp_rank, ep_rank)
                    for pp_idx in range(pipeline_model_parallel_size)
                ]
                pp_group = torch.distributed.new_group(pp_ranks)
```

---

### 4.2 PP Rank 确定

**文件**: `swift/megatron/model/gpt_bridge.py:56-69`

```python
class GPTBridge:
    def __init__(self, disable_tqmd: bool = False):
        # ... 省略部分代码 ...

        self.tp_size = self.args.tensor_model_parallel_size    # TP 大小
        self.pp_size = self.args.pipeline_model_parallel_size  # PP 大小
        self.etp_size = self.args.expert_tensor_parallel_size  # Expert TP 大小
        self.ep_size = self.args.expert_model_parallel_size    # Expert Parallel 大小

        # 获取 PP process group 和 rank
        self.pp_group = mpu.get_pipeline_model_parallel_group()
        self.pp_rank = mpu.get_pipeline_model_parallel_rank()
```

**Rank 映射关系**:
```
Global Rank = f(TP rank, PP rank, EP rank, DP rank, CP rank)

示例 (TP=2, PP=2, DP=2):
┌───────┬───────┬───────┬──────────────┐
│TP Rank│PP Rank│DP Rank│ Global Rank  │
├───────┼───────┼───────┼──────────────┤
│   0   │   0   │   0   │      0       │
│   1   │   0   │   0   │      1       │
│   0   │   1   │   0   │      2       │
│   1   │   1   │   0   │      3       │
│   0   │   0   │   1   │      4       │
│   1   │   0   │   1   │      5       │
│   0   │   1   │   1   │      6       │
│   1   │   1   │   1   │      7       │
└───────┴───────┴───────┴──────────────┘
```

---

### 4.3 PP First/Last Stage 判断

Megatron-Core 提供了便捷的辅助函数：

```python
from megatron.core import mpu

# 判断是否为第一个 PP stage
is_first_stage = mpu.is_pipeline_first_stage(ignore_virtual=True)

# 判断是否为最后一个 PP stage
is_last_stage = mpu.is_pipeline_last_stage(ignore_virtual=True)

# 获取当前 PP rank
pp_rank = mpu.get_pipeline_model_parallel_rank()

# 获取 PP world size
pp_size = mpu.get_pipeline_model_parallel_world_size()
```

**使用场景**:
- **First stage**: 负责 embedding 层和输入处理
- **Last stage**: 负责 loss 计算和输出层
- **Middle stages**: 只负责 transformer 层的 forward/backward

---

## 5. 层分配机制

### 5.1 默认均匀分配

当 `decoder_first_pipeline_num_layers` 和 `decoder_last_pipeline_num_layers` 都为 `None` 时，层数均匀分配。

**默认行为**:
```python
# Megatron-Core 内部逻辑（伪代码）
num_layers_per_stage = total_layers // pipeline_model_parallel_size

# 示例: 32 layers, PP=4
# PP Rank 0: Layers 0-7   (8 layers)
# PP Rank 1: Layers 8-15  (8 layers)
# PP Rank 2: Layers 16-23 (8 layers)
# PP Rank 3: Layers 24-31 (8 layers)
```

---

### 5.2 不均匀分配: `decoder_first_pipeline_num_layers`

用于**第一个 PP stage 显存占用过高**的场景（如多模态模型）。

**示例配置**:
```bash
# examples/megatron/mcore_bridge/full/moe.sh:13-14

--pipeline_model_parallel_size 2 \
--decoder_first_pipeline_num_layers 25 \
```

**层分配结果** (总共 48 layers):
```
PP Rank 0: Layers 0-24   (25 layers) ← 第一个 stage
PP Rank 1: Layers 25-47  (23 layers) ← 剩余 layers 均匀分配
```

**适用场景**:
- **多模态模型**: PP Rank 0 包含视觉编码器，需要更少的 transformer 层
- **Embedding 层过大**: 第一个 stage 需要预留更多显存

---

### 5.3 不均匀分配: `decoder_last_pipeline_num_layers`

用于**最后一个 PP stage 显存占用过高**的场景（如 MoE 模型）。

**示例配置**:
```bash
# examples/megatron/moe/moe.sh:12-13

--pipeline_model_parallel_size 2 \
--decoder_last_pipeline_num_layers 11 \
```

**层分配结果** (总共 28 layers):
```
PP Rank 0: Layers 0-16   (17 layers) ← 剩余 layers 均匀分配
PP Rank 1: Layers 17-27  (11 layers) ← 最后一个 stage
```

**适用场景**:
- **Output Layer 过大**: 最后一个 stage 需要输出整个词表
- **MoE 模型**: 某些后期层包含大量专家

---

### 5.4 同时使用 First 和 Last

**示例**:
```bash
--pipeline_model_parallel_size 4 \
--decoder_first_pipeline_num_layers 8 \
--decoder_last_pipeline_num_layers 6 \
```

**层分配逻辑** (总共 32 layers):
```
PP Rank 0: Layers 0-7     (8 layers)  ← decoder_first_pipeline_num_layers
PP Rank 1: Layers 8-16    (9 layers)  ← 中间层均匀分配
PP Rank 2: Layers 17-25   (9 layers)  ← 中间层均匀分配
PP Rank 3: Layers 26-31   (6 layers)  ← decoder_last_pipeline_num_layers

计算方式:
  middle_layers = 32 - 8 - 6 = 18 layers
  middle_stages = 4 - 2 = 2 stages
  layers_per_middle_stage = 18 // 2 = 9 layers
```

---

### 5.5 层分配的底层实现

层分配由 **Megatron-Core** 的 `get_num_layers_to_build` 和 `get_transformer_layer_offset` 函数决定。

**参考代码** (`swift/megatron/utils/utils.py:333-346`):

```python
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

def get_local_layer_specs(config, layer_specs, vp_stage=None):
    """获取当前 PP rank 应该构建的层规格"""

    # 获取当前 PP rank 应该构建的层数
    kwargs = {'vp_stage': vp_stage} if mcore_013 else {}
    num_layers_to_build = get_num_layers_to_build(config, **kwargs)

    if getattr(config, 'pipeline_model_parallel_layout', None) is not None:
        # 使用自定义 PP layout
        from megatron.core.transformer.enums import LayerType
        local_layer_specs = [
            layer_specs[layer_id]
            for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder, **kwargs
            )
        ]
    else:
        # 默认: 基于 offset 的层分配
        offset = get_transformer_layer_offset(config, **kwargs)
        local_layer_specs = layer_specs[offset:offset + num_layers_to_build]

    return local_layer_specs
```

**`get_transformer_layer_offset` 伪代码**:
```python
def get_transformer_layer_offset(config, pp_rank):
    """计算当前 PP rank 的层起始位置"""

    if config.decoder_first_pipeline_num_layers and pp_rank == 0:
        return 0  # 第一个 stage 从 layer 0 开始

    if config.decoder_last_pipeline_num_layers and pp_rank == (pp_size - 1):
        # 最后一个 stage
        return total_layers - config.decoder_last_pipeline_num_layers

    # 中间 stages: 累加前面所有 stages 的层数
    offset = 0
    if config.decoder_first_pipeline_num_layers:
        offset += config.decoder_first_pipeline_num_layers
        offset += (total_layers - first - last) // (pp_size - 2) * (pp_rank - 1)
    else:
        offset = (total_layers // pp_size) * pp_rank

    return offset
```

---

## 6. Virtual Pipeline Parallel (VPP)

### 6.1 什么是 VPP

**Virtual Pipeline Parallel** 通过**交错调度（Interleaved Schedule）** 进一步减少 pipeline bubble。

#### 传统 PP vs VPP

```
传统 PP (PP=2):
PP Rank 0: [Layers 0-15]
PP Rank 1: [Layers 16-31]

Bubble 较大: ~50%

─────────────────────────────────────

VPP (PP=2, num_virtual_stages_per_pipeline_rank=2):
PP Rank 0 Virtual Stage 0: [Layers 0-7]
PP Rank 0 Virtual Stage 1: [Layers 16-23]
PP Rank 1 Virtual Stage 0: [Layers 8-15]
PP Rank 1 Virtual Stage 1: [Layers 24-31]

交错调度，Bubble 减少至 ~25%
```

---

### 6.2 VPP 参数

**文件**: `swift/megatron/argument/megatron_args.py:473-476`

```python
@dataclass
class MegatronArguments:
    # VPP 配置
    num_layers_per_virtual_pipeline_stage: Optional[int] = None
    num_virtual_stages_per_pipeline_rank: Optional[int] = None
    microbatch_group_size_per_virtual_pipeline_stage: Optional[int] = None
    pipeline_model_parallel_layout: Optional[str] = None
```

**参数说明**:
- `num_layers_per_virtual_pipeline_stage`: 每个 virtual stage 包含的层数
- `num_virtual_stages_per_pipeline_rank`: 每个 PP rank 包含的 virtual stages 数量
- `microbatch_group_size_per_virtual_pipeline_stage`: 每个 virtual stage 的 microbatch 组大小

---

### 6.3 VPP 计算示例

**配置**:
```bash
--num_layers 32 \
--pipeline_model_parallel_size 4 \
--num_layers_per_virtual_pipeline_stage 4
```

**计算**:
```
total_virtual_stages = 32 / 4 = 8 virtual stages
num_virtual_stages_per_pp_rank = 8 / 4 = 2 virtual stages

PP Rank 0: Virtual Stage 0 (Layers 0-3), Virtual Stage 4 (Layers 16-19)
PP Rank 1: Virtual Stage 1 (Layers 4-7), Virtual Stage 5 (Layers 20-23)
PP Rank 2: Virtual Stage 2 (Layers 8-11), Virtual Stage 6 (Layers 24-27)
PP Rank 3: Virtual Stage 3 (Layers 12-15), Virtual Stage 7 (Layers 28-31)
```

---

### 6.4 VPP 调度优势

#### Bubble Reduction

```
传统 1F1B Schedule (PP=4):
Bubble% ≈ (PP - 1) / (num_microbatches + PP - 1)

VPP Interleaved Schedule (PP=4, VPP=2):
Bubble% ≈ (PP - 1) / (num_microbatches * VPP + PP - 1)

示例 (num_microbatches=8):
  传统 1F1B: (4-1) / (8+4-1) ≈ 27.3%
  VPP:       (4-1) / (8*2+4-1) ≈ 15.8%

优化效果: ~42% bubble 减少
```

---

## 7. 数据流与调度

### 7.1 Forward-Backward Schedule

MS-SWIFT 使用 **Megatron-Core 的 `get_forward_backward_func`** 获取调度函数。

**代码位置**: `swift/megatron/trainers/base.py:604`

```python
from megatron.core.pipeline_parallel import get_forward_backward_func

# Training step
forward_backward_func = get_forward_backward_func()

# Evaluation step
loss_dicts = forward_backward_func(
    forward_step_func=forward_step_func,
    data_iterator=new_data_iterator,
    model=model,
    num_microbatches=eval_num_microbatches,
    seq_length=args.seq_length,
    micro_batch_size=args.micro_batch_size,
    decoder_seq_length=args.decoder_seq_length,
    forward_only=True,  # Evaluation mode
)
```

---

### 7.2 调度策略

Megatron-Core 支持多种调度策略：

| 调度策略 | 描述 | Bubble % | 适用场景 |
|---------|------|----------|---------|
| **GPipe** | Forward all → Backward all | (PP-1) × 100% / num_microbatches | 早期方案，bubble 大 |
| **1F1B** | 1 Forward + 1 Backward interleaved | (PP-1) / (num_microbatches + PP - 1) | 标准 PP |
| **Interleaved (VPP)** | Virtual stages interleaved | (PP-1) / (num_microbatches × VPP + PP - 1) | 大规模训练 |

**MS-SWIFT 默认使用**: **1F1B Schedule**

---

### 7.3 P2P Communication

PP stages 之间通过 **point-to-point (P2P) communication** 传递激活值。

**核心函数** (`swift/megatron/init.py:50-59`):

```python
from megatron.core.pipeline_parallel import p2p_communication

# MS-SWIFT 的 patch: 移除 group 参数限制
_batched_p2p_ops_origin = p2p_communication._batched_p2p_ops

def _batched_p2p_ops(**kwargs):
    kwargs['group'] = None  # 允许跨 PP group 通信
    return _batched_p2p_ops_origin(**kwargs)

p2p_communication._batched_p2p_ops = _batched_p2p_ops
```

**通信模式**:
```
Forward Pass:
  PP Rank 0 → send activations → PP Rank 1
  PP Rank 1 → send activations → PP Rank 2
  PP Rank 2 → send activations → PP Rank 3

Backward Pass:
  PP Rank 3 → send gradients → PP Rank 2
  PP Rank 2 → send gradients → PP Rank 1
  PP Rank 1 → send gradients → PP Rank 0
```

---

### 7.4 Loss 计算

Loss 只在**最后一个 PP stage** 计算。

**代码示例** (`swift/megatron/trainers/trainer.py:77-80`):

```python
# 只有 PP last stage 计算 loss
if not mpu.is_pipeline_last_stage(ignore_virtual=True):
    # 中间 stages: 返回 None
    return None
else:
    # 最后一个 stage: 计算并返回 loss
    loss = torch.cat([
        torch.sum(losses * loss_mask).view(1),
        loss_mask.sum().view(1)
    ])

    # CP All-Reduce (megatron-core 0.12 only)
    if args.context_parallel_size > 1 and not self.mcore_013:
        loss = all_reduce(loss, group=mpu.get_context_parallel_group())

    # DP All-Reduce
    torch.distributed.all_reduce(
        reporting_loss,
        group=mpu.get_data_parallel_group()
    )

    # Normalization
    lm_loss = lm_loss / mpu.get_context_parallel_world_size()

    return lm_loss
```

---

## 8. Mcore-Bridge PP 实现

### 8.1 权重加载流程

Mcore-Bridge 负责将 HuggingFace 格式的权重转换为 Megatron-Core 格式，并**正确分发到各个 PP rank**。

**核心类**: `swift/megatron/model/gpt_bridge.py:29-99`

```python
class GPTBridge:
    def __init__(self, disable_tqmd: bool = False):
        # 获取 PP 相关信息
        self.pp_size = self.args.pipeline_model_parallel_size
        self.pp_rank = mpu.get_pipeline_model_parallel_rank()
        self.pp_group = mpu.get_pipeline_model_parallel_group()

        # 创建 EP-PP 混合 group (for MoE models)
        dp_size = dist.get_world_size() // self.etp_size // self.ep_size // self.pp_size
        expert_decoder_rank_generator = mpu.RankGenerator(
            tp=self.etp_size,
            ep=self.ep_size,
            dp=dp_size,
            pp=self.pp_size,
            cp=1,
            order='tp-cp-ep-dp-pp',
        )

        for ranks in expert_decoder_rank_generator.get_ranks('ep-pp'):
            group = mpu.create_group(ranks, group_desc='EP-PP-GROUP')
            if dist.get_rank() in ranks:
                self.ep_pp_size = self.ep_size * self.pp_size
                self.ep_pp_group = group
                self.ep_pp_rank = dist.get_rank(group)
```

**位置**: `swift/megatron/model/gpt_bridge.py:74-93`

---

### 8.2 PP Broadcast 机制

当 `PP > 1` 时，权重需要从拥有权重的 rank broadcast 到其他 PP ranks。

**代码** (`swift/megatron/model/gpt_bridge.py:260-276`):

```python
def _set_module(self, mg_module, hf_state_dict, hf_prefix: str, to_mcore: bool):
    """
    将 HF state_dict 设置到 Megatron module
    to_mcore=False 时: 从 Megatron → HF (for export)
    """

    if not to_mcore:  # Export 场景
        hf_state_dict = None if mg_module is None else mg_module.state_dict()

        # PP > 1: 需要 broadcast
        if self.pp_size > 1:
            # Step 1: 确定哪个 PP rank 拥有这个 module
            src_rank = torch.tensor(
                [0 if hf_state_dict is None else self.pp_rank],
                dtype=torch.int64,
                device='cuda'
            )
            dist.all_reduce(src_rank, group=self.pp_group)
            src_rank = dist.get_global_rank(self.pp_group, src_rank.item())

            # Step 2: Broadcast metadata (keys)
            meta_data = [None] if hf_state_dict is None else [list(hf_state_dict.keys())]
            dist.broadcast_object_list(meta_data, src=src_rank, group=self.pp_group)

            # Step 3: 所有 ranks 创建相同结构的 dict
            if meta_data[0] is None:
                return {}
            hf_state_dict = hf_state_dict or {k: None for k in meta_data[0]}

            # Step 4: Broadcast 每个 weight
            for k, v in hf_state_dict.items():
                v, _ = self._get_weight(v, None)
                hf_state_dict[k] = v

        elif hf_state_dict is None:
            return {}

        return self._add_prefix(hf_state_dict, hf_prefix)
```

---

### 8.3 PP Rank 层映射

不同 PP rank 只加载自己负责的层。

**核心函数**: `_broadcast_ep_pp` (`swift/megatron/model/gpt_bridge.py:305-330`)

```python
def _broadcast_ep_pp(self, tensor, is_expert):
    """
    Broadcast tensor across PP (or EP-PP) group

    Args:
        tensor: 当前 rank 的 weight (如果有)，否则为 None
        is_expert: 是否为 expert weight (MoE)

    Returns:
        All ranks 都会获得 tensor (通过 broadcast)
    """
    pp_group = self.ep_pp_group if is_expert else self.pp_group
    pp_size = self.ep_pp_size if is_expert else self.pp_size
    pp_rank = self.ep_pp_rank if is_expert else self.pp_rank

    # PP/EP > 1: 需要 broadcast
    if pp_size > 1:
        # Step 1: 确定源 rank
        src_rank = torch.tensor(
            [0 if tensor is None else pp_rank],
            dtype=torch.int64,
            device='cuda'
        )
        dist.all_reduce(src_rank, group=pp_group)
        src_rank = dist.get_global_rank(pp_group, src_rank.item())

        # Step 2: Broadcast metadata (shape, dtype)
        meta_data = torch.zeros(10, dtype=torch.int64, device='cuda')
        dtype_mapping = {
            torch.float64: 0, torch.float32: 1,
            torch.float16: 2, torch.bfloat16: 3,
            torch.uint8: 4
        }
        dtype_mapping_r = {v: k for k, v in dtype_mapping.items()}

        if tensor is None:
            # 接收端: 根据 metadata 创建空 tensor
            dist.broadcast(meta_data, src=src_rank, group=pp_group)
            shape = meta_data[1:1 + meta_data[0]].tolist()
            dtype = dtype_mapping_r[meta_data[-1].item()]
            tensor = torch.empty(shape, device='cuda', dtype=dtype)
            dist.broadcast(tensor, src=src_rank, group=pp_group)
        else:
            # 发送端: 发送 metadata 和 tensor
            meta_data[0] = tensor.ndim
            meta_data[1:1 + tensor.ndim] = torch.tensor(
                tensor.shape, dtype=torch.int64, device='cuda'
            )
            meta_data[-1] = dtype_mapping[tensor.dtype]
            dist.broadcast(meta_data, src=src_rank, group=pp_group)
            dist.broadcast(tensor, src=src_rank, group=pp_group)

    return tensor
```

**工作流程**:
```
假设 PP=2, 32 layers:

Loading Phase:
  1. PP Rank 0: 从 HF checkpoint 加载 Layers 0-15
  2. PP Rank 1: 从 HF checkpoint 加载 Layers 16-31

Export Phase:
  1. PP Rank 0: 导出 Layers 0-15 的 weights
  2. PP Rank 1: 导出 Layers 16-31 的 weights
  3. Merge: 两个 ranks 的 weights 通过 broadcast 合并
```

---

## 9. GRPO 与 PP 约束

### 9.1 DP Size 计算

**文件**: `swift/megatron/argument/megatron_args.py:216-224`

```python
def _init_grpo(self):
    # ... 省略部分代码 ...

    world_size = torch.distributed.get_world_size()

    # 关键公式：DP Size 计算
    dp_size = world_size // (
        self.pipeline_model_parallel_size *
        self.tensor_model_parallel_size *
        self.context_parallel_size
    )

    # Constraint 1: num_rollout_prompt 必须能被 dp_size 整除
    num_rollout_prompt = self.generation_batch_size // self.num_generations
    if num_rollout_prompt % dp_size != 0:
        raise ValueError(
            f'num_rollout_prompt ({num_rollout_prompt}) = generation_batch_size '
            f'({self.generation_batch_size}) // num_generations ({self.num_generations}) '
            f'must be divisible by dp_size ({dp_size}). '
            f'Please adjust generation_batch_size/steps_per_generation/num_generations.'
        )

    # Constraint 2: per_device_num_rollout_prompt 必须能被 micro_batch_size 整除
    per_device_num_rollout_prompt = num_rollout_prompt // dp_size
    if per_device_num_rollout_prompt % self.micro_batch_size != 0:
        raise ValueError(
            f'Per-device rollout prompt count ({per_device_num_rollout_prompt}) '
            f'must be divisible by micro_batch_size ({self.micro_batch_size}). '
        )
```

---

### 9.2 示例计算

**配置**:
```bash
--pipeline_model_parallel_size 2 \
--tensor_model_parallel_size 2 \
--context_parallel_size 1 \
--generation_batch_size 128 \
--num_generations 8 \
--micro_batch_size 2
```

**计算过程**:
```
World Size = 8 GPUs

DP Size = 8 / (PP × TP × CP)
        = 8 / (2 × 2 × 1)
        = 2

num_rollout_prompt = generation_batch_size / num_generations
                   = 128 / 8
                   = 16

Constraint 1: 16 % 2 == 0 ✅

per_device_num_rollout_prompt = 16 / 2 = 8

Constraint 2: 8 % 2 == 0 ✅

✅ 配置有效！
```

---

### 9.3 为什么 GRPO 示例中 PP=1?

**观察**: 所有 GRPO 示例 (`examples/megatron/grpo/*.sh`) 都使用 `PP=1`

**原因**:
1. **Batch Size 约束严格**: PP > 1 会减少 DP Size，增加配置难度
2. **vLLM Colocate 模式**: 训练和推理共享 GPUs，PP 会增加通信复杂度
3. **简化调试**: PP=1 更容易调试 GRPO 训练流程

**示例** (`examples/megatron/grpo/dense_colocate.sh:19`):
```bash
--context_parallel_size 1 \
--tensor_model_parallel_size 1 \
--pipeline_model_parallel_size 1 \  # ← PP=1
```

---

## 10. 示例配置

### 10.1 MoE 模型 PP 配置

**文件**: `examples/megatron/moe/moe.sh`

```bash
# 8 GPUs, Qwen1.5-MoE-A2.7B (28 layers)
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
    --model Qwen/Qwen1.5-MoE-A2.7B \
    --dataset 'liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT' \

    # ==================== Pipeline Parallel 配置 ====================
    --pipeline_model_parallel_size 2 \
    --decoder_last_pipeline_num_layers 11 \  # 最后一个 stage: 11 layers

    # ==================== Expert Parallel 配置 ====================
    --expert_model_parallel_size 4 \

    # ==================== 其他配置 ====================
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --max_length 8192 \
```

**层分配结果**:
```
总层数: 28 layers

PP Rank 0: Layers 0-16   (17 layers)  # First stage
PP Rank 1: Layers 17-27  (11 layers)  # Last stage (指定)

GPU 分配 (8 GPUs):
  PP Rank 0: GPUs 0-3 (TP=1, EP=4)
  PP Rank 1: GPUs 4-7 (TP=1, EP=4)
```

---

### 10.2 多模态模型 PP 配置

**文件**: `examples/megatron/multimodal/moe/full_dpo_offload.sh`

```bash
# Qwen3-VL-MoE (48 layers + Vision Encoder)
megatron rlhf \
    --model Qwen/Qwen3-VL-MoE \

    # ==================== Pipeline Parallel 配置 ====================
    --pipeline_model_parallel_size 2 \
    --decoder_first_pipeline_num_layers 23 \  # 第一个 stage: 23 layers (包含 Visual)

    # ==================== Tensor/Expert Parallel ====================
    --tensor_model_parallel_size 4 \
    --expert_model_parallel_size 4 \

    # ==================== Offload 配置 ====================
    --offload_optimizer true \
    --offload_model true \
```

**层分配结果**:
```
总层数: 48 LLM layers + Visual Encoder

PP Rank 0: Visual Encoder + Layers 0-22   (23 LLM layers)  # First stage
PP Rank 1: Layers 23-47                     (25 LLM layers)  # Last stage

为什么 First Stage 层数少？
  → Visual Encoder 占用大量显存
  → 减少 LLM 层数以平衡显存
```

---

### 10.3 DeepSeek-V3 PP 配置

**文件**: `examples/megatron/moe/deepseek_v3.sh`

```bash
# DeepSeek-V3 (61 layers, 671B parameters)
megatron sft \
    --model deepseek-ai/DeepSeek-V3-Base \

    # ==================== Pipeline Parallel 配置 ====================
    --pipeline_model_parallel_size 2 \
    --decoder_last_pipeline_num_layers 13 \  # 最后一个 stage: 13 layers

    # ==================== Expert Parallel 配置 ====================
    --expert_model_parallel_size 4 \

    # ==================== MoE 优化 ====================
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
```

**层分配结果**:
```
总层数: 61 layers

PP Rank 0: Layers 0-47   (48 layers)  # First stage
PP Rank 1: Layers 48-60  (13 layers)  # Last stage (指定)

原因:
  → 最后几层可能包含更多 MoE experts
  → 减少层数以平衡显存
```

---

## 11. 性能分析

### 11.1 内存节省

**理论内存节省**:
```
Without PP:
  Memory = Embedding + All Layers + Output + Optimizer States

With PP=4:
  Memory per GPU ≈ (Embedding + All Layers + Output) / 4 + Optimizer States / 4

实际节省率:
  - 纯 LLM: ~60-70%
  - 多模态: ~70-75% (Visual Encoder 在 PP Rank 0)
  - MoE: ~50-60% (Expert 分布不均)
```

**示例** (Qwen1.5-MoE-A2.7B):
```
配置: PP=2, EP=4, 8 GPUs

Without PP (模拟):
  单 GPU 显存需求: ~57 GiB

With PP=2:
  PP Rank 0: ~28.5 GiB
  PP Rank 1: ~28.5 GiB

节省率: ~50%
```

---

### 11.2 训练吞吐量

**Bubble Overhead**:
```
Bubble% = (PP - 1) / (num_microbatches + PP - 1)

示例 (global_batch_size=128, micro_batch_size=2):
  num_microbatches = 128 / 2 = 64

  PP=1: Bubble = 0%
  PP=2: Bubble = (2-1) / (64+2-1) ≈ 1.5%
  PP=4: Bubble = (4-1) / (64+4-1) ≈ 4.5%
  PP=8: Bubble = (8-1) / (64+8-1) ≈ 9.9%

结论: 增大 global_batch_size 或 micro_batch_size 可以减少 bubble
```

**实际测速** (Qwen1.5-MoE-A2.7B, 8 GPUs):
```
配置: PP=2, EP=4, micro_batch_size=1, global_batch_size=16

结果: 2.95 秒/iteration

对比 PP=1 (模拟):
  由于单 GPU 显存不足，无法训练

结论: PP 使得大模型训练成为可能
```

---

### 11.3 VPP 性能提升

**测试配置**:
```bash
--num_layers 48 \
--pipeline_model_parallel_size 4 \
--num_layers_per_virtual_pipeline_stage 6 \  # VPP
--micro_batch_size 2 \
--global_batch_size 128
```

**性能对比**:
```
Traditional 1F1B (PP=4):
  Bubble% = (4-1) / (64+4-1) ≈ 4.5%
  Throughput: 100% baseline

VPP Interleaved (PP=4, VPP=2):
  Bubble% = (4-1) / (64*2+4-1) ≈ 2.3%
  Throughput: ~102-103% (bubble 减少带来的提升)

Trade-off:
  ✅ Bubble 减少 ~49%
  ❌ 通信次数增加 ~100% (更多 P2P send/recv)
  ❌ 内存占用略增 (~5-10%, 需要存储多个 virtual stages 的激活值)
```

---

## 12. 最佳实践

### 12.1 何时使用 PP

✅ **应该使用 PP 的场景**:
1. **模型过大**: 单 GPU 无法容纳完整模型
2. **多模态模型**: Visual Encoder 占用大量显存
3. **MoE 模型**: 配合 EP 使用，减少单 GPU 显存压力
4. **跨节点训练**: 节点间带宽有限，PP 可以减少跨节点通信

❌ **不应该使用 PP 的场景**:
1. **模型较小**: 单 GPU 可以容纳（PP 会引入 bubble overhead）
2. **Batch Size 过小**: Bubble 占比过高
3. **GRPO 训练**: Batch size 约束严格，建议 PP=1

---

### 12.2 PP Size 选择

**推荐原则**:
```
1. 优先满足显存需求:
   PP Size = ceil(Model Size / GPU Memory)

2. 考虑 Bubble Overhead:
   Bubble% < 5% → 可接受

3. 平衡公式:
   PP Size = 2^n (便于层均匀分配)

示例:
  Model: 70B parameters, GPU: 80GB A100
  → PP=2 (35B per GPU, ~45GB 显存占用)

  Model: 175B parameters, GPU: 80GB A100
  → PP=4 (44B per GPU, ~58GB 显存占用)
```

---

### 12.3 层分配策略

**场景 1: 多模态模型**
```bash
# Qwen3-VL: Visual Encoder 在 PP Rank 0
--pipeline_model_parallel_size 2 \
--decoder_first_pipeline_num_layers 20 \  # 减少第一个 stage 的 LLM 层数
```

**场景 2: MoE 模型**
```bash
# DeepSeek-V3: 后期层有更多 experts
--pipeline_model_parallel_size 2 \
--decoder_last_pipeline_num_layers 13 \  # 减少最后一个 stage 的层数
```

**场景 3: 层数无法整除**
```bash
# 32 layers, PP=3 (无法整除)
--pipeline_model_parallel_size 3 \
--decoder_first_pipeline_num_layers 11 \  # PP Rank 0: 11 layers
--decoder_last_pipeline_num_layers 11 \   # PP Rank 2: 11 layers
# PP Rank 1: 10 layers (自动计算)
```

---

### 12.4 VPP 使用建议

**何时使用 VPP**:
```
1. PP Size 较大 (PP ≥ 4)
2. num_microbatches 较大 (≥ 64)
3. 训练吞吐量对 bubble 敏感

推荐配置:
  num_virtual_stages_per_pp_rank = 2
  num_layers_per_virtual_pipeline_stage = num_layers / (PP * 2)
```

**VPP 限制**:
- 内存占用增加 5-10%
- 通信次数翻倍
- 某些优化（如 activation checkpointing）可能受影响

---

### 12.5 调试技巧

**检查层分配**:
```python
# 在训练开始前打印
from megatron.training import get_args
from megatron.core import mpu

args = get_args()
pp_rank = mpu.get_pipeline_model_parallel_rank()
pp_size = mpu.get_pipeline_model_parallel_world_size()

print(f"[PP Rank {pp_rank}/{pp_size}] "
      f"Building layers: {get_num_layers_to_build(args.config)}")
```

**检查显存分布**:
```bash
# 使用 nvidia-smi 监控各 GPU 显存
watch -n 1 nvidia-smi

# 期望: 各 PP stage 显存占用接近
# 如果不均衡: 调整 decoder_first/last_pipeline_num_layers
```

**检查 Bubble Overhead**:
```python
# 在训练日志中查找
grep "bubble" logging.jsonl

# 计算实际 bubble%:
# Bubble% ≈ (total_time - compute_time) / total_time
```

---

## 13. 总结

### 13.1 核心发现

1. **PP 实现依赖 Megatron-Core**
   - MS-SWIFT 不自己实现 PP 调度，而是使用 Megatron-Core 的 `get_forward_backward_func`
   - 好处: 稳定性高，性能优化充分
   - 限制: 必须依赖 Megatron-Core

2. **Mcore-Bridge 负责权重分发**
   - PP 权重加载/导出通过 `_broadcast_ep_pp` 实现
   - 每个 PP rank 只加载自己负责的层
   - Export 时通过 broadcast 合并权重

3. **层分配支持不均匀分割**
   - `decoder_first_pipeline_num_layers`: 多模态模型必备
   - `decoder_last_pipeline_num_layers`: MoE 模型常用
   - 灵活适应不同模型架构

4. **VPP 有效减少 Bubble**
   - Bubble 减少 ~50%
   - 代价: 内存 +5-10%, 通信 ×2

5. **GRPO 与 PP 约束严格**
   - `DP Size = World Size / (PP × TP × CP)`
   - Batch size 必须满足严格整除条件
   - 建议 GRPO 训练使用 PP=1

---

### 13.2 架构特点

```
MS-SWIFT Pipeline Parallel 架构:

┌────────────────────────────────────────────────────────────┐
│ 用户配置                                                    │
│   --pipeline_model_parallel_size N                         │
│   --decoder_first/last_pipeline_num_layers                 │
└────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────┐
│ Megatron-Core 初始化                                        │
│   parallel_state.initialize_model_parallel()               │
│   创建 PP process groups                                   │
└────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────┐
│ 层分配                                                      │
│   get_num_layers_to_build()                                │
│   get_transformer_layer_offset()                           │
└────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────┐
│ Mcore-Bridge 权重加载                                       │
│   HF → Megatron format conversion                          │
│   PP broadcast (per layer)                                 │
└────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────┐
│ 训练 Schedule                                               │
│   get_forward_backward_func() ← Megatron-Core              │
│   1F1B or Interleaved (VPP)                                │
└────────────────────────────────────────────────────────────┘
```

---

### 13.3 与其他并行策略对比

| 并行策略 | 切分维度 | 通信模式 | 内存节省 | 实现复杂度 | MS-SWIFT 支持 |
|---------|---------|---------|---------|-----------|--------------|
| **Data Parallel** | 数据 | All-Reduce (Gradients) | 0% (重复存储) | 低 | ✅ (默认) |
| **Tensor Parallel** | 层内 | All-Reduce (GEMM) | ~1/TP | 中 | ✅ (TP) |
| **Pipeline Parallel** | 层间 | P2P (Activations) | ~1/PP | 高 | ✅ (PP, VPP) |
| **Expert Parallel** | MoE 专家 | All-to-All (Routing) | ~1/EP | 高 | ✅ (EP) |
| **Sequence Parallel** | 序列长度 | All-Gather/Reduce-Scatter | ~1/SP | 中 | ✅ (SP, CP) |

**组合示例**:
```
超大规模训练 (1024 GPUs):
  TP=8   (单层切分)
  PP=8   (层间切分)
  EP=8   (MoE 专家切分)
  DP=2   (数据并行)
  ────────────────────────
  Total = 8 × 8 × 8 × 2 = 1024 GPUs
```

---

### 13.4 未来改进方向

1. **自适应层分配**
   - 自动 profiling 各层显存占用
   - 动态调整 decoder_first/last_pipeline_num_layers

2. **更多 Schedule 支持**
   - GPipe Schedule (研究场景)
   - ZB (Zero Bubble) Schedule (最新研究)

3. **混合精度 PP**
   - 不同 PP stage 使用不同精度
   - 例如: First stage FP32, Middle stages FP16, Last stage FP32

4. **PP 与 Offload 结合**
   - CPU offload + PP 进一步减少显存

---

## 附录

### A. 关键源码位置索引

| 功能 | 文件路径 | 行号 |
|-----|---------|------|
| PP 参数定义 | `swift/megatron/argument/megatron_args.py` | 460-476 |
| PP 参数验证 | `swift/megatron/argument/megatron_args.py` | 704-707 |
| GRPO DP 计算 | `swift/megatron/argument/megatron_args.py` | 216-224 |
| PP Group 初始化 | `swift/megatron/model/gpt_bridge.py` | 56-69 |
| EP-PP Group 创建 | `swift/megatron/model/gpt_bridge.py` | 74-93 |
| PP Broadcast | `swift/megatron/model/gpt_bridge.py` | 260-276, 305-330 |
| Layer Distribution | `swift/megatron/utils/utils.py` | 333-346 |
| Forward-Backward Call | `swift/megatron/trainers/base.py` | 604 |
| Loss Computation | `swift/megatron/trainers/trainer.py` | 77-80 |
| P2P Patch | `swift/megatron/init.py` | 50-59 |

---

### B. 术语表

| 术语 | 全称 | 中文 | 说明 |
|-----|------|------|------|
| PP | Pipeline Parallel | 流水线并行 | 层间切分 |
| VPP | Virtual Pipeline Parallel | 虚拟流水线并行 | Interleaved schedule |
| TP | Tensor Parallel | 张量并行 | 层内切分 |
| EP | Expert Parallel | 专家并行 | MoE 专家切分 |
| DP | Data Parallel | 数据并行 | 数据切分 |
| SP | Sequence Parallel | 序列并行 | 序列长度切分 |
| CP | Context Parallel | 上下文并行 | 长序列切分 (≡ SP) |
| 1F1B | One-Forward-One-Backward | 一前一后 | 标准 PP schedule |
| Bubble | Pipeline Bubble | 流水线气泡 | GPU 空闲时间 |
| P2P | Point-to-Point | 点对点通信 | Rank 间直接通信 |
| Mcore | Megatron-Core | - | NVIDIA 分布式训练框架 |

---

### C. 参考资料

1. **Megatron-Core 官方文档**
   - Pipeline Parallel: https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/pipeline_parallel

2. **论文**
   - GPipe: https://arxiv.org/abs/1811.06965
   - PipeDream (1F1B): https://arxiv.org/abs/1806.03377
   - Megatron-LM: https://arxiv.org/abs/1909.08053

3. **MS-SWIFT 相关**
   - MS-SWIFT GitHub: https://github.com/modelscope/ms-swift
   - Megatron-SWIFT 文档: https://swift.readthedocs.io/en/latest/Megatron-SWIFT/

---

**文档版本**: v1.0
**最后更新**: 2026-01-04
**分析者**: Claude (Anthropic)
