# Mcore-Bridge 架构深度分析

> 作者：LLM Training 框架开发者
> 分析日期：2026-01-01
> 基于版本：ms-swift main branch (commit cd18f085)

## 目录

1. [概述](#概述)
2. [核心问题与解决方案](#核心问题与解决方案)
3. [架构设计](#架构设计)
4. [权重转换机制](#权重转换机制)
5. [并行策略处理](#并行策略处理)
6. [LoRA 集成](#lora-集成)
7. [关键优化技术](#关键优化技术)
8. [设计模式分析](#设计模式分析)
9. [实现细节](#实现细节)
10. [扩展性分析](#扩展性分析)

---

## 概述

### 什么是 Mcore-Bridge？

**Mcore-Bridge** 是 ms-swift 中实现的一个双向权重转换系统，作为 **HuggingFace/Transformers 格式**（safetensors）和 **NVIDIA Megatron-LM 格式**（torch distributed checkpoints）之间的桥梁。

### 核心价值

使 Megatron 训练变得像 transformers 一样简单：

```python
# 传统 Megatron 训练需要手动转换权重
# 步骤1: 下载 HF 模型
# 步骤2: 运行转换脚本
# 步骤3: 配置复杂的并行参数
# 步骤4: 训练
# 步骤5: 再次转换回 HF 格式

# Mcore-Bridge 简化为：
megatron sft \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --load_safetensors true \
    --save_safetensors true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2
# 自动完成加载、转换、训练、保存全流程
```

---

## 核心问题与解决方案

### 1. 格式不兼容性

**问题：**
- HuggingFace: 单文件或分片 safetensors（`model.safetensors`, `model-00001-of-00002.safetensors`）
- Megatron: 分布式检查点（`mp_rank_00/model_optim_rng.pt`, `mp_rank_01/...`）

**解决方案：**
```python
# 文件位置：swift/megatron/model/gpt_bridge.py
class GPTBridge:
    def load_weights(self, mg_model, hf_model_dir):
        # 使用 SafetensorLazyLoader 读取 HF 格式
        with SafetensorLazyLoader(hf_model_dir) as loader:
            state_dict = loader.get_state_dict()
            self._convert([mg_model], state_dict, to_mcore=True)

    def save_weights(self, mg_models, output_dir):
        # 使用 StreamingSafetensorSaver 写入 HF 格式
        with StreamingSafetensorSaver(output_dir) as saver:
            for key, tensor in self.export_weights(mg_models):
                saver.save_tensor(key, tensor)
```

**代码位置：**
- `/home/scbjtfy/ms-swift/swift/megatron/model/gpt_bridge.py`: 1396-1475 行
- `/home/scbjtfy/ms-swift/swift/megatron/utils/io_utils.py`: 24-169 行

### 2. 权重布局差异

**问题：**

| 模块 | HuggingFace 格式 | Megatron 格式 |
|------|------------------|---------------|
| Attention | `q_proj`, `k_proj`, `v_proj` 分离 | `linear_qkv` 融合 |
| MLP | `gate_proj`, `up_proj` 分离 | `linear_fc1` 融合为 `[2, hidden, ffn_hidden]` |
| Experts (MoE) | `experts.{i}.gate_proj.weight` 分散 | 聚合为 `[num_experts, hidden, ffn_hidden]` |

**解决方案：**

```python
# QKV 融合（HF → Megatron）
# 位置：gpt_bridge.py lines 512-531
def _set_attention_state(self, mg_attn, hf_state_dict, ...):
    # 加载分离的 Q/K/V
    q_weight = hf_state_dict['q_proj.weight'].load()
    k_weight = hf_state_dict['k_proj.weight'].load()
    v_weight = hf_state_dict['v_proj.weight'].load()

    # 按 query group 重组
    q_weight = q_weight.reshape((num_query_groups, -1, hidden_size))
    k_weight = k_weight.reshape((num_query_groups, -1, hidden_size))
    v_weight = v_weight.reshape((num_query_groups, -1, hidden_size))

    # 拼接为 QKV
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=1)
    qkv_weight = qkv_weight.reshape(-1, hidden_size)

    # 设置到 Megatron 参数（自动处理 TP 切分）
    self._set_weight(mg_attn.linear_qkv.weight, qkv_weight, 'linear_qkv.weight')

# QKV 解融合（Megatron → HF）
# 位置：gpt_bridge.py lines 560-577
def _get_attention_state(self, mg_attn, ...):
    # 从 Megatron 获取融合的 QKV（自动处理 TP gather）
    qkv_weight, _ = self._get_weight(mg_attn.linear_qkv.weight, 'linear_qkv.weight')

    # 重组为 query groups
    qkv_weight = qkv_weight.reshape((num_query_groups, -1, hidden_size))

    # 分离为 Q/K/V
    q_weight = qkv_weight[:, :q_dim, :].reshape(-1, hidden_size)
    k_weight = qkv_weight[:, q_dim:q_dim+kv_dim, :].reshape(-1, hidden_size)
    v_weight = qkv_weight[:, -kv_dim:, :].reshape(-1, hidden_size)

    return {
        'q_proj.weight': q_weight,
        'k_proj.weight': k_weight,
        'v_proj.weight': v_weight,
    }
```

### 3. 词表填充

**问题：**
- Megatron 要求 vocab_size 是 TP size 的倍数（为了效率）
- HuggingFace 没有这个要求

**解决方案：**

```python
# 位置：swift/megatron/utils/utils.py lines 316-331
def get_padding_to(value: int, padding_divisor: int) -> int:
    """计算需要填充到的值"""
    if padding_divisor <= 1:
        return value
    return math.ceil(value / padding_divisor) * padding_divisor

# 位置：swift/megatron/argument/config.py lines 95-105
def convert_hf_config(config):
    args = get_args()
    # 计算填充后的词表大小
    padded_vocab_size = get_padding_to(
        config.vocab_size,
        args.tensor_model_parallel_size * args.make_vocab_size_divisible_by
    )

    return {
        'padded_vocab_size': padded_vocab_size,
        'true_vocab_size': config.vocab_size,  # 记录原始大小
    }

# 加载时：只加载前 true_vocab_size 个 embeddings
# 保存时：只保存前 true_vocab_size 个 embeddings
```

**代码位置：**
- `/home/scbjtfy/ms-swift/swift/megatron/model/gpt_bridge.py`: 434-453 行（加载 embeddings）
- `/home/scbjtfy/ms-swift/swift/megatron/model/gpt_bridge.py`: 810-828 行（保存 embeddings）

### 4. LoRA 增量权重转换

**问题：**
- HuggingFace LoRA: PEFT 格式，adapter_name 嵌入 key 中
- Megatron LoRA: LoraParallelLinear 包装，遵循 TP/EP 切分规则

**解决方案见 [LoRA 集成](#lora-集成) 章节**

---

## 架构设计

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Mcore-Bridge System                               │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  User Interface Layer                                         │      │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐             │      │
│  │  │ megatron   │  │ megatron   │  │ swift      │             │      │
│  │  │ sft        │  │ export     │  │ infer      │             │      │
│  │  │--load_safe │  │            │  │ (自动检测)  │             │      │
│  │  │--save_safe │  │            │  │            │             │      │
│  │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘             │      │
│  └────────┼───────────────┼───────────────┼────────────────────┘      │
│           │               │               │                            │
│  ┌────────▼───────────────▼───────────────▼────────────────────┐      │
│  │  Conversion Orchestration Layer                              │      │
│  │  (swift/megatron/convert.py)                                 │      │
│  │                                                               │      │
│  │  • convert_hf2mcore()    - HF → Megatron 全流程             │      │
│  │  • convert_mcore2hf()    - Megatron → HF 全流程             │      │
│  │  • test_convert_precision() - 精度验证                       │      │
│  └────────┬──────────────────────────────────────────┬──────────┘      │
│           │                                           │                 │
│  ┌────────▼───────────────────────────────────────────▼─────────┐      │
│  │  Bridge Layer (GPTBridge)                                    │      │
│  │  (swift/megatron/model/gpt_bridge.py - 1481 lines)           │      │
│  │                                                               │      │
│  │  ┌─────────────────┐         ┌──────────────────┐           │      │
│  │  │ load_weights()  │         │ export_weights() │           │      │
│  │  │ save_weights()  │         │ (generator)      │           │      │
│  │  └────────┬────────┘         └────────┬─────────┘           │      │
│  │           │                            │                     │      │
│  │  ┌────────▼────────────────────────────▼─────────┐          │      │
│  │  │         _convert() - Core Logic                │          │      │
│  │  │  • _convert_pre_process (embeddings)           │          │      │
│  │  │  • _set_layer_state (per-layer conversion)     │          │      │
│  │  │  • _convert_post_process (output layer)        │          │      │
│  │  └────────┬───────────────────────────────────────┘          │      │
│  │           │                                                   │      │
│  │  ┌────────▼──────────────────────────────────────┐           │      │
│  │  │  Component Handlers                            │           │      │
│  │  │  • _set_attention_state() - QKV fusion         │           │      │
│  │  │  • _set_mlp_state()       - MLP fusion         │           │      │
│  │  │  • _set_moe_state()       - MoE expert groups  │           │      │
│  │  └────────┬───────────────────────────────────────┘           │      │
│  └───────────┼───────────────────────────────────────────────────┘      │
│              │                                                           │
│  ┌───────────▼──────────────────────────────────────────────────┐      │
│  │  Parallelism Handler Layer                                    │      │
│  │                                                                │      │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │      │
│  │  │ _set_weight()│  │ _get_weight()│  │ _set_module()│        │      │
│  │  │              │  │              │  │              │        │      │
│  │  │ • Split TP   │  │ • Gather TP  │  │ • Load state │        │      │
│  │  │ • Broadcast  │  │ • Broadcast  │  │   dict       │        │      │
│  │  │   PP         │  │   PP         │  │              │        │      │
│  │  └──────┬───────┘  └──────┬───────┘  └──────────────┘        │      │
│  │         │                  │                                  │      │
│  │  ┌──────▼──────────────────▼─────────────────────┐           │      │
│  │  │  Low-level Parallelism Primitives              │           │      │
│  │  │  • _split_tp()          - TP 切分               │           │      │
│  │  │  • _all_gather_tp()     - TP 全收集             │           │      │
│  │  │  • _broadcast_ep_pp()   - EP/PP 广播           │           │      │
│  │  │  • _get_tp_split_dim()  - 判断切分维度          │           │      │
│  │  └────────────────────────────────────────────────┘           │      │
│  └────────────────────────────────────────────────────────────────      │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  I/O Layer                                                    │      │
│  │                                                                │      │
│  │  ┌─────────────────────┐      ┌──────────────────────┐       │      │
│  │  │ SafetensorLazyLoader│      │StreamingSafetensor   │       │      │
│  │  │                     │      │Saver                 │       │      │
│  │  │ • 懒加载 HF 权重    │      │ • 流式保存 HF 权重   │       │      │
│  │  │ • LazyTensor 包装   │      │ • 5GB 分片           │       │      │
│  │  │ • safe_open()       │      │ • 多线程写入         │       │      │
│  │  └─────────────────────┘      └──────────────────────┘       │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  Storage Layer                                                │      │
│  │                                                                │      │
│  │  ┌─────────────────────┐      ┌──────────────────────┐       │      │
│  │  │ HF Safetensors      │      │ Megatron torch_dist  │       │      │
│  │  │                     │      │                      │       │      │
│  │  │ model.safetensors   │      │ mp_rank_00/          │       │      │
│  │  │ model-00001-of-*.   │      │   model_optim_rng.pt │       │      │
│  │  │ model.safetensors.  │      │ mp_rank_01/...       │       │      │
│  │  │ index.json          │      │                      │       │      │
│  │  └─────────────────────┘      └──────────────────────┘       │      │
│  └──────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 核心组件

#### 1. GPTBridge 类

**职责：** 核心转换逻辑

**关键方法：**

| 方法 | 功能 | 位置 |
|------|------|------|
| `__init__()` | 初始化并行组、Meta HF 模型 | 39-99 行 |
| `load_weights()` | HF → Megatron 加载 | 1396-1403 行 |
| `save_weights()` | Megatron → HF 保存 | 1421-1475 行 |
| `export_weights()` | 生成器模式导出权重 | 1405-1419 行 |
| `_convert()` | 核心转换逻辑（template method） | 486-612 行 |
| `_set_weight()` | 设置权重 + TP 切分 | 153-193 行 |
| `_get_weight()` | 获取权重 + TP gather | 332-387 行 |

**代码位置：** `/home/scbjtfy/ms-swift/swift/megatron/model/gpt_bridge.py`

#### 2. SafetensorLazyLoader

**职责：** 延迟加载 HuggingFace 权重

**特性：**
- 使用 `safetensors.safe_open()` 打开文件句柄
- 返回 `LazyTensor` 包装对象
- 只在 `.load()` 调用时真正读取数据

**代码实现：**

```python
# 位置：swift/megatron/utils/io_utils.py lines 24-79
class SafetensorLazyLoader:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self._file_handles = {}
        self._weight_map = self._load_index()

    def _load_index(self):
        # 读取 model.safetensors.index.json
        index_file = os.path.join(self.model_dir, 'model.safetensors.index.json')
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                index = json.load(f)
            return index['weight_map']
        else:
            # 单文件 model.safetensors
            return {key: 'model.safetensors' for key in ...}

    def get_state_dict(self):
        """返回 LazyTensor 字典"""
        state_dict = {}
        for key in self._weight_map.keys():
            state_dict[key] = LazyTensor(
                key=key,
                loader=lambda k=key: self._load_tensor(k)
            )
        return state_dict

    def _load_tensor(self, key):
        """实际加载张量"""
        filename = self._weight_map[key]
        if filename not in self._file_handles:
            filepath = os.path.join(self.model_dir, filename)
            self._file_handles[filename] = safe_open(filepath, framework='pt')
        return self._file_handles[filename].get_tensor(key)

class LazyTensor:
    def __init__(self, key, loader):
        self.key = key
        self.loader = loader
        self.tensor = None

    def load(self):
        """延迟加载"""
        if self.tensor is None:
            self.tensor = self.loader()
        return self.tensor
```

**内存优势：**
- 不需要一次性加载所有权重（例如 Qwen3-235B 需要 ~470GB）
- 只在需要时加载当前层的权重
- 配合 `_convert()` 的层级遍历，内存峰值 = 单层权重大小

#### 3. StreamingSafetensorSaver

**职责：** 流式保存 HuggingFace 权重

**特性：**
- 累积权重直到达到分片大小限制（默认 5GB）
- 自动创建分片文件（`model-00001-of-00010.safetensors`）
- 生成 `model.safetensors.index.json`
- 支持多线程写入（通过 `thread_count` 参数）

**代码实现：**

```python
# 位置：swift/megatron/utils/io_utils.py lines 81-169
class StreamingSafetensorSaver:
    def __init__(self, output_dir, max_shard_size='5GB'):
        self.output_dir = output_dir
        self.max_shard_size = self._parse_size(max_shard_size)  # 5GB = 5e9 bytes
        self.current_shard = {}
        self.current_size = 0
        self.shard_idx = 1
        self.weight_map = {}

    def save_tensor(self, key, tensor):
        """累积张量"""
        tensor_size = tensor.numel() * tensor.element_size()

        # 检查是否需要开始新分片
        if self.current_size + tensor_size > self.max_shard_size and self.current_shard:
            self._flush_shard()

        self.current_shard[key] = tensor
        self.current_size += tensor_size
        self.weight_map[key] = f'model-{self.shard_idx:05d}-of-?????.safetensors'

    def _flush_shard(self):
        """写入当前分片"""
        filename = f'model-{self.shard_idx:05d}-of-?????.safetensors'
        filepath = os.path.join(self.output_dir, filename)
        save_file(self.current_shard, filepath)

        self.current_shard = {}
        self.current_size = 0
        self.shard_idx += 1

    def finalize(self):
        """完成保存，生成 index.json"""
        if self.current_shard:
            self._flush_shard()

        total_shards = self.shard_idx - 1
        # 重命名文件，填充总分片数
        for i in range(1, total_shards + 1):
            old_name = f'model-{i:05d}-of-?????.safetensors'
            new_name = f'model-{i:05d}-of-{total_shards:05d}.safetensors'
            os.rename(
                os.path.join(self.output_dir, old_name),
                os.path.join(self.output_dir, new_name)
            )

        # 生成 index.json
        index = {
            'metadata': {'total_size': sum(...)},
            'weight_map': self.weight_map
        }
        with open(os.path.join(self.output_dir, 'model.safetensors.index.json'), 'w') as f:
            json.dump(index, f, indent=2)
```

---

## 权重转换机制

### HF → Megatron 转换流程

**完整流程图：**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 初始化阶段                                                │
│    (convert.py: convert_hf2mcore)                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────┐                  │
│  │ prepare_model_template()              │                  │
│  │ • 加载 HF model/tokenizer             │                  │
│  │ • 获取 config                         │                  │
│  └──────────────┬───────────────────────┘                  │
│                 │                                            │
│  ┌──────────────▼───────────────────────┐                  │
│  │ convert_hf_config()                   │                  │
│  │ • 映射 HF config → Megatron args      │                  │
│  │ • 计算 padded_vocab_size              │                  │
│  │ • 设置 num_layers, hidden_size 等     │                  │
│  └──────────────┬───────────────────────┘                  │
│                 │                                            │
│  ┌──────────────▼───────────────────────┐                  │
│  │ initialize_megatron()                 │                  │
│  │ • 初始化分布式环境                    │                  │
│  │ • 创建并行组 (TP/PP/EP/CP)            │                  │
│  └──────────────┬───────────────────────┘                  │
│                 │                                            │
│  ┌──────────────▼───────────────────────┐                  │
│  │ model_provider()                      │                  │
│  │ • 创建空 Megatron 模型                │                  │
│  │ • 参数未初始化（random 或 zeros）     │                  │
│  └──────────────┬───────────────────────┘                  │
└─────────────────┼───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│ 2. 权重加载阶段                                              │
│    (gpt_bridge.py: load_weights)                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────┐                  │
│  │ SafetensorLazyLoader(hf_model_dir)    │                  │
│  │ • 打开 model.safetensors.index.json   │                  │
│  │ • 创建 LazyTensor 字典                │                  │
│  └──────────────┬───────────────────────┘                  │
│                 │                                            │
│  ┌──────────────▼───────────────────────┐                  │
│  │ _convert(mg_model, hf_state_dict,     │                  │
│  │          to_mcore=True)               │                  │
│  │                                        │                  │
│  │ ┌────────────────────────────────┐   │                  │
│  │ │ _convert_pre_process()          │   │                  │
│  │ │ • word_embeddings (PP rank 0)   │   │                  │
│  │ │ • position_embeddings (如果有)  │   │                  │
│  │ └────────────────────────────────┘   │                  │
│  │                                        │                  │
│  │ ┌────────────────────────────────┐   │                  │
│  │ │ for layer_idx in tqdm(...):     │   │                  │
│  │ │   _set_layer_state()            │   │                  │
│  │ │                                 │   │                  │
│  │ │   ┌─────────────────────────┐  │   │                  │
│  │ │   │ 判断 PP 归属               │  │   │                  │
│  │ │   │ pp_layer_idx = layer_idx % │  │   │                  │
│  │ │   │   layers_per_pp_rank      │  │   │                  │
│  │ │   └────────┬────────────────┘  │   │                  │
│  │ │            │                    │   │                  │
│  │ │   ┌────────▼────────────────┐  │   │                  │
│  │ │   │ if belongs_to_this_rank: │  │   │                  │
│  │ │   │   _set_attention_state() │  │   │                  │
│  │ │   │   _set_mlp_state()       │  │   │                  │
│  │ │   │   _set_moe_state() (MoE) │  │   │                  │
│  │ │   └─────────────────────────┘  │   │                  │
│  │ └────────────────────────────────┘   │                  │
│  │                                        │                  │
│  │ ┌────────────────────────────────┐   │                  │
│  │ │ _convert_post_process()         │   │                  │
│  │ │ • final_layernorm (PP last rank)│   │                  │
│  │ │ • output_layer/score (task head)│   │                  │
│  │ └────────────────────────────────┘   │                  │
│  └──────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│ 3. 组件级转换示例 (Attention)                                │
│    (_set_attention_state)                                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────┐                  │
│  │ 1. 加载 HF 权重                       │                  │
│  │    q_weight = hf['q_proj.weight'].load()│                │
│  │    k_weight = hf['k_proj.weight'].load()│                │
│  │    v_weight = hf['v_proj.weight'].load()│                │
│  └──────────────┬───────────────────────┘                  │
│                 │                                            │
│  ┌──────────────▼───────────────────────┐                  │
│  │ 2. 重组为 Query Groups               │                  │
│  │    (GQA/MQA 支持)                     │                  │
│  │    q = q.reshape(num_query_groups,    │                  │
│  │                  -1, hidden_size)     │                  │
│  │    k = k.reshape(num_query_groups,    │                  │
│  │                  kv_dim, hidden_size) │                  │
│  │    v = v.reshape(num_query_groups,    │                  │
│  │                  kv_dim, hidden_size) │                  │
│  └──────────────┬───────────────────────┘                  │
│                 │                                            │
│  ┌──────────────▼───────────────────────┐                  │
│  │ 3. 拼接 QKV                           │                  │
│  │    qkv = torch.cat([q, k, v], dim=1)  │                  │
│  │    qkv = qkv.reshape(-1, hidden_size) │                  │
│  └──────────────┬───────────────────────┘                  │
│                 │                                            │
│  ┌──────────────▼───────────────────────┐                  │
│  │ 4. 设置到 Megatron 参数               │                  │
│  │    _set_weight(                       │                  │
│  │      mg_attn.linear_qkv.weight,       │                  │
│  │      qkv_weight,                      │                  │
│  │      'linear_qkv.weight'              │                  │
│  │    )                                  │                  │
│  │    ↓ 内部处理:                        │                  │
│  │    • 判断 TP split dim = 0 (Column)   │                  │
│  │    • 切分: qkv.chunk(tp_size, dim=0)  │                  │
│  │    • 复制到参数: param.data.copy_()   │                  │
│  └──────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

**关键代码片段：**

```python
# 位置：gpt_bridge.py lines 486-612
def _convert(self, mg_models, hf_state_dict, hf_prefix, to_mcore, tqdm_desc='Converting'):
    """核心转换逻辑"""

    # 1. 预处理（embeddings）
    hf_state_dict.update(self._convert_pre_process(
        mg_models, hf_state_dict, hf_prefix, to_mcore
    ))

    # 2. 逐层转换
    num_layers = self.args.num_layers
    layers_per_pp_rank = num_layers // self.pp_size

    for layer_idx in tqdm(range(num_layers), desc=tqdm_desc, disable=self.disable_tqmd):
        # 判断这一层属于哪个 PP rank
        pp_layer_idx = layer_idx % layers_per_pp_rank
        pp_rank = layer_idx // layers_per_pp_rank

        # 只有拥有这一层的 PP rank 才执行转换
        if pp_rank == self.pp_rank:
            mg_layer = mg_models[0].decoder.layers[pp_layer_idx]
            layer_prefix = f'{hf_prefix}{layer_idx}.'

            hf_state_dict.update(self._set_layer_state(
                mg_layer, hf_state_dict, layer_prefix, to_mcore
            ))

    # 3. 后处理（output layer）
    hf_state_dict.update(self._convert_post_process(
        mg_models, hf_state_dict, hf_prefix, to_mcore
    ))

    return hf_state_dict

# 位置：gpt_bridge.py lines 496-531
def _set_attention_state(self, mg_attn, hf_state_dict, hf_prefix, to_mcore):
    """Attention 层转换"""
    if not to_mcore:
        # Megatron → HF (见下一节)
        return self._get_attention_state(mg_attn, hf_prefix)

    # HF → Megatron
    num_query_groups = self.args.num_query_groups
    q_dim = self.args.kv_channels * self.args.num_attention_heads // num_query_groups
    kv_dim = self.args.kv_channels
    hidden_size = self.args.hidden_size

    # 加载 HF 权重
    q_weight = hf_state_dict[f'{hf_prefix}q_proj.weight'].load()
    k_weight = hf_state_dict[f'{hf_prefix}k_proj.weight'].load()
    v_weight = hf_state_dict[f'{hf_prefix}v_proj.weight'].load()

    # 重组为 query groups（支持 GQA/MQA）
    q_weight = q_weight.reshape(num_query_groups, q_dim, hidden_size)
    k_weight = k_weight.reshape(num_query_groups, kv_dim, hidden_size)
    v_weight = v_weight.reshape(num_query_groups, kv_dim, hidden_size)

    # 拼接 QKV
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=1)
    qkv_weight = qkv_weight.reshape(-1, hidden_size)

    # 设置权重（自动处理 TP 切分）
    self._set_weight(mg_attn.linear_qkv.weight, qkv_weight, 'linear_qkv.weight')

    # 处理 bias（如果有）
    if hasattr(mg_attn.linear_qkv, 'bias'):
        q_bias = hf_state_dict[f'{hf_prefix}q_proj.bias'].load()
        k_bias = hf_state_dict[f'{hf_prefix}k_proj.bias'].load()
        v_bias = hf_state_dict[f'{hf_prefix}v_proj.bias'].load()
        qkv_bias = torch.cat([q_bias, k_bias, v_bias])
        self._set_weight(mg_attn.linear_qkv.bias, qkv_bias, 'linear_qkv.bias')

    # 输出投影（O projection）
    o_weight = hf_state_dict[f'{hf_prefix}o_proj.weight'].load()
    self._set_weight(mg_attn.linear_proj.weight, o_weight, 'linear_proj.weight')

    return {}  # 已处理的 keys，从 hf_state_dict 中移除
```

### Megatron → HF 转换流程

**完整流程图：**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 加载 Megatron 检查点                                      │
│    (convert.py: convert_mcore2hf)                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────┐                  │
│  │ load_checkpoint()                     │                  │
│  │ • 从 torch_dist 格式加载               │                  │
│  │ • 每个 rank 加载自己的分片             │                  │
│  │ • 恢复 model/optimizer/rng 状态       │                  │
│  └──────────────┬───────────────────────┘                  │
│                 │                                            │
│  ┌──────────────▼───────────────────────┐                  │
│  │ LoRA 合并（如果需要）                 │                  │
│  │ with adapter_state_dict_context():    │                  │
│  │   merge_lora_weights()                │                  │
│  └──────────────┬───────────────────────┘                  │
└─────────────────┼───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│ 2. 权重导出阶段                                              │
│    (gpt_bridge.py: export_weights - 生成器模式)             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────┐                  │
│  │ for key, tensor in bridge.export_     │                  │
│  │                    weights(mg_models):│                  │
│  │                                        │                  │
│  │   ┌────────────────────────────────┐ │                  │
│  │   │ _convert(mg_models,             │ │                  │
│  │   │          to_mcore=False)        │ │                  │
│  │   │                                 │ │                  │
│  │   │ ┌──────────────────────────┐   │ │                  │
│  │   │ │ _get_pre_process_state() │   │ │                  │
│  │   │ │ • 获取 word_embeddings   │   │ │                  │
│  │   │ │ • _get_weight() 自动:    │   │ │                  │
│  │   │ │   - TP gather            │   │ │                  │
│  │   │ │   - PP broadcast         │   │ │                  │
│  │   │ │   - 移到 CPU             │   │ │                  │
│  │   │ │ • yield 'embed_tokens'   │   │ │                  │
│  │   │ └──────────────────────────┘   │ │                  │
│  │   │                                 │ │                  │
│  │   │ ┌──────────────────────────┐   │ │                  │
│  │   │ │ for layer in layers:      │   │ │                  │
│  │   │ │   _get_layer_state()      │   │ │                  │
│  │   │ │   • _get_attention_state()│   │ │                  │
│  │   │ │     - QKV unfuse          │   │ │                  │
│  │   │ │     - yield q/k/v weights │   │ │                  │
│  │   │ │   • _get_mlp_state()      │   │ │                  │
│  │   │ │     - gate_up unfuse      │   │ │                  │
│  │   │ │     - yield gate/up       │   │ │                  │
│  │   │ └──────────────────────────┘   │ │                  │
│  │   │                                 │ │                  │
│  │   │ ┌──────────────────────────┐   │ │                  │
│  │   │ │ _get_post_process_state()│   │ │                  │
│  │   │ │ • yield final_layernorm  │   │ │                  │
│  │   │ │ • yield lm_head          │   │ │                  │
│  │   │ └──────────────────────────┘   │ │                  │
│  │   └─────────────────────────────────┘ │                  │
│  └──────────────┬───────────────────────┘                  │
└─────────────────┼───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│ 3. 流式保存阶段                                              │
│    (StreamingSafetensorSaver)                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────┐                  │
│  │ with StreamingSafetensorSaver() as    │                  │
│  │                                saver: │                  │
│  │   for key, tensor in export_weights():│                  │
│  │     saver.save_tensor(key, tensor)    │                  │
│  │                                        │                  │
│  │   ┌────────────────────────────────┐ │                  │
│  │   │ 内部逻辑:                       │ │                  │
│  │   │                                 │ │                  │
│  │   │ if current_size + tensor_size > │ │                  │
│  │   │        max_shard_size:          │ │                  │
│  │   │   flush_shard()  # 保存当前分片 │ │                  │
│  │   │                                 │ │                  │
│  │   │ current_shard[key] = tensor     │ │                  │
│  │   │ current_size += tensor_size     │ │                  │
│  │   └────────────────────────────────┘ │                  │
│  │                                        │                  │
│  │ saver.finalize()                       │                  │
│  │ • flush 最后一个分片                   │                  │
│  │ • 重命名文件 (填充总分片数)            │                  │
│  │ • 生成 model.safetensors.index.json   │                  │
│  └────────────────────────────────────────                  │
└─────────────────────────────────────────────────────────────┘
```

**关键代码片段：**

```python
# 位置：gpt_bridge.py lines 560-577
def _get_attention_state(self, mg_attn, hf_prefix):
    """Attention 层导出（Megatron → HF）"""

    # 获取融合的 QKV（自动处理 TP gather 和 PP broadcast）
    qkv_weight, _ = self._get_weight(
        mg_attn.linear_qkv.weight,
        'linear_qkv.weight'
    )

    if qkv_weight is None:
        return {}  # 这个 PP rank 不拥有这层

    num_query_groups = self.args.num_query_groups
    q_dim = self.args.kv_channels * self.args.num_attention_heads // num_query_groups
    kv_dim = self.args.kv_channels

    # 重组为 query groups
    qkv_weight = qkv_weight.reshape(num_query_groups, -1, self.args.hidden_size)

    # 分离为 Q/K/V
    q_weight = qkv_weight[:, :q_dim, :].reshape(-1, self.args.hidden_size)
    k_weight = qkv_weight[:, q_dim:q_dim+kv_dim, :].reshape(-1, self.args.hidden_size)
    v_weight = qkv_weight[:, -kv_dim:, :].reshape(-1, self.args.hidden_size)

    hf_state_dict = {
        f'{hf_prefix}q_proj.weight': q_weight,
        f'{hf_prefix}k_proj.weight': k_weight,
        f'{hf_prefix}v_proj.weight': v_weight,
    }

    # 输出投影
    o_weight, _ = self._get_weight(mg_attn.linear_proj.weight, 'linear_proj.weight')
    hf_state_dict[f'{hf_prefix}o_proj.weight'] = o_weight

    return hf_state_dict

# 位置：gpt_bridge.py lines 332-387
def _get_weight(self, mg_weight, mg_key, offset=0, is_expert=False):
    """从 Megatron 参数获取完整权重"""

    # 1. 处理 FP8 量化（如果有）
    mg_scale_inv = None
    if self._is_fp8_param(mg_weight):
        mg_scale_inv = mg_weight._rowwise_scale_inv
        mg_weight = mg_weight._rowwise_data

    # 2. 处理多个参数（例如 VPP）
    if not isinstance(mg_weight, (list, tuple)):
        mg_weight = [mg_weight]
    tensor = torch.concat(mg_weight, dim=0)

    # 3. TP all_gather
    tp_dim = self._get_tp_split_dim(mg_key)
    tensor = self._all_gather_tp(tensor, tp_dim, is_expert)

    # 4. PP broadcast（从拥有这层的 rank 广播）
    tensor = self._broadcast_ep_pp(tensor, is_expert)

    # 5. 移到目标设备（通常是 CPU）
    if self._target_device is not None:
        tensor = tensor.to(device=self._target_device)

    # 6. 只在最后一个 rank 保留（节省内存）
    if self._only_last_rank and not is_last_rank():
        tensor = None

    return tensor, mg_scale_inv

# 位置：gpt_bridge.py lines 278-303
def _all_gather_tp(self, tensor, tp_dim, is_expert):
    """TP 全收集"""
    if tensor is None:
        return None

    tp_size = self.etp_size if is_expert else self.tp_size
    tp_group = self.etp_group if is_expert else self.tp_group

    if tp_dim is not None and tp_size > 1:
        tensor = tensor.to('cuda')  # 必须在 GPU 上进行通信

        if tp_dim == 0:
            # 维度 0 切分：使用 all_gather_into_tensor（更高效）
            tensor_shape = list(tensor.shape)
            tensor_shape[0] *= tp_size
            output = tensor.new_empty(tensor_shape)
            dist.all_gather_into_tensor(output, tensor, group=tp_group)
            tensor = output
        else:
            # 维度 1 切分：使用 all_gather + cat
            output = [torch.empty_like(tensor) for _ in range(tp_size)]
            dist.all_gather(output, tensor, group=tp_group)
            tensor = torch.cat(output, dim=tp_dim)

        del output

    return tensor
```

---

## 并行策略处理

### Tensor Parallelism (TP)

**原理：** 将模型权重沿某个维度切分到多个 GPU

**实现细节：**

```python
# 位置：gpt_bridge.py lines 100-151
def _get_tp_split_dim(self, mg_key):
    """判断权重应该在哪个维度进行 TP 切分"""
    if mg_key is None:
        return None

    # ColumnParallelLinear: 输出维度切分（dim=0）
    dim0_keys = {
        'word_embeddings',      # [vocab, hidden] → 每个 rank 负责部分 vocab
        'linear_qkv',           # [3*hidden, hidden] → 每个 rank 负责部分 heads
        'linear_q_proj',        # MLA (DeepSeek)
        'linear_q_up_proj',     # MLA
        'linear_kv_up_proj',    # MLA
        'output_layer',         # LM head (causal_lm)
    }

    # RowParallelLinear: 输入维度切分（dim=1）
    dim1_keys = {
        'linear_proj',          # Attention output projection
        'linear_fc2',           # MLP down projection
    }

    # 特殊情况：linear_fc1 是 [2, hidden, ffn_hidden]
    # 第一维是 gate/up，第二维切分
    if 'linear_fc1' in mg_key:
        return 1

    # 解析 key
    key, suffix = mg_key.rsplit('.', 2)[-2:]

    if key in dim0_keys:
        return 0
    elif key in dim1_keys and suffix != 'bias':
        return 1
    else:
        return None  # 不切分（例如 layer_norm）

# 位置：gpt_bridge.py lines 144-151
def _split_tp(self, hf_weight, tp_dim, is_expert):
    """TP 切分"""
    tp_size = self.etp_size if is_expert else self.tp_size
    tp_rank = self.etp_rank if is_expert else self.tp_rank

    if tp_dim is not None and tp_size > 1:
        # 沿 tp_dim 切分，取当前 rank 的分片
        tensor = hf_weight.chunk(tp_size, dim=tp_dim)[tp_rank]
    else:
        tensor = hf_weight

    return tensor
```

**LoRA TP 切分特殊逻辑：**

```python
# 位置：gpt_bridge.py lines 121-143
def _get_tp_split_dim(self, mg_key):
    """LoRA 权重的 TP 切分维度"""
    if 'lora_A' in mg_key or 'lora_B' in mg_key:
        mg_key_splited = mg_key.rsplit('.', 3)
        key, lora_name = mg_key_splited[:2]

        if lora_name == 'lora_A':
            # lora_A: 与基础层输入维度对应
            # RowParallel (dim1_keys) → lora_A 在 dim=1 切分
            if key in dim1_keys:
                return 1
            # ColumnParallel → 不切分
            else:
                return None

        elif lora_name == 'lora_B':
            # lora_B: 与基础层输出维度对应
            # ColumnParallel (dim0_keys) → lora_B 在 dim=0 切分
            if key in dim0_keys:
                return 0
            # linear_fc1 → dim=1 切分
            elif key == 'linear_fc1':
                return 1
            # RowParallel → 不切分
            else:
                return None
```

**示例：QKV TP 切分**

假设 `num_attention_heads=32`, `hidden_size=4096`, `tp_size=4`:

```
HF Format:
  q_proj.weight: [32 * 128, 4096] = [4096, 4096]
  k_proj.weight: [8 * 128, 4096]  = [1024, 4096]  (GQA, num_kv_heads=8)
  v_proj.weight: [8 * 128, 4096]  = [1024, 4096]

Megatron Format (融合后):
  linear_qkv.weight: [4096 + 1024 + 1024, 4096] = [6144, 4096]

TP 切分 (dim=0):
  Rank 0: linear_qkv.weight[0:1536, :]     # 8 Q heads + 2 KV heads
  Rank 1: linear_qkv.weight[1536:3072, :]  # 8 Q heads + 2 KV heads
  Rank 2: linear_qkv.weight[3072:4608, :]  # 8 Q heads + 2 KV heads
  Rank 3: linear_qkv.weight[4608:6144, :]  # 8 Q heads + 2 KV heads

LoRA TP 切分:
  lora_A.weight: [lora_rank, 4096] → 不切分 (所有 ranks 相同)
  lora_B.weight: [6144, lora_rank] → dim=0 切分 (与 base layer 对应)
```

### Pipeline Parallelism (PP)

**原理：** 将模型按层切分到多个 GPU，形成流水线

**实现细节：**

```python
# 位置：gpt_bridge.py lines 486-612
def _convert(self, mg_models, hf_state_dict, hf_prefix, to_mcore, tqdm_desc):
    """PP 感知的转换逻辑"""

    num_layers = self.args.num_layers  # 例如 40 层
    pp_size = self.pp_size              # 例如 4 个 PP ranks
    pp_rank = self.pp_rank              # 当前 rank: 0, 1, 2, 3

    # 计算每个 PP rank 负责的层数
    layers_per_pp_rank = num_layers // pp_size  # 40 // 4 = 10

    # 遍历所有层
    for layer_idx in tqdm(range(num_layers)):
        # 判断这一层属于哪个 PP rank
        owner_pp_rank = layer_idx // layers_per_pp_rank
        # layer 0-9 → rank 0, layer 10-19 → rank 1, ...

        # 只有拥有这一层的 rank 才执行转换
        if owner_pp_rank == pp_rank:
            # 本地层索引
            local_layer_idx = layer_idx % layers_per_pp_rank
            mg_layer = mg_models[0].decoder.layers[local_layer_idx]

            # 转换这一层
            layer_prefix = f'{hf_prefix}{layer_idx}.'
            self._set_layer_state(mg_layer, hf_state_dict, layer_prefix, to_mcore)
        # 其他 ranks 跳过这一层

# 位置：gpt_bridge.py lines 305-330
def _broadcast_ep_pp(self, tensor, is_expert):
    """PP 广播：从拥有该层的 rank 广播到其他 ranks"""

    pp_group = self.ep_pp_group if is_expert else self.pp_group
    pp_size = self.ep_pp_size if is_expert else self.pp_size
    pp_rank = self.ep_pp_rank if is_expert else self.pp_rank

    if pp_size == 1:
        return tensor  # 没有 PP，直接返回

    # 1. 确定源 rank（哪个 rank 拥有这个 tensor）
    src_rank = torch.tensor([0 if tensor is None else pp_rank],
                            dtype=torch.int64, device='cuda')
    dist.all_reduce(src_rank, group=pp_group, op=dist.ReduceOp.SUM)
    src_rank = dist.get_global_rank(pp_group, src_rank.item())

    # 2. 广播 metadata（shape, dtype）
    meta_data = torch.zeros(10, dtype=torch.int64, device='cuda')
    if tensor is not None:
        # 源 rank 填充 metadata
        meta_data[0] = tensor.ndim
        meta_data[1:1+tensor.ndim] = torch.tensor(tensor.shape)
        meta_data[-1] = dtype_mapping[tensor.dtype]

    dist.broadcast(meta_data, src=src_rank, group=pp_group)

    # 3. 创建空 tensor（非源 ranks）
    if tensor is None:
        shape = meta_data[1:1+meta_data[0]].tolist()
        dtype = dtype_mapping_r[meta_data[-1].item()]
        tensor = torch.empty(shape, device='cuda', dtype=dtype)

    # 4. 广播 tensor 数据
    dist.broadcast(tensor, src=src_rank, group=pp_group)

    return tensor
```

**Virtual Pipeline Parallelism (VPP):**

```python
# 位置：swift/megatron/model/model_provider.py lines 61-63, 163
def model_provider(pre_process=True, post_process=True, vp_stage=None):
    """支持 VPP 的 model provider"""

    # VPP: 每个 PP rank 可能有多个 model chunks
    # vp_stage: 当前 rank 中的虚拟 stage 索引

    if args.virtual_pipeline_model_parallel_size is not None:
        # 每个 PP rank 有多个 stages
        # 例如 pp_size=2, vpp_size=4 → 每个 rank 有 2 个 stages
        num_chunks = args.virtual_pipeline_model_parallel_size // args.pipeline_model_parallel_size

        # pre_process: 只在第一个 chunk 处理
        # post_process: 只在最后一个 chunk 处理
        pre_process = (vp_stage == 0)
        post_process = (vp_stage == num_chunks - 1)

    model = GPTModel(
        ...,
        pre_process=pre_process,
        post_process=post_process,
    )

    return model
```

### Expert Parallelism (EP)

**原理：** MoE 模型中，将 experts 切分到多个 GPU

**实现细节：**

```python
# 位置：gpt_bridge.py lines 663-677
def _set_moe_state(self, mg_moe, hf_state_dict, hf_prefix, to_mcore):
    """MoE expert 转换"""

    num_experts = self.args.num_experts  # 例如 64
    ep_size = self.ep_size                # 例如 8
    ep_rank = self.ep_rank                # 0-7

    # 每个 EP rank 负责的 experts 数量
    num_local_experts = num_experts // ep_size  # 64 // 8 = 8

    # 计算当前 rank 负责的 expert 索引范围
    expert_start = ep_rank * num_local_experts
    expert_end = (ep_rank + 1) * num_local_experts
    # Rank 0: experts 0-7, Rank 1: experts 8-15, ...

    if to_mcore:
        # HF → Megatron
        for ep_idx in range(ep_size):
            # 加载这个 EP rank 的 experts
            expert_weights = []
            for i in range(num_local_experts):
                expert_idx = ep_idx * num_local_experts + i
                expert_key = f'{hf_prefix}experts.{expert_idx}.'

                # 加载 gate_proj, up_proj
                gate_weight = hf_state_dict[f'{expert_key}gate_proj.weight'].load()
                up_weight = hf_state_dict[f'{expert_key}up_proj.weight'].load()

                # 融合为 gate_up [2, hidden, ffn_hidden]
                gate_up = torch.stack([gate_weight, up_weight], dim=0)
                expert_weights.append(gate_up)

            # 堆叠为 [num_local_experts, 2, hidden, ffn_hidden]
            expert_weights = torch.stack(expert_weights, dim=0)

            # 只设置属于当前 EP rank 的部分
            if ep_idx == ep_rank:
                self._set_weight(
                    mg_moe.experts.linear_fc1.weight,
                    expert_weights,
                    'linear_fc1.weight',
                    is_expert=True
                )
    else:
        # Megatron → HF
        # 从所有 EP ranks 收集 experts
        ...

# 位置：gpt_bridge.py lines 354-387
def _get_weight(self, mg_weight, mg_key, offset=0, is_expert=False):
    """Expert 权重获取"""

    if is_expert:
        # Expert 权重需要从 EP ranks 收集
        num_local_experts = self.args.num_experts // self.ep_size

        # 先 TP gather（ETP）
        tensor = self._all_gather_tp(tensor, tp_dim, is_expert=True)
        # 使用 etp_group 而不是 tp_group

        # 再 EP-PP broadcast
        tensor = self._broadcast_ep_pp(tensor, is_expert=True)
        # 使用 ep_pp_group（EP 和 PP 的组合）

        # Reshape 为 [num_local_experts, ...]
        if tensor is not None:
            if mg_key.endswith('bias'):
                tensor = tensor.view(num_local_experts, -1)
            else:
                tensor = tensor.view(num_local_experts, -1, tensor.shape[-1])

    return tensor, mg_scale_inv
```

**Expert Tensor Parallelism (ETP):**

MoE 模型支持两种 TP 模式：
1. **Standard TP**: 所有 experts 使用相同的 TP
2. **Expert TP (ETP)**: Experts 和 Dense 层使用不同的 TP

```python
# 初始化 (gpt_bridge.py lines 56-69)
self.tp_size = self.args.tensor_model_parallel_size      # Dense 层 TP
self.etp_size = self.args.expert_tensor_parallel_size    # Expert 层 TP

self.tp_group = mpu.get_tensor_model_parallel_group()
self.etp_group = mpu.get_expert_tensor_parallel_group()

# 切分时根据 is_expert 选择不同的 TP 配置
def _split_tp(self, hf_weight, tp_dim, is_expert):
    tp_size = self.etp_size if is_expert else self.tp_size
    tp_rank = self.etp_rank if is_expert else self.tp_rank
    ...
```

**Grouped vs Ungrouped Experts:**

```python
# 位置：gpt_bridge.py lines 1004-1145
# HF 格式有两种:

# 1. Ungrouped (分散存储):
experts.0.gate_proj.weight: [ffn_hidden, hidden]
experts.1.gate_proj.weight: [ffn_hidden, hidden]
...
experts.63.gate_proj.weight: [ffn_hidden, hidden]

# 2. Grouped (聚合存储):
experts.gate_proj.weight: [num_experts, ffn_hidden, hidden]

# Megatron 始终使用 Grouped 格式
# Bridge 需要自动检测并转换
hf_grouped = 'experts.0.gate_proj.weight' not in hf_state_dict
```

### Context Parallelism (CP) 和 Sequence Parallelism (SP)

**CP (Context Parallelism):**
- 用于超长序列（>128K tokens）
- 将序列切分到多个 GPUs
- Ring Attention 或 Ulysses Attention

**SP (Sequence Parallelism):**
- 用于中等长度序列
- 在 LayerNorm 和 Dropout 上切分序列维度
- 减少 activation memory

**代码位置：** `/home/scbjtfy/ms-swift/swift/megatron/utils/utils.py` lines 316-331

```python
def get_padding_to(value, padding_divisor):
    """计算需要填充到的值（用于 CP/SP）"""
    if padding_divisor <= 1:
        return value
    return math.ceil(value / padding_divisor) * padding_divisor

# 在数据加载时填充序列长度
max_length = get_padding_to(
    args.max_length,
    args.context_parallel_size  # CP size
)
```

---

## LoRA 集成

### LoRA 架构

**Megatron LoRA 实现：**

```python
# 位置：swift/megatron/tuners/lora.py
class LoraParallelLinear(nn.Module):
    """LoRA for ColumnParallelLinear / RowParallelLinear"""

    def __init__(self, base_layer, lora_rank, lora_alpha, target_modules):
        super().__init__()
        self.base_layer = base_layer

        # LoRA 参数（遵循 TP 规则）
        # 如果 base_layer 是 ColumnParallel (dim=0 split):
        #   lora_A: [hidden, rank] - 不切分
        #   lora_B: [rank, hidden_per_tp] - dim=1 切分

        # 如果 base_layer 是 RowParallel (dim=1 split):
        #   lora_A: [hidden_per_tp, rank] - dim=0 切分
        #   lora_B: [rank, hidden] - 不切分

        self.lora_A = nn.Parameter(torch.zeros(in_features, lora_rank))
        self.lora_B = nn.Parameter(torch.zeros(lora_rank, out_features))
        self.scaling = lora_alpha / lora_rank

    def forward(self, x):
        # Base forward
        output = self.base_layer(x)

        # LoRA forward: x @ lora_A @ lora_B * scaling
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling

        return output + lora_output
```

### LoRA 格式转换

**PEFT 格式（HuggingFace）:**

```python
# Adapter 名称嵌入 key 中
{
    'model.layers.0.self_attn.q_proj.lora_A.default.weight': tensor,
    'model.layers.0.self_attn.q_proj.lora_B.default.weight': tensor,
    'model.layers.0.self_attn.k_proj.lora_A.default.weight': tensor,
    'model.layers.0.self_attn.k_proj.lora_B.default.weight': tensor,
    ...
}

# adapter_config.json
{
    'peft_type': 'LORA',
    'r': 8,
    'lora_alpha': 32,
    'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    'modules_to_save': [],  # 例如 ['embed_tokens', 'lm_head']
}
```

**Megatron 格式:**

```python
# 位置：swift/megatron/model/gpt_bridge.py lines 222-276
{
    'decoder.layers.0.self_attention.linear_qkv.lora_A.default.weight': tensor,
    'decoder.layers.0.self_attention.linear_qkv.lora_B.default.weight': tensor,
    'decoder.layers.0.self_attention.linear_proj.lora_A.default.weight': tensor,
    'decoder.layers.0.self_attention.linear_proj.lora_B.default.weight': tensor,
    ...
}

# 注意：QKV 融合，所以 q/k/v 的 lora_A 必须相同！
```

### LoRA 转换逻辑

**HF PEFT → Megatron:**

```python
# 位置：gpt_bridge.py lines 413-422
def _set_attention_lora(self, mg_attn, hf_state_dict, hf_prefix, adapter_name='default'):
    """QKV LoRA 转换"""

    # 1. 加载 HF LoRA 权重
    q_lora_A = hf_state_dict[f'{hf_prefix}q_proj.lora_A.{adapter_name}.weight'].load()
    k_lora_A = hf_state_dict[f'{hf_prefix}k_proj.lora_A.{adapter_name}.weight'].load()
    v_lora_A = hf_state_dict[f'{hf_prefix}v_proj.lora_A.{adapter_name}.weight'].load()

    # 2. 验证 Q/K/V 的 lora_A 相同（Megatron 要求）
    assert (q_lora_A == k_lora_A).all() and (k_lora_A == v_lora_A).all(), \
        "Q/K/V must share the same lora_A for QKV fusion in Megatron"

    # 3. 融合 lora_B
    q_lora_B = hf_state_dict[f'{hf_prefix}q_proj.lora_B.{adapter_name}.weight'].load()
    k_lora_B = hf_state_dict[f'{hf_prefix}k_proj.lora_B.{adapter_name}.weight'].load()
    v_lora_B = hf_state_dict[f'{hf_prefix}v_proj.lora_B.{adapter_name}.weight'].load()

    # 按 query groups 重组
    q_lora_B = q_lora_B.reshape(num_query_groups, -1, lora_rank)
    k_lora_B = k_lora_B.reshape(num_query_groups, -1, lora_rank)
    v_lora_B = v_lora_B.reshape(num_query_groups, -1, lora_rank)

    # 拼接
    qkv_lora_B = torch.cat([q_lora_B, k_lora_B, v_lora_B], dim=1)
    qkv_lora_B = qkv_lora_B.reshape(-1, lora_rank)

    # 4. 设置到 Megatron 参数（自动处理 TP 切分）
    self._set_weight(
        mg_attn.linear_qkv.lora_A[adapter_name].weight,
        q_lora_A,  # Q/K/V 共享
        f'linear_qkv.lora_A.{adapter_name}.weight'
    )
    self._set_weight(
        mg_attn.linear_qkv.lora_B[adapter_name].weight,
        qkv_lora_B,
        f'linear_qkv.lora_B.{adapter_name}.weight'
    )
```

**Megatron → HF PEFT:**

```python
# 位置：gpt_bridge.py lines 542-558
def _get_attention_lora(self, mg_attn, hf_prefix, adapter_name='default'):
    """QKV LoRA 导出"""

    # 1. 获取融合的 LoRA 权重
    lora_A, _ = self._get_weight(
        mg_attn.linear_qkv.lora_A[adapter_name].weight,
        f'linear_qkv.lora_A.{adapter_name}.weight'
    )
    lora_B, _ = self._get_weight(
        mg_attn.linear_qkv.lora_B[adapter_name].weight,
        f'linear_qkv.lora_B.{adapter_name}.weight'
    )

    if lora_A is None:
        return {}

    # 2. 解融合 lora_B
    lora_B = lora_B.reshape(num_query_groups, -1, lora_rank)

    q_lora_B = lora_B[:, :q_dim, :].reshape(-1, lora_rank)
    k_lora_B = lora_B[:, q_dim:q_dim+kv_dim, :].reshape(-1, lora_rank)
    v_lora_B = lora_B[:, -kv_dim:, :].reshape(-1, lora_rank)

    # 3. 返回 HF PEFT 格式
    return {
        f'{hf_prefix}q_proj.lora_A.{adapter_name}.weight': lora_A,
        f'{hf_prefix}q_proj.lora_B.{adapter_name}.weight': q_lora_B,
        f'{hf_prefix}k_proj.lora_A.{adapter_name}.weight': lora_A,  # 共享
        f'{hf_prefix}k_proj.lora_B.{adapter_name}.weight': k_lora_B,
        f'{hf_prefix}v_proj.lora_A.{adapter_name}.weight': lora_A,  # 共享
        f'{hf_prefix}v_proj.lora_B.{adapter_name}.weight': v_lora_B,
    }
```

### modules_to_save 处理

**概念：** PEFT 中，某些模块（如 embeddings, lm_head）可能需要完整训练而非 LoRA

**转换逻辑：**

```python
# 位置：gpt_bridge.py lines 248-259
def _set_module(self, mg_module, hf_state_dict, hf_prefix, to_mcore):
    """处理 modules_to_save"""

    if self._is_peft_format:
        # PEFT 格式: modules_to_save 有 adapter_name 后缀
        # 例如: 'embed_tokens.modules_to_save.default.weight'

        new_state_dict = {}
        for k, v in hf_state_dict.items():
            # 重命名 key
            k = k.replace('.modules_to_save.', f'.modules_to_save.{self._adapter_name}.')
            new_state_dict[k] = v

        # 加载到 Megatron
        mg_module.load_state_dict(new_state_dict, strict=False)
```

---

## 关键优化技术

### 1. 内存优化

**Lazy Loading:**

```python
# 传统方式（内存峰值 = 模型全部权重）
hf_model = AutoModel.from_pretrained('Qwen/Qwen3-235B')  # 需要 ~470GB RAM
mg_model = convert_to_megatron(hf_model)

# Mcore-Bridge (内存峰值 = 单层权重)
with SafetensorLazyLoader(model_dir) as loader:
    for layer_idx in range(num_layers):
        layer_weights = load_layer(layer_idx)  # 只加载当前层
        convert_layer(layer_weights)
        del layer_weights  # 立即释放
```

**Streaming Saving:**

```python
# 传统方式（内存峰值 = 模型全部权重）
full_state_dict = {}
for layer in layers:
    full_state_dict.update(export_layer(layer))
save_file(full_state_dict, 'model.safetensors')  # OOM!

# Mcore-Bridge (内存峰值 = 单个分片 ~5GB)
with StreamingSafetensorSaver(output_dir, max_shard_size='5GB') as saver:
    for layer in layers:
        for key, tensor in export_layer(layer):
            saver.save_tensor(key, tensor)  # 累积到 5GB 后自动 flush
```

**Generator Pattern:**

```python
# 位置：gpt_bridge.py lines 1405-1419
def export_weights(self, mg_models, is_peft_format=False):
    """使用生成器模式，避免构建完整 state_dict"""

    # 不使用 list()，直接返回生成器
    return self._convert(
        mg_models,
        hf_state_dict=None,
        hf_prefix='',
        to_mcore=False,
        tqdm_desc='Exporting'
    )
    # 调用者逐个消费 (key, tensor) pairs

# 使用
for key, tensor in bridge.export_weights(mg_models):
    # 处理一个 tensor
    saver.save_tensor(key, tensor)
    # tensor 可以被垃圾回收
```

### 2. 通信优化

**all_gather_into_tensor vs all_gather:**

```python
# 位置：gpt_bridge.py lines 285-301
if tp_dim == 0:
    # 维度 0 切分：使用 all_gather_into_tensor（更高效）
    # 优势：一次性分配目标 buffer，避免中间 list
    tensor_shape = list(tensor.shape)
    tensor_shape[0] *= tp_size
    output = tensor.new_empty(tensor_shape)
    dist.all_gather_into_tensor(output, tensor, group=tp_group)
    tensor = output
else:
    # 维度 1 切分：使用 all_gather
    # 需要先创建 list，然后 cat
    output = [torch.empty_like(tensor) for _ in range(tp_size)]
    dist.all_gather(output, tensor, group=tp_group)
    tensor = torch.cat(output, dim=tp_dim)
```

**多线程写入:**

```python
# 位置：convert.py lines 254-257
# 根据模型大小自动调整线程数
checkpoint_size = sum(get_n_params_grads(hf_model)[0]) * bits // 8e9  # GB
thread_count = max(math.ceil(checkpoint_size / 10), 2)  # 每 10GB 一个线程

# Patch torch.distributed.shard 使用多线程
patch_torch_dist_shard(thread_count)

# 写入时并行处理多个分片
with ThreadPoolExecutor(max_workers=thread_count) as executor:
    futures = []
    for shard in shards:
        futures.append(executor.submit(save_file, shard, filepath))
    wait(futures)
```

### 3. FP8 量化支持

**FP8 格式处理:**

```python
# 位置：gpt_bridge.py lines 162-212
def _set_weight(self, mg_param, hf_weight, mg_key, ...):
    """处理 FP8 量化参数"""

    if self._is_fp8_param(mg_param):
        # Megatron FP8 格式:
        # - _rowwise_data: uint8 编码的权重
        # - _rowwise_scale_inv: 每 128 个元素一个 scale

        # 从 HF 加载（可能是 FP8 或 FP32）
        if hf_scale_inv is None:
            # FP32 → FP8: 即时量化
            fp8_tensor = self.fp8_quantizer.quantize(hf_weight)
            mg_param._rowwise_data.copy_(fp8_tensor._rowwise_data)
            self._copy_scale_inv(mg_param, fp8_tensor._rowwise_scale_inv)
        else:
            # FP8 → FP8: 直接复制
            mg_param._rowwise_data.copy_(hf_weight.view(torch.uint8))
            self._copy_scale_inv(mg_param, hf_scale_inv)
    else:
        # 常规参数
        mg_param.data.copy_(hf_weight)

@property
def fp8_quantizer(self):
    """Lazy 初始化 FP8 量化器"""
    if self._fp8_quantizer is None:
        from transformer_engine.pytorch import Float8BlockQuantizer
        from transformer_engine_torch import DType as TE_DType
        self._fp8_quantizer = Float8BlockQuantizer(
            TE_DType.kFloat8E4M3,  # E4M3 格式
            rowwise=True,
            columnwise=True,
        )
    return self._fp8_quantizer
```

### 4. 精度验证

**test_convert_precision:**

```python
# 位置：convert.py lines 157-231
def test_convert_precision(args, mg_models, hf_model, template):
    """验证转换精度"""

    # 1. 准备测试数据
    examples = get_examples(args.is_multimodal)
    inputs = template.encode(examples)

    # 2. HF 前向传播
    with torch.no_grad():
        hf_outputs = hf_model(**inputs)
        hf_logits = hf_outputs.logits
        hf_loss = hf_outputs.loss if hasattr(hf_outputs, 'loss') else None

    # 3. Megatron 前向传播
    with torch.no_grad():
        mg_outputs = forward_step_helper(mg_models[0], inputs)
        mg_logits = mg_outputs['logits']
        mg_loss = mg_outputs.get('loss')

    # 4. 计算差异
    logits_diff = (hf_logits - mg_logits).abs()
    mean_diff = logits_diff.mean().item()
    max_diff = logits_diff.max().item()

    # 对于 multimodal，只关注有 loss 的位置
    if args.is_multimodal and hf_loss is not None:
        loss_mask = inputs['labels'] != -100
        mean_diff_with_loss = logits_diff[loss_mask].mean().item()
        logger.info(f'mean_diff (with loss): {mean_diff_with_loss:.6e}')

    # 5. 验证阈值
    assert mean_diff < 1e-3, f'Conversion precision issue: mean_diff={mean_diff}'

    logger.info(f'✓ Precision test passed: mean_diff={mean_diff:.6e}, max_diff={max_diff:.6e}')
```

---

## 设计模式分析

### 1. Bridge Pattern (桥接模式)

**定义：** 将抽象部分与实现部分分离，使它们可以独立变化

**在 Mcore-Bridge 中的应用:**

```
┌──────────────────┐         ┌──────────────────┐
│  Abstraction     │         │  Implementor     │
│                  │         │                  │
│ convert_hf2mcore │────────▶│  GPTBridge       │
│ convert_mcore2hf │         │  - load_weights  │
│                  │         │  - save_weights  │
└──────────────────┘         └────────┬─────────┘
                                      │
                      ┌───────────────┼───────────────┐
                      │               │               │
              ┌───────▼────────┐ ┌───▼────────┐ ┌───▼────────┐
              │ GPTBridge      │ │ Qwen2_5VL  │ │ DeepSeek   │
              │ (Base)         │ │ Bridge     │ │ V3Bridge   │
              └────────────────┘ └────────────┘ └────────────┘
```

**好处:**
- 格式转换逻辑（HF ↔ Megatron）与具体模型解耦
- 新增模型只需继承 `GPTBridge` 并覆盖特定方法
- 转换流程（`convert.py`）无需修改

### 2. Template Method Pattern (模板方法模式)

**定义：** 在方法中定义算法骨架，将某些步骤延迟到子类

**在 _convert() 中的应用:**

```python
# 位置：gpt_bridge.py lines 486-612
class GPTBridge:
    def _convert(self, mg_models, hf_state_dict, hf_prefix, to_mcore):
        """Template method: 定义转换算法骨架"""

        # 步骤 1: 预处理（可被子类覆盖）
        hf_state_dict.update(self._convert_pre_process(...))

        # 步骤 2: 逐层转换（可被子类覆盖）
        for layer_idx in range(num_layers):
            hf_state_dict.update(self._set_layer_state(...))

        # 步骤 3: 后处理（可被子类覆盖）
        hf_state_dict.update(self._convert_post_process(...))

        return hf_state_dict

    def _set_layer_state(self, mg_layer, hf_state_dict, hf_prefix, to_mcore):
        """Hook method: 子类可以覆盖"""
        self._set_attention_state(...)
        self._set_mlp_state(...)
        if has_moe:
            self._set_moe_state(...)

# 子类覆盖特定步骤
class Qwen2_5VLBridge(MultimodalGPTBridge):
    def _convert_pre_process(self, ...):
        """覆盖预处理：处理 vision encoder"""
        # 调用父类处理 text embeddings
        hf_state_dict = super()._convert_pre_process(...)

        # 额外处理 vision encoder
        self._set_module(mg_model.vision_model, hf_state_dict, 'vision_model.', to_mcore)

        return hf_state_dict
```

### 3. Strategy Pattern (策略模式)

**定义：** 定义一系列算法，将每个算法封装起来，使它们可以互换

**在并行策略中的应用:**

```python
# 不同的权重获取策略
class GPTBridge:
    def _get_weight(self, mg_weight, mg_key, is_expert=False):
        """根据 is_expert 选择不同的并行策略"""

        # 策略 1: Dense layer (TP)
        if not is_expert:
            tp_dim = self._get_tp_split_dim(mg_key)
            tensor = self._all_gather_tp(tensor, tp_dim, is_expert=False)
            tensor = self._broadcast_ep_pp(tensor, is_expert=False)

        # 策略 2: Expert layer (ETP + EP)
        else:
            tp_dim = self._get_tp_split_dim(mg_key)
            tensor = self._all_gather_tp(tensor, tp_dim, is_expert=True)  # 使用 ETP
            tensor = self._broadcast_ep_pp(tensor, is_expert=True)         # 使用 EP-PP group

        return tensor
```

### 4. Iterator Pattern (迭代器模式) - Generator

**定义：** 提供一种顺序访问聚合对象元素的方法，不暴露内部表示

**在 export_weights() 中的应用:**

```python
# 位置：gpt_bridge.py lines 1405-1419
def export_weights(self, mg_models, is_peft_format=False):
    """Generator pattern: 按需生成权重"""

    # 返回生成器，而不是完整 dict
    for key, tensor in self._convert(mg_models, ...):
        yield key, tensor  # 按需生成

    # 调用者可以逐个消费，控制内存使用
    for key, tensor in bridge.export_weights(mg_models):
        process(key, tensor)
        del tensor  # 可以立即释放
```

### 5. Lazy Initialization Pattern (延迟初始化模式)

**定义：** 延迟对象的创建、值的计算或其他昂贵过程，直到第一次需要时

**在多处应用:**

```python
# 1. LazyTensor
class LazyTensor:
    def __init__(self, loader):
        self.loader = loader
        self.tensor = None  # 未初始化

    def load(self):
        if self.tensor is None:  # 首次访问时加载
            self.tensor = self.loader()
        return self.tensor

# 2. Meta HF Model
def _init_meta_hf_model(self):
    """创建 meta 设备上的模型（不分配内存）"""
    with torch.device('meta'):
        self.hf_model, self.processor = get_model_tokenizer(
            self.args.model_dir,
            return_dummy_model=True  # 只创建结构，不加载权重
        )

# 3. FP8 Quantizer
@property
def fp8_quantizer(self):
    if self._fp8_quantizer is None:  # 延迟初始化
        from transformer_engine.pytorch import Float8BlockQuantizer
        self._fp8_quantizer = Float8BlockQuantizer(...)
    return self._fp8_quantizer
```

---

## 实现细节

### 特殊架构支持

#### 1. MLA (Multi-Latent Attention) - DeepSeek-V2/V3

**特点：** 压缩 K/V，使用 latent vectors

**HF 格式:**
```python
q_a_proj.weight: [1536, 5120]   # Q down projection
q_b_proj.weight: [5120, 1536]   # Q up projection
kv_a_proj_with_mqa.weight: [576, 5120]  # KV down projection
kv_b_proj.weight: [5120, 512]   # KV up projection
```

**Megatron 格式:**
```python
linear_q_down_proj.weight: [1536, 5120]
linear_q_up_proj.weight: [5120, 1536]
linear_kv_down_proj.weight: [576, 5120]
linear_kv_up_proj.weight: [5120, 512]
```

**转换代码:**

```python
# 位置：gpt_bridge.py lines 1151-1182
def _set_attention_state_mla(self, mg_attn, hf_state_dict, hf_prefix, to_mcore):
    """MLA attention 转换"""

    if to_mcore:
        # HF → Megatron: 直接映射
        self._set_state_dict(mg_attn, 'linear_q_down_proj.weight',
                             hf_state_dict, 'q_a_proj.weight', to_mcore)
        self._set_state_dict(mg_attn, 'linear_q_up_proj.weight',
                             hf_state_dict, 'q_b_proj.weight', to_mcore)
        self._set_state_dict(mg_attn, 'linear_kv_down_proj.weight',
                             hf_state_dict, 'kv_a_proj_with_mqa.weight', to_mcore)
        self._set_state_dict(mg_attn, 'linear_kv_up_proj.weight',
                             hf_state_dict, 'kv_b_proj.weight', to_mcore)
    else:
        # Megatron → HF: 反向映射
        ...
```

#### 2. MTP (Multi-Token Prediction) - Qwen3-Next

**特点：** 多头预测下一个 token

**HF 格式:**
```python
lm_head.weight: [vocab_size, hidden_size]
lm_heads.{i}.weight: [vocab_size, hidden_size]  # i = 0, 1, 2, ...
```

**Megatron 格式:**
```python
output_layer.weight: [vocab_size, hidden_size]
additional_lm_heads.{i}.weight: [vocab_size, hidden_size]
```

**代码位置:** `/home/scbjtfy/ms-swift/swift/megatron/model/gpt/qwen3_next.py`

#### 3. Multimodal - Vision + Language

**架构示例（Qwen2-VL）:**

```
┌─────────────────────────────────────────────┐
│              Qwen2-VL Model                  │
├─────────────────────────────────────────────┤
│                                              │
│  ┌──────────────────┐                       │
│  │ Vision Encoder   │                       │
│  │ (ViT)            │                       │
│  │ visual.conv1     │                       │
│  │ visual.blocks.*  │                       │
│  │ visual.merger    │                       │
│  └────────┬─────────┘                       │
│           │                                  │
│  ┌────────▼─────────┐                       │
│  │ Aligner/Resampler│                       │
│  │ (reduce tokens)  │                       │
│  └────────┬─────────┘                       │
│           │                                  │
│  ┌────────▼─────────┐                       │
│  │ Language Model   │                       │
│  │ model.embed_tokens│                       │
│  │ model.layers.*   │                       │
│  │ lm_head          │                       │
│  └──────────────────┘                       │
└─────────────────────────────────────────────┘
```

**Bridge 实现:**

```python
# 位置：gpt_bridge.py lines 1477-1481
class MultimodalGPTBridge(GPTBridge):
    hf_layers_prefix = 'model.language_model.layers'
    # HF: model.language_model.layers.*
    # Megatron: decoder.layers.*

    def _convert_pre_process(self, mg_models, hf_state_dict, hf_prefix, to_mcore):
        """处理 vision encoder 和 aligner"""

        # 1. 处理 language model embeddings
        hf_state_dict = super()._convert_pre_process(...)

        # 2. 处理 vision encoder (通过 module_mapping)
        if self.module_mapping:
            for hf_module_name, mg_module_name in self.module_mapping.items():
                mg_module = getattr(mg_models[0], mg_module_name, None)
                if mg_module:
                    hf_state_dict.update(
                        self._set_module(mg_module, hf_state_dict,
                                        f'{hf_module_name}.', to_mcore)
                    )

        return hf_state_dict

# 子类定义 module_mapping
class Qwen2_5VLBridge(MultimodalGPTBridge):
    def __init__(self, ...):
        super().__init__(...)
        # 定义模块映射
        self.module_mapping = {
            'visual': 'vision_model',      # HF visual → Megatron vision_model
            'aligner': 'vision_projection', # HF aligner → Megatron vision_projection
        }
```

### 错误处理与验证

**1. 检查点不完整：**

```python
# 位置：gpt_bridge.py lines 235-242
incompatible_keys = mg_module.load_state_dict(hf_state_dict, strict=False)
missing_keys = incompatible_keys.missing_keys

if self._is_peft_format:
    # PEFT 格式：只检查 LoRA keys
    missing_keys = [k for k in missing_keys if '.lora_A.' in k or '.lora_B.' in k]

assert len(missing_keys) == 0, \
    f'Incomplete checkpoint: missing_keys={missing_keys}'
```

**2. LoRA 一致性检查：**

```python
# 位置：gpt_bridge.py lines 413-422
# QKV fusion 要求 Q/K/V 的 lora_A 相同
q_lora_A = hf_state_dict['q_proj.lora_A.weight']
k_lora_A = hf_state_dict['k_proj.lora_A.weight']
v_lora_A = hf_state_dict['v_proj.lora_A.weight']

assert (q_lora_A == k_lora_A).all() and (k_lora_A == v_lora_A).all(), \
    "Q/K/V must share the same lora_A for QKV fusion in Megatron. " \
    "Please retrain with shared lora_A or use merge_lora=True."
```

**3. 版本兼容性：**

```python
# 位置：gpt_bridge.py lines 25, 52
import megatron.core
mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')
mcore_014 = version.parse(megatron.core.__version__) >= version.parse('0.14.0rc0')

# 根据版本调整行为
if not self.mcore_014:
    # Megatron-Core < 0.14: MLA 使用不同的 projection 名称
    dim0_keys.update({'linear_q_down_proj', 'linear_kv_down_proj'})
```

---

## 扩展性分析

### 添加新模型支持

**步骤：**

1. **定义模型架构**（如果 Megatron-Core 未支持）
   - 文件：`swift/megatron/model/gpt/<model_name>.py`
   - 继承 `megatron.core.models.gpt.GPTModel`

2. **创建 Bridge 子类**（如果有特殊转换逻辑）
   - 文件：`swift/megatron/model/gpt_bridge.py`
   - 继承 `GPTBridge` 或 `MultimodalGPTBridge`

3. **注册模型**
   - 文件：`swift/megatron/model/register.py`

**示例：添加 LLaMA 4 支持**

```python
# 1. 定义架构（如果需要）
# swift/megatron/model/gpt/llama4.py
from megatron.core.models.gpt import GPTModel

def llama4_model_provider():
    # 使用标准 GPTModel，配置特定参数
    return GPTModel(
        config=...,
        transformer_layer_spec=...,  # 自定义 layer spec
    )

# 2. 创建 Bridge（如果需要）
# 如果转换逻辑与 GPT 相同，可以不创建
class LLaMA4Bridge(GPTBridge):
    # 只覆盖有差异的部分
    def _set_attention_state(self, mg_attn, hf_state_dict, hf_prefix, to_mcore):
        # LLaMA 4 特殊的 attention 处理
        ...

# 3. 注册
# swift/megatron/model/register.py
@dataclass
class MegatronModelMeta:
    megatron_model_type: str
    model_types: List[str]
    bridge_cls: Type[GPTBridge]
    model_cls: Type[nn.Module]

register_megatron_model(
    MegatronModelMeta(
        megatron_model_type='llama4',
        model_types=['llama4'],  # ms-swift 中的 model_type
        bridge_cls=LLaMA4Bridge,
        model_cls=None,  # 使用默认 GPTModel
    )
)
```

### 添加新的并行策略

**示例：Sequence Parallelism with All-to-All**

```python
# 1. 在 gpt_bridge.py 中添加 all-to-all 逻辑
class GPTBridge:
    def __init__(self):
        # 添加 SP group
        self.sp_size = self.args.sequence_parallel_size
        self.sp_group = mpu.get_sequence_parallel_group()
        self.sp_rank = mpu.get_sequence_parallel_rank()

    def _all_to_all_sp(self, tensor):
        """Sequence Parallelism all-to-all"""
        if self.sp_size == 1:
            return tensor

        # All-to-all: [bs, seq/sp, hidden*sp] → [bs, seq, hidden]
        output = torch.empty_like(tensor)
        dist.all_to_all_single(
            output, tensor,
            output_split_sizes=[tensor.shape[1]] * self.sp_size,
            input_split_sizes=[tensor.shape[1]] * self.sp_size,
            group=self.sp_group,
        )
        return output

    def _get_weight(self, mg_weight, mg_key, ...):
        # 在 TP gather 之后，PP broadcast 之前
        tensor = self._all_gather_tp(tensor, tp_dim, is_expert)

        # 新增：SP all-to-all
        if self.sp_size > 1:
            tensor = self._all_to_all_sp(tensor)

        tensor = self._broadcast_ep_pp(tensor, is_expert)
        return tensor

# 2. 在 convert.py 中传递 SP 参数
def convert_hf_config(config):
    return {
        ...
        'sequence_parallel_size': args.sequence_parallel_size,
    }
```

### 性能监控与调试

**添加性能统计：**

```python
# 在 gpt_bridge.py 中添加
import time
from collections import defaultdict

class GPTBridge:
    def __init__(self):
        self.profiling = defaultdict(float)

    def _set_weight(self, mg_param, hf_weight, mg_key, ...):
        start = time.time()

        # 原有逻辑
        ...

        self.profiling[f'set_weight.{mg_key}'] += time.time() - start

    def print_profiling(self):
        """打印性能统计"""
        total = sum(self.profiling.values())
        for key, value in sorted(self.profiling.items(), key=lambda x: -x[1]):
            percentage = value / total * 100
            logger.info(f'{key}: {value:.2f}s ({percentage:.1f}%)')

# 使用
bridge.load_weights(mg_model, model_dir)
bridge.print_profiling()
```

---

## 总结

**Mcore-Bridge 的核心创新：**

1. **双向无缝转换：** 真正实现 HF ↔ Megatron 的零摩擦转换
2. **并行策略透明：** 自动处理 TP/PP/EP/CP/VPP，用户无需手动配置权重切分
3. **内存高效：** Lazy loading + Streaming saving，支持超大模型（235B+）
4. **LoRA 完整支持：** 包括增量权重转换、merge-lora、精度验证
5. **可扩展架构：** 基于 Bridge pattern，轻松添加新模型/新策略
6. **生产就绪：** 完善的错误处理、版本兼容、精度验证

**技术亮点：**

- **Generator Pattern** 避免内存峰值
- **Template Method** 统一转换流程
- **Strategy Pattern** 灵活的并行策略
- **Lazy Initialization** 延迟加载优化
- **Multi-threading** 并行写入加速

**适用场景：**

- ✅ 使用 HF checkpoint 直接训练 Megatron（无需预转换）
- ✅ 训练后直接保存为 HF 格式（无需后转换）
- ✅ LoRA 训练 + 导出（支持增量和 merge 两种格式）
- ✅ 超大模型（235B+）的分布式训练
- ✅ MoE 模型的高效训练（EP 加速）
- ✅ Multimodal 模型的端到端训练

这个系统真正实现了 **"making Megatron training as simple and easy to use as transformers"** 的目标。
