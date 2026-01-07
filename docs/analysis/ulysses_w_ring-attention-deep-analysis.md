# MS-Swift Ulysses + Ring-Attention 深度分析

> **作者**: 深度代码分析
> **日期**: 2025-12-28
> **版本**: 1.0
> **关键词**: Ulysses, Ring-Attention, Sequence Parallelism, GCD分解, Zigzag调度

---

## 执行摘要

本文档深入分析 ms-swift 框架如何通过创新的 **GCD（最大公约数）分解策略**，实现 "Ulysses can now be used with ring-attention"，从而突破传统 Ulysses 序列并行的 `num_heads` 限制，使 sequence 可以切分成**任意数量的 chunks**。

**核心创新点**：
1. **GCD 分解**：`sp_world_size = gcd(num_heads, sequence_parallel_size)`
2. **双层并行**：Ulysses 处理头维度 + Ring-Attention 处理序列维度
3. **Zigzag 调度**：平衡因果性与负载的智能分片策略
4. **LSE 累积**：数值稳定的块级输出合并算法

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [核心创新：GCD 分解策略](#2-核心创新gcd-分解策略)
3. [技术架构](#3-技术架构)
4. [Ulysses 与 Ring-Attention 协同机制](#4-ulysses-与-ring-attention-协同机制)
5. [Zigzag 分片策略详解](#5-zigzag-分片策略详解)
6. [LSE 累积算法](#6-lse-累积算法)
7. [代码实现细节](#7-代码实现细节)
8. [训练流程](#8-训练流程)
9. [性能与约束](#9-性能与约束)
10. [总结](#10-总结)

---

## 1. 背景与动机

### 1.1 传统 Ulysses 的限制

**Ulysses 序列并行** 是一种通过在多个 GPU 之间分片注意力头（attention heads）来实现序列并行的方法。其核心约束是：

```
num_heads % sequence_parallel_size == 0
```

这个约束导致了以下问题：

| 模型 | num_heads | 可用并行度 | 限制 |
|------|-----------|-----------|------|
| Qwen2.5-3B | 16 | 1, 2, 4, 8, 16 | 无法使用 24 卡 |
| LLaMA-7B | 32 | 1, 2, 4, 8, 16, 32 | 无法使用 48 卡 |
| GPT-3 | 96 | 1, 2, 3, 4, 6, 8, ... | 无法使用质数卡数 |

**核心痛点**：当 `sequence_parallel_size` 不能整除 `num_heads` 时，传统 Ulysses 无法工作。

### 1.2 Ring-Attention 的互补性

**Ring-Attention** 通过在 GPU ring 上轮转 K/V 来实现序列并行，不受 `num_heads` 限制，但有其他约束：

- 需要 Flash-Attention 2 的 varlen 支持
- 通信开销随 ring 大小线性增长
- 必须使用 padding-free 模式

### 1.3 MS-Swift 的解决方案

MS-Swift 通过 **GCD 分解** 将两种方法结合，实现了：

```
任意 sequence_parallel_size → gcd(num_heads, size) → sp + rp 双层并行
```

**结果**：sequence 可以切分成 `sp_world_size × rp_world_size × 2` 个 chunks，不再受 `num_heads` 硬限制。

---

## 2. 核心创新：GCD 分解策略

### 2.1 数学原理

**代码位置**：`swift/trainers/sequence_parallel/ulysses.py:722-751`

```python
def _init_device_mesh(self):
    """Initialize device mesh for sequence and ring parallel.

    The logic is unified:
    1. Determine the Sequence Parallel (SP) size first based on GCD to satisfy constraints.
    2. Allocate all remaining model parallelism to Ring Parallel (RP).
    """
    rank, local_rank, world_size, local_world_size = get_dist_setting()
    self.dp_world_size = world_size // self.world_size

    # 1. SP size is the greatest common divisor of num_heads and world_size.
    # This guarantees it divides both, satisfying all constraints.
    sp_world_size = math.gcd(self.num_heads, self.world_size)
    self.sp_world_size = sp_world_size

    # 2. RP size is the remaining factor of the model parallel world size.
    # This ensures all GPUs in the model parallel group are used.
    rp_world_size = self.world_size // self.sp_world_size
    self.rp_world_size = rp_world_size

    if self.rp_world_size > 1:
        self.device_mesh = init_device_mesh(
            get_device().split(':')[0],
            mesh_shape=(self.dp_world_size, self.rp_world_size, self.sp_world_size),
            mesh_dim_names=('data', 'ring', 'sequence'))
    else:
        self.device_mesh = init_device_mesh(
            get_device().split(':')[0],
            mesh_shape=(self.dp_world_size, self.sp_world_size),
            mesh_dim_names=('data', 'sequence'))
```

### 2.2 分解示例

假设 `num_heads = 32`，用户设置 `--sequence_parallel_size W`：

| W | gcd(32,W) | sp_world_size | rp_world_size | 实际策略 | Chunks数 |
|---|-----------|---------------|---------------|---------|---------|
| 8 | 8 | 8 | 1 | 纯Ulysses | 8 |
| 12 | 4 | 4 | 3 | Ulysses(4卡分头) + Ring(3卡分序列) | 4×3×2=24 |
| 24 | 8 | 8 | 3 | Ulysses(8卡分头) + Ring(3卡分序列) | 8×3×2=48 |
| 7 | 1 | 1 | 7 | 纯Ring-Attention | 1×7×2=14 |
| 48 | 16 | 16 | 3 | Ulysses(16卡分头) + Ring(3卡分序列) | 16×3×2=96 |

**关键观察**：
- `sp_world_size` 始终满足 `num_heads % sp_world_size == 0`
- `rp_world_size` 接管剩余的并行因子
- 总 chunks 数 = `sp_world_size × rp_world_size × 2`（Ring 使用 zigzag，每个 rank 持有 2 个非连续段）

### 2.3 为什么 GCD 是最优解？

**数学证明**：

1. **Ulysses 约束**：`num_heads % sp_world_size == 0`
2. **全部 GPU 利用**：`sp_world_size × rp_world_size == sequence_parallel_size`

要同时满足这两个约束，`sp_world_size` 必须是 `num_heads` 和 `sequence_parallel_size` 的**公约数**。为了最大化 Ulysses 的效率（减少 Ring 通信开销），应该选择**最大公约数**。

**GCD 属性**：
```python
gcd(a, b) 是 a 和 b 的最大公约数
∀ d | gcd(a, b) → d | a ∧ d | b
∀ d | a ∧ d | b → d | gcd(a, b)
```

因此，`sp_world_size = gcd(num_heads, sequence_parallel_size)` 是：
- **最大**的满足 Ulysses 约束的并行度
- **最小化** Ring-Attention 的通信开销

---

## 3. 技术架构

### 3.1 系统分层

```
┌─────────────────────────────────────────────────────────────┐
│                    数据并行层 (DP)                           │
│   Rank 0-N: 每个 DP rank 处理不同的 batch                   │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
┌───────▼──────────────┐           ┌───────────▼─────────────┐
│  Ring 并行层 (RP)    │           │  Sequence 并行层 (SP)   │
│  处理序列维度分片     │           │  处理头维度分片          │
│  使用 P2P 通信        │           │  使用 All-to-All        │
└──────────────────────┘           └─────────────────────────┘
         │                                      │
         └──────────────┬───────────────────────┘
                        │
                ┌───────▼────────┐
                │  Local Attention│
                │  Flash-Attn 2   │
                └─────────────────┘
```

### 3.2 DeviceMesh 结构

**3D Mesh**（当 `rp_world_size > 1`）：
```python
device_mesh = init_device_mesh(
    'cuda',
    mesh_shape=(dp_world_size, rp_world_size, sp_world_size),
    mesh_dim_names=('data', 'ring', 'sequence')
)
```

**示例**：8 卡，`num_heads=32`，`sequence_parallel_size=4`
- `sp_world_size = gcd(32, 4) = 4`
- `rp_world_size = 4 / 4 = 1`
- `dp_world_size = 8 / 4 = 2`
- Mesh: `(2, 1, 4)` → 2 个数据并行组，每组 4 卡做序列并行

**2D Mesh**（当 `rp_world_size == 1`，纯 Ulysses）：
```python
device_mesh = init_device_mesh(
    'cuda',
    mesh_shape=(dp_world_size, sp_world_size),
    mesh_dim_names=('data', 'sequence')
)
```

---

## 4. Ulysses 与 Ring-Attention 协同机制

### 4.1 DistributedAttention 执行流程

**代码位置**：`swift/trainers/sequence_parallel/ulysses.py:105-162`

```python
class DistributedAttention(torch.nn.Module):
    def forward(self, query, key, value, attention_mask, *args, **kwargs):
        if self.sequence_parallel.world_size == 1:
            return self.local_attn(...)  # SP 禁用

        # ─────────── 步骤 1: Ulysses All-to-All ───────────
        if self.sequence_parallel.sp_world_size > 1:
            # 输入形状：[batch, local_seq, all_heads, head_dim]
            # 输出形状：[batch, global_seq, local_heads, head_dim]
            query_layer = _SeqAllToAll.apply(self.sp_group, query, scatter_idx=2, gather_idx=1)
            key_layer = _SeqAllToAll.apply(self.sp_group, key, scatter_idx=2, gather_idx=1)
            value_layer = _SeqAllToAll.apply(self.sp_group, value, scatter_idx=2, gather_idx=1)
        else:
            query_layer, key_layer, value_layer = query, key, value

        # ─────────── 步骤 2: Ring-Attention ───────────
        if self.sequence_parallel.rp_world_size > 1:
            # 获取真实的 position_ids（支持 MRoPE）
            position_ids = self.sequence_parallel.real_position_ids
            # Pad 到 rp_world_size*2 的倍数
            position_ids = self.sequence_parallel.pad(position_ids, padding_value=-1, position_ids=position_ids)
            # 调用 zigzag_ring_flash_attn_varlen_func
        else:
            # 纯 Ulysses：all-gather position_ids
            position_ids = kwargs.pop('position_ids')
            if position_ids is not None:
                shape0 = position_ids.shape[0]
                position_ids_output = torch.empty(
                    (shape0 * self.sp_world_size, position_ids.shape[1]),
                    dtype=position_ids.dtype, device=position_ids.device)
                dist.all_gather_into_tensor(
                    position_ids_output, position_ids, group=self.sp_group)
                position_ids = torch.cat(position_ids_output.split(shape0, dim=0), dim=1)

        # ─────────── 步骤 3: 本地注意力计算 ───────────
        context_layer = self.local_attn(
            query_layer, key_layer, value_layer, attention_mask,
            *args, position_ids=position_ids, **kwargs)

        # ─────────── 步骤 4: 反向 All-to-All ───────────
        if self.sp_world_size > 1:
            # 恢复形状：[batch, local_seq, all_heads, head_dim]
            output = _SeqAllToAll.apply(self.sp_group, context_layer, gather_idx=2, scatter_idx=1)
        else:
            output = context_layer

        return output
```

### 4.2 执行流程图

```
输入: Q, K, V
  形状: [batch, local_seq, all_heads, head_dim]
  ↓
┌─────────────────────────────────────────────────────┐
│ 步骤 1: Ulysses All-to-All (sp_world_size > 1)     │
│ - scatter_idx=2 (头维度分散)                        │
│ - gather_idx=1 (序列维度聚集)                       │
│ 输出: [batch, global_seq, local_heads, head_dim]   │
└─────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────┐
│ 步骤 2: Ring-Attention (rp_world_size > 1)         │
│ - Zigzag 分片：2*rp 个 chunks                      │
│ - Ring 轮转：P2P send/recv K/V                     │
│ - LSE 累积：数值稳定的输出合并                      │
└─────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────┐
│ 步骤 3: 本地 Flash-Attention                        │
│ - 使用 flash_attn_varlen_forward                   │
│ - 或 zigzag_ring_flash_attn_varlen_func            │
└─────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────┐
│ 步骤 4: 反向 Ulysses All-to-All                     │
│ - gather_idx=2 (头维度聚集)                         │
│ - scatter_idx=1 (序列维度分散)                      │
│ 输出: [batch, local_seq, all_heads, head_dim]      │
└─────────────────────────────────────────────────────┘
  ↓
输出: Attention Output
```

### 4.3 张量形状变换

假设：
- `batch = 2`
- `global_seq = 8192`
- `num_heads = 32`
- `head_dim = 128`
- `sp_world_size = 4`，`rp_world_size = 2`

| 阶段 | Q 形状 | K 形状 | V 形状 |
|------|--------|--------|--------|
| 输入 | `[2, 2048, 32, 128]` | `[2, 2048, 32, 128]` | `[2, 2048, 32, 128]` |
| Ulysses All-to-All | `[2, 8192, 8, 128]` | `[2, 8192, 8, 128]` | `[2, 8192, 8, 128]` |
| Zigzag Split | `[2, 4096, 8, 128]` | `[2, 4096, 8, 128]` | `[2, 4096, 8, 128]` |
| Local Attn | `[2, 4096, 8, 128]` | - | - |
| 反向 All-to-All | `[2, 2048, 32, 128]` | - | - |

---

## 5. Zigzag 分片策略详解

### 5.1 Zigzag 的动机

Ring-Attention 需要满足 **causal attention** 的约束：

```
Attention(Q_i, K_j) 仅当 i ≥ j
```

如果简单地将序列均匀分成 `rp_world_size` 份，会导致：
- **负载不均衡**：早期 rank 的查询需要等待晚期 rank 的 K/V
- **因果性违反**：需要复杂的掩码逻辑

**Zigzag** 通过让每个 rank 持有**两个非连续段**（前半和后半）来解决这个问题。

### 5.2 Zigzag 分片算法

**代码位置**：`swift/trainers/sequence_parallel/ulysses.py:567-608`

```python
def _split_packed(self, value, cu_seqlens, dim=1):
    """Split and re-group in zigzag

    将序列分成 2*rp_world_size 个 chunks，每个 rank 拿两个：
    - 前部分：chunks[rp_rank]
    - 后部分：chunks[2*rp_world_size - 1 - rp_rank]
    """
    local_values = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        sub_value = value[:, start:end]

        # 分成 2*rp_world_size 个 chunks
        local_value = sub_value.chunk(2 * self.rp_world_size, dim=dim)

        # 每个 rank 拿两个非连续的 chunks
        local_values.extend([
            local_value[self.rp_rank],                          # 前部分
            local_value[2 * self.rp_world_size - 1 - self.rp_rank],  # 后部分
        ])
    return torch.cat(local_values, dim=dim).contiguous()
```

### 5.3 Zigzag 示意图

假设 `rp_world_size = 4`，序列分成 8 段（0-7）：

```
序列段:    0    1    2    3    4    5    6    7
          ┌────┬────┬────┬────┬────┬────┬────┬────┐
Rank 0:   │ 0  │    │    │    │    │    │    │ 7  │  (最前 + 最后)
          └────┴────┴────┴────┴────┴────┴────┴────┘
Rank 1:   │    │ 1  │    │    │    │    │ 6  │    │
Rank 2:   │    │    │ 2  │    │    │ 5  │    │    │
Rank 3:   │    │    │    │ 3  │ 4  │    │    │    │  (中间两段)
```

**配对逻辑**：
- Rank `i` 持有段 `[i]` 和 `[2*rp_world_size - 1 - i]`
- Rank 0: `[0, 7]`，Rank 1: `[1, 6]`，Rank 2: `[2, 5]`，Rank 3: `[3, 4]`

### 5.4 Zigzag 的优势

**1. 负载均衡**：每个 rank 持有相同长度的序列

**2. 因果性满足**：

```
考虑 Rank 1（持有段 [1, 6]）的查询：

Step 0（本地 K/V）:
    Q[1] ✅ K[1]   ← 因果（i==j）
    Q[1] ❌ K[6]   ← 违反（i<j）
    Q[6] ✅ K[1]   ← 满足（i>j）
    Q[6] ✅ K[6]   ← 因果（i==j）
    使用 causal=True

Step 1（K/V 来自 Rank 0，段 [0, 7]）:
    Q[1] ✅ K[0]   ← 满足（i>j）
    Q[1] ❌ K[7]   ← 违反（i<j）
    Q[6] ✅ K[0]   ← 满足（i>j）
    Q[6] ❌ K[7]   ← 违反（i<j）
    只需计算 Q 与 K[0] 的注意力
    使用 causal=False

Step 2（K/V 来自 Rank 3，段 [3, 4]）:
    Q[1] ❌ K[3]   ← 违反（i<j）
    Q[1] ❌ K[4]   ← 违反（i<j）
    Q[6] ✅ K[3]   ← 满足（i>j）
    Q[6] ✅ K[4]   ← 满足（i>j）
    只需计算 Q[6] 与 K[3,4] 的注意力
    使用 causal=False
```

**3. 通信效率**：通过分阶段计算（`step <= comm.rank` vs `step > comm.rank`），避免不必要的计算和通信。

### 5.5 Ring-Attention 前向传播

**代码位置**：`swift/trainers/sequence_parallel/zigzag_ring_attn.py:290-370`

```python
def zigzag_ring_flash_attn_varlen_forward(
    process_group, q, k, v, cu_seqlens, max_seqlen,
    half_index0, half_index1,  # 前半和后半的索引
    softmax_scale, dropout_p=0, causal=True,
    window_size=(-1, -1), alibi_slopes=None, deterministic=False
):
    comm = RingComm(process_group)
    q, k, v = squeeze_batch(q, k, v)
    q1 = q[half_index1]  # 后半部分查询

    # 缩放 cu_seqlens 和 max_seqlen
    cu_seqlens = cu_seqlens // comm.world_size
    max_seqlen = max_seqlen // comm.world_size
    block_seq_len = q.shape[0] // 2

    out = None
    lse = None  # log-sum-exp 累积器

    # 循环轮转 K/V
    for step in range(comm.world_size):
        # 发起非阻塞的 P2P 通信
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)

        # ─────────── Step 0: 本地 K/V，causal=True ───────────
        if step == 0:
            block_out, block_lse = forward(
                q, k, v, causal=True, cu_seqlens, max_seqlen,
                block_seq_len, dropout_p, softmax_scale, alibi_slopes, window_size)
            out, lse, sig_diff = update_out_and_lse(out, lse, block_out, block_lse)

        # ─────────── Step 1-rank: 前半 Q 与早期 K/V ───────────
        elif step <= comm.rank:
            k0 = k[half_index0]  # K/V 的前半
            v0 = v[half_index0]
            block_out, block_lse = forward(
                q, k0, v0, causal=False, cu_seqlens, max_seqlen,
                block_seq_len, dropout_p, softmax_scale, alibi_slopes, window_size)
            out, lse, sig_diff = update_out_and_lse(out, lse, block_out, block_lse)

        # ─────────── Step rank+1-end: 后半 Q 与晚期 K/V ───────────
        else:
            block_out, block_lse = forward(
                q1, k, v, causal=False, cu_seqlens, max_seqlen,
                block_seq_len, dropout_p, softmax_scale, alibi_slopes, window_size)
            # 只更新后半的输出
            out[half_index1], lse[half_index1], sig_diff = \
                update_out_and_lse(out[half_index1], lse[half_index1], block_out, block_lse)

        # 等待通信完成
        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    # 输出格式转换
    out = out.to(q.dtype)
    lse = lse.squeeze(-1).transpose(0, 1)
    return out.unsqueeze(0), lse.unsqueeze(0)
```

**关键点**：
1. **Step 0**：所有 Q 与本地 K/V 做因果注意力
2. **Step 1 ~ rank**：所有 Q 与来自早期 rank 的 K/V 前半部分做非因果注意力
3. **Step rank+1 ~ end**：后半 Q 与来自晚期 rank 的 K/V 做非因果注意力

---

## 6. LSE 累积算法

### 6.1 为什么需要 LSE？

在 Ring-Attention 中，每个 step 计算一个 **局部注意力输出** `block_out` 和对应的 **LSE**（log-sum-exp）。为了合并多个 step 的输出，不能简单相加，而需要：

```python
# 错误的做法
out = block_out1 + block_out2 + ...  # ❌ 忽略了 softmax 权重

# 正确的做法
out = softmax_normalize(exp(lse1) * block_out1 + exp(lse2) * block_out2 + ...)  # ✅
```

但直接计算 `exp(lse)` 会导致**数值溢出**。LSE 累积算法通过 **logsumexp trick** 实现数值稳定的合并。

### 6.2 LSE 更新公式

**数学推导**：

给定累积输出 `out` 和 LSE `lse`，以及新的块输出 `block_out` 和 `block_lse`，新的累积输出为：

```
new_lse = log(exp(lse) + exp(block_lse))
        = lse + log(1 + exp(block_lse - lse))

new_out = (exp(lse) / exp(new_lse)) * out + (exp(block_lse) / exp(new_lse)) * block_out
        = exp(lse - new_lse) * out + exp(block_lse - new_lse) * block_out
```

**数值稳定实现**：使用 sigmoid 函数避免直接计算 exp：

```python
sigmoid(x) = 1 / (1 + exp(-x)) = exp(x) / (1 + exp(x))
```

因此：
```python
exp(block_lse - new_lse) = sigmoid(block_lse - lse)
exp(lse - new_lse) = 1 - sigmoid(block_lse - lse)
```

### 6.3 代码实现

**代码位置**：`swift/trainers/sequence_parallel/zigzag_ring_attn.py:69-99`

```python
def update_out_and_lse(out, lse, block_out, block_lse):
    """Update output and lse:
    new_lse = lse + log(1 + exp(block_lse - lse))
    new_out = exp(lse - new_lse) * out + exp(block_lse - new_lse) * block_out

    Args:
        out: [seqlen, num_heads, hidden_size]
        lse: [num_heads, seqlen, 1]
        block_out: [seqlen, num_heads, hidden_size]
        block_lse: [num_heads, seqlen]

    Returns:
        新的 out, lse, 和 sigmoid(block_lse - lse)
    """
    if out is None:
        # 第一个块：直接使用
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
        sig_diff = None
    else:
        block_out = block_out.to(torch.float32)
        block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

        diff = block_lse - lse
        sig_diff = torch.sigmoid(diff)  # = exp(block_lse) / (exp(lse) + exp(block_lse))

        # 稳定的输出合并
        out = out - sig_diff * (out - block_out)
        # 等价于：out = (1 - sig_diff) * out + sig_diff * block_out

        # 稳定的 LSE 更新
        lse = lse - F.logsigmoid(lse - block_lse)
        # logsigmoid(x) = -log(1 + exp(-x))
        # 因此：lse - logsigmoid(lse - block_lse)
        #     = lse + log(1 + exp(block_lse - lse))
        #     = log(exp(lse) * (1 + exp(block_lse - lse)))
        #     = log(exp(lse) + exp(block_lse))
        #     = new_lse ✅

    return out, lse, sig_diff
```

### 6.4 数值稳定性分析

| 方法 | 公式 | 数值范围 | 稳定性 |
|------|------|----------|--------|
| 直接计算 | `exp(lse) + exp(block_lse)` | `[0, +∞)` | ❌ 溢出 |
| Logsumexp | `log(exp(lse) + exp(block_lse))` | `(-∞, +∞)` | ✅ 稳定 |
| Sigmoid | `sigmoid(diff)` | `[0, 1]` | ✅ 非常稳定 |

**关键优势**：
- `sigmoid` 的值域是 `[0, 1]`，永不溢出
- `logsigmoid` 在 PyTorch 中有优化实现，避免中间计算溢出

---

## 7. 代码实现细节

### 7.1 关键文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `swift/trainers/sequence_parallel/ulysses.py` | 805 | 核心实现：SequenceParallel, DistributedAttention, GCD分解 |
| `swift/trainers/sequence_parallel/zigzag_ring_attn.py` | 677 | Ring-Attention 算法：zigzag 前向/反向传播 |
| `swift/trainers/sequence_parallel/utils.py` | 216 | 通信原语：RingComm, GatherLoss |
| `swift/trainers/sequence_parallel/__init__.py` | ~50 | 全局单例导出 |
| `swift/llm/train/sft.py` | ~1000 | SFT 训练入口：调用 sequence_parallel.prepare() |
| `swift/trainers/trainers.py` | ~500 | 训练器基类：_prepare_inputs, compute_loss |

### 7.2 All-to-All 实现

**代码位置**：`swift/trainers/sequence_parallel/ulysses.py:84-102`

```python
class _SeqAllToAll(torch.autograd.Function):
    """Sequence All-to-All 通信原语，支持自动微分"""

    @staticmethod
    def forward(ctx, group, input, scatter_idx, gather_idx):
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        res = single_all_to_all(input, scatter_idx, gather_idx, group)
        return res

    @staticmethod
    def backward(ctx, *grad_output):
        # 反向传播时交换 scatter_idx 和 gather_idx
        return None, _SeqAllToAll.apply(
            ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx
        ), None, None
```

**Layout 变换**（`scatter_idx=2, gather_idx=1`）：

```python
# 输入：[batch, local_seq, all_heads, head_dim]
# 1. Reshape: [batch, seq_world_size, local_seq/seq_world_size, all_heads, head_dim]
# 2. Permute: [seq_world_size, batch, local_seq/seq_world_size, all_heads, head_dim]
# 3. All-to-All: 每个 rank 发送 1/seq_world_size 给其他 ranks
# 4. Permute: [batch, local_seq/seq_world_size, seq_world_size, all_heads/seq_world_size, head_dim]
# 5. Reshape: [batch, global_seq, local_heads, head_dim]
```

### 7.3 RingComm 通信原语

**代码位置**：`swift/trainers/sequence_parallel/utils.py:165-216`

```python
class RingComm:
    """Ring topology 点对点通信"""

    def __init__(self, process_group):
        self._process_group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)

        # Ring 的发送/接收方向
        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

    def send_recv_kv(self, k, v, k_buffer=None, v_buffer=None):
        """非阻塞的 P2P K/V 交换

        返回：下一个 rank 的 K/V
        """
        next_k = self.send_recv(k, k_buffer)
        next_v = self.send_recv(v, v_buffer)
        self.commit()  # 批量提交 isend/irecv
        return next_k, next_v

    def send_recv(self, to_send, recv_tensor=None):
        res = torch.empty_like(to_send) if recv_tensor is None else recv_tensor
        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []
```

**通信模式**：

```
World Size = 4 的 Ring:

   Rank 0 ──► Rank 1 ──► Rank 2 ──► Rank 3
      ▲                                │
      └────────────────────────────────┘

每个 step：
  - Rank i 发送 K/V 给 Rank (i+1) % 4
  - Rank i 接收 K/V 从 Rank (i-1) % 4
```

### 7.4 Pad 和 Split

**代码位置**：`swift/trainers/sequence_parallel/ulysses.py:471-608`

```python
def pad(self, tensor, padding_value, position_ids=None, dim=1):
    """Pad tensor for sequence parallel

    如果 rp_world_size > 1：pad 到 world_size*2 的倍数
    否则：pad 到 world_size 的倍数
    """
    if self.rp_world_size > 1:
        world_size = self.world_size * 2
    else:
        world_size = self.world_size

    def _do_pad(tensor):
        length = tensor.shape[dim]
        pad_num = world_size - (length % world_size)
        if pad_num == 0 or pad_num == world_size:
            return tensor
        # 创建 padding 张量并拼接
        pad_shape = ((*tensor.shape[:dim], pad_num, *tensor.shape[dim + 1:])
                     if dim != -1 else (*tensor.shape[:dim], pad_num))
        pad = torch.full(pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=dim)

    # 如果有 ring-attention，需要对每个序列单独 pad
    if position_ids is not None and self.rp_world_size > 1:
        cu_seqlens = get_cu_seqlens_from_position_ids(position_ids)
        all_tensors = []
        for i in range(len(cu_seqlens) - 1):
            sub_tensor = tensor[:, cu_seqlens[i]:cu_seqlens[i + 1]]
            all_tensors.append(_do_pad(sub_tensor))
        return torch.cat(all_tensors, dim=dim)

    return _do_pad(tensor)

def split(self, input, dim, position_ids=None):
    """Split tensor for sequence parallel

    如果 rp_world_size > 1：使用 zigzag 分片
    否则：均匀分片
    """
    if self.world_size == 1:
        return input

    if self.rp_world_size > 1:
        cu_seqlens = get_cu_seqlens_from_position_ids(position_ids)
        assert torch.all(cu_seqlens % (2 * self.rp_world_size) == 0)
        value_chunks = self._split_packed(input, cu_seqlens, dim=dim)
        local_value = value_chunks.chunk(self.sp_world_size, dim=dim)[self.sp_rank]
        return local_value.contiguous()
    else:
        rank = self.sp_rank
        dim_size = input.size(dim)
        assert dim_size % self.sp_world_size == 0
        tensor_list = torch.split(input, dim_size // self.sp_world_size, dim=dim)
        return tensor_list[rank].contiguous()
```

---

## 8. 训练流程

### 8.1 初始化阶段

**步骤 1：模型加载时初始化**

**代码位置**：`swift/llm/train/sft.py:55-58`

```python
if args.sequence_parallel_size > 1:
    from swift.trainers.sequence_parallel import sequence_parallel
    sequence_parallel.prepare(
        args.sequence_parallel_size,
        model=self.model,
        tokenizer=self.processor,
        padding_free=args.padding_free
    )
```

**`prepare()` 函数**（`ulysses.py:429-461`）：

1. 获取 `num_heads` 配置：
   ```python
   self.num_heads = HfConfigFactory.get_config_attr(model.config, 'num_key_value_heads')
   if self.num_heads is None:
       self.num_heads = HfConfigFactory.get_config_attr(model.config, 'num_attention_heads')
   ```

2. 计算 SP 和 RP 大小：
   ```python
   sp_world_size = math.gcd(self.num_heads, self.world_size)
   rp_world_size = self.world_size // sp_world_size
   ```

3. 初始化 DeviceMesh

4. Patch Flash-Attention 和 SDPA：
   ```python
   self._prepare_flash_attn(llm_model)
   ```

5. 注册前向 hook：
   ```python
   self._prepare_forward_hook(llm_model)
   ```

6. 检查约束：
   ```python
   if self.rp_world_size > 1 and not self.padding_free:
       raise NotImplementedError('Ring-attention needs --padding_free true')
   ```

### 8.2 训练循环

**步骤 1：输入准备**

**代码位置**：`swift/trainers/trainers.py:289-295`

```python
def _prepare_inputs(self, inputs):
    inputs = super()._prepare_inputs(inputs)
    if self.template.sequence_parallel_size > 1:
        from swift.trainers.sequence_parallel import sequence_parallel
        sequence_parallel.prepare_inputs(inputs)
    return inputs
```

**`prepare_inputs()`** 会：
- 保存 `text_position_ids` 到 `extra_kwargs`
- Pad 和 split `labels`

**步骤 2：前向传播**

```
Model Forward
  ↓
Embedding Layer
  ↓
[Hook] pre_forward_split_hook: Pad + Split input_ids/position_ids
  ↓
Transformer Layers
  ├─ Self-Attention
  │   ├─ [Patched] _flash_attention_forward
  │   │   ├─ DistributedAttention.forward
  │   │   │   ├─ Ulysses All-to-All
  │   │   │   ├─ Ring-Attention (if rp > 1)
  │   │   │   ├─ Local Flash-Attn
  │   │   │   └─ 反向 All-to-All
  │   └─ ...
  ├─ FFN
  └─ ...
  ↓
LM Head
  ↓
Logits
```

**步骤 3：损失计算**

**代码位置**：`swift/trainers/trainers.py:356-357`

```python
if self.template.sequence_parallel_size > 1:
    outputs.loss = per_token_loss_func_sp(outputs, labels, ...)
```

**`per_token_loss_func_sp()`** 会：
1. 计算逐 token 的 cross-entropy loss
2. 使用 `GatherLoss.apply()` 聚合 SP 维度的损失：
   ```python
   loss, labels = GatherLoss.apply(loss, labels, gather_idx=1, position_ids=position_ids)
   ```
3. 处理 ring-attn 的 zigzag 重排（reverse）

**步骤 4：反向传播**

```
Loss.backward()
  ↓
GatherLoss.backward: Split gradients
  ↓
LM Head.backward
  ↓
Transformer Layers.backward
  ├─ FFN.backward
  ├─ Self-Attention.backward
  │   ├─ DistributedAttention.backward
  │   │   ├─ 反向 All-to-All.backward (swap scatter/gather)
  │   │   ├─ Ring-Attention.backward
  │   │   │   ├─ 重新计算前向 LSE
  │   │   │   ├─ 计算 LSE 梯度
  │   │   │   └─ Ring 轮转梯度
  │   │   └─ Ulysses All-to-All.backward
  └─ ...
  ↓
Embedding.backward
  ↓
Optimizer.step()
```

---

## 9. 性能与约束

### 9.1 性能特性

| 维度 | 纯 Ulysses | Ulysses + Ring | 纯 Ring |
|------|-----------|----------------|---------|
| 通信模式 | All-to-All | All-to-All + P2P | P2P |
| 通信复杂度（每层） | O(1) | O(rp_world_size) | O(rp_world_size) |
| 通信量 | 2×activation_size | 2×activation_size + 2×rp×KV_size | 2×rp×KV_size |
| 内存占用 | ~1/sp | ~1/(sp×rp) | ~1/rp |
| 支持序列长度 | 单卡限制 | 线性扩展 (rp×) | 线性扩展 (rp×) |

**关键发现**：
- **Ulysses** 通信量小但受 `num_heads` 限制
- **Ring** 无限制但通信量随 `rp_world_size` 线性增长
- **Hybrid** 在两者间取得平衡

### 9.2 约束条件

| 约束 | 原因 | 解决方案 |
|------|------|----------|
| `rp_world_size > 1` 需要 `padding_free=true` | Ring-Attn 需要 varlen | 使用 `--padding_free true` |
| Ring-Attn 不支持 SDPA | 需要 Flash-Attn 2 的 varlen 接口 | 使用 `--attn_impl flash_attn` |
| 序列长度需是 `world_size*2` 的倍数（ring） | Zigzag 分片要求 | 自动 padding |
| 序列长度需是 `sp_world_size` 的倍数（ulysses） | All-to-All 均匀分割 | 自动 padding |

### 9.3 性能优化建议

**1. 选择合适的 `sequence_parallel_size`**：

```python
# 优先级：
# 1. 最大化 sp_world_size（减少 Ring 通信）
# 2. 最小化 rp_world_size
# 3. 确保 sp_world_size = gcd(num_heads, size) 尽可能大

# 示例：num_heads = 32
# Bad:  sequence_parallel_size = 48 → sp=16, rp=3 (可行但非最优)
# Good: sequence_parallel_size = 32 → sp=32, rp=1 (纯 Ulysses，最佳)
# OK:   sequence_parallel_size = 16 → sp=16, rp=1 (纯 Ulysses)
```

**2. 使用 `CELOSS_PARALLEL_SIZE` 优化大词表**：

```bash
CELOSS_PARALLEL_SIZE=2048 swift sft ...
```

当词表大小 > 100k 时，这个选项可以减少 logits 的内存占用。

**3. 启用 DeepSpeed ZeRO**：

```bash
swift sft --deepspeed zero3_offload --sequence_parallel_size 8 ...
```

ZeRO 与 SP 兼容，可以进一步减少内存占用。

### 9.4 扩展性分析

**理论序列长度上限**：

```
max_seq_len = per_gpu_memory / (
    activation_size_per_token × batch_size / (sp_world_size × rp_world_size)
)
```

**实际测试**（基于 `examples/train/sequence_parallel/sequence_parallel_512k.sh`）：

| 模型 | 卡数 | SP Size | SP/RP | 最大序列长度 | 备注 |
|------|------|---------|-------|-------------|------|
| QwQ-32B | 8 | 8 | 8/1 | 512k | 使用 ZeRO3 Offload |
| Qwen2.5-VL-3B | 4 | 4 | 4/1 | 128k | DPO 训练 |
| Qwen2.5-3B | 8 | 8 | 8/1 | 1M | 理论值 |

---

## 10. 总结

### 10.1 核心贡献

MS-Swift 的 Ulysses + Ring-Attention 实现通过以下创新突破了传统限制：

1. **GCD 分解策略**：
   ```python
   sp_world_size = gcd(num_heads, sequence_parallel_size)
   rp_world_size = sequence_parallel_size / sp_world_size
   ```
   - 自动满足 Ulysses 约束
   - 利用剩余因子做 Ring-Attention
   - 支持**任意** `sequence_parallel_size`

2. **双层并行架构**：
   - **Ulysses 层**：All-to-All 通信，处理头维度分片
   - **Ring-Attention 层**：P2P 通信，处理序列维度分片
   - 两层协同，各司其职

3. **Zigzag 调度**：
   - 每个 rank 持有两个非连续序列段
   - 满足因果性的同时保证负载均衡
   - 减少不必要的计算和通信

4. **LSE 累积算法**：
   - 使用 sigmoid 实现数值稳定的输出合并
   - 避免 exp 溢出
   - 支持任意多个 block 的累积

### 10.2 实际影响

**突破前**：
```
Qwen2.5-3B (num_heads=16) 只能使用 1/2/4/8/16 卡
→ 24 卡集群：浪费 8 卡或降级到 16 卡
```

**突破后**：
```
Qwen2.5-3B + 24 卡：
  sp_world_size = gcd(16, 24) = 8
  rp_world_size = 24 / 8 = 3
  → 8 卡做 Ulysses + 3 卡做 Ring
  → 支持 3× 长序列（相比 16 卡纯 Ulysses）
```

### 10.3 未来方向

1. **自适应 SP/RP 分解**：
   - 根据模型架构和硬件自动选择最优 `sp_world_size`
   - 考虑通信带宽、内存容量等因素

2. **3D 并行融合**：
   - Tensor Parallel (TP) + Ulysses + Ring
   - 需要设计 TP×SP×RP 的协同策略

3. **异构支持**：
   - 支持 CPU offload 的 Ring-Attention
   - 支持 NVLink/InfiniBand 的自适应通信

4. **算子融合**：
   - 融合 All-to-All 和 Flash-Attention
   - 减少中间 tensor 的开销

---

## 附录

### A. 关键代码位置速查

| 功能 | 文件 | 行数 |
|------|------|------|
| GCD 分解 | `ulysses.py` | 722-751 |
| DistributedAttention | `ulysses.py` | 105-162 |
| Zigzag Split | `ulysses.py` | 567-608 |
| Ring Forward | `zigzag_ring_attn.py` | 290-370 |
| Ring Backward | `zigzag_ring_attn.py` | 373-545 |
| LSE Update | `zigzag_ring_attn.py` | 69-99 |
| RingComm | `utils.py` | 165-216 |
| GatherLoss | `utils.py` | 19-53 |

### B. 参考资源

- **Ring-Attention 原论文**：[arXiv:2310.01889](https://arxiv.org/abs/2310.01889)
- **Ulysses 原论文**：[arXiv:2309.14509](https://arxiv.org/abs/2309.14509)
- **MS-Swift GitHub**：[https://github.com/modelscope/ms-swift](https://github.com/modelscope/ms-swift)
- **Flash-Attention 2**：[https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

---

**文档版本历史**：
- v1.0 (2025-12-28): 初始版本，深度分析 GCD 分解和 Ulysses+Ring 协同机制
