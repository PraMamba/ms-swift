# Ulysses + Ring-Attention 序列并行：ms-swift 实现深度分析（Codex）

> 目标问题：ms-swift 如何实现 “**Ulysses can now be used with ring-attention**”，从而让 **sequence 可以切成任意数量的 chunks**（不再受 `num_heads` 限制）？
>
> 结论一句话：ms-swift 将用户指定的 `sequence_parallel_size` 分解为 **SP(Ulysses)** × **RP(ring-attention)** 两个维度，其中  
> `sp_world_size = gcd(num_heads, sequence_parallel_size)`，`rp_world_size = sequence_parallel_size / sp_world_size`，  
> 让 head-divisibility 约束只落在 `sp_world_size` 上；剩余因子交给 ring-attention 在 **序列维度**上继续切分。

---

## 1. 背景：为什么 “原生 Ulysses” 会受 num_heads 限制？

Ulysses（这里指 DeepSpeed 风格的 sequence parallel / all-to-all attention）本质是在 attention 内部做一次 `all_to_all`：

- 输入侧（每卡）通常是 **局部序列** + **全量 heads**：`[B, local_seq, num_heads, head_dim]`
- 通过 `all_to_all` 交换后变成 **全局序列** + **局部 heads**：`[B, global_seq, num_heads/sp, head_dim]`

由于 Ulysses 会把 heads 在 `sp_world_size` 上均分，因此它有一个硬约束：

> `num_heads % sp_world_size == 0`

在 ms-swift 的实现里，这个约束直接写在 Ulysses 的 layout 生成里（`swift/trainers/sequence_parallel/ulysses.py`）：
- `_generate_layout_params(...)`：`assert num_total_head % seq_world_size == 0`

因此如果你把 `sequence_parallel_size` 设成一个不整除 heads 的数（或大于 heads 的数），纯 Ulysses 就会失败——这就是 “limited by number of heads” 的来源。

---

## 2. ms-swift 的总体实现位置（入口与关键文件）

### 2.1 启用入口Feature A

在 SFT/CPT 等 HF Trainer 路径中，当 `--sequence_parallel_size > 1` 时会初始化 SP：

- `swift/llm/train/sft.py`：`sequence_parallel.prepare(args.sequence_parallel_size, ..., padding_free=args.padding_free)`
- `swift/trainers/trainers.py:_prepare_inputs()`：`sequence_parallel.prepare_inputs(inputs)`（在 forward 前切分 batch）

### 2.2 核心文件

- **并行组划分 + all2all(Ulysses) + attention patch + 输入切分**：`swift/trainers/sequence_parallel/ulysses.py`
- **ring-attention kernel（zigzag + flash-attn varlen）**：`swift/trainers/sequence_parallel/zigzag_ring_attn.py`
- **ring 通信原语**：`swift/trainers/sequence_parallel/utils.py:RingComm`

---

## 3. “不再受 num_heads 限制”的关键：SP×RP 分解（GCD 分配策略）

ms-swift 在 `SequenceParallel._init_device_mesh()`（`swift/trainers/sequence_parallel/ulysses.py`）用一个非常工程化的规则把用户的 `sequence_parallel_size`（记为 `W`）拆成两段：

```python
sp_world_size = gcd(num_heads, W)   # Ulysses 维度（满足 head 可整分）
rp_world_size = W // sp_world_size  # Ring 维度（剩余因子给 ring-attn）
```

然后构造 device mesh：
- 无 ring：`mesh_dim_names=('data', 'sequence')`
- 有 ring：`mesh_dim_names=('data', 'ring', 'sequence')`

对应代码位置：
- `swift/trainers/sequence_parallel/ulysses.py:_init_device_mesh()`

### 3.1 这为什么能 “任意 chunks”？

因为 head-divisibility 只要求 `num_heads % sp_world_size == 0`。而 `sp_world_size = gcd(num_heads, W)` **天然满足**：

- `sp_world_size | num_heads`
- `sp_world_size | W`

所以无论你把 `W` 设成什么值（只要是正整数、且参与分布式 world_size 划分），ms-swift 都能：

1) 取出一段 “heads 可整分” 的 `sp_world_size` 给 Ulysses  
2) 把剩余因子给 `rp_world_size`，用 ring-attention 在序列维度继续切分

### 3.2 直观例子

假设 `num_heads=32`：

- 你想用 `W=24`：  
  - `gcd(32,24)=8` → `sp=8`、`rp=3`  
  - Ulysses 只在 8 卡上分 heads（合法），序列再在 ring 维度切 3 份（zigzag 后是 6 个 chunks）
- 你想用 `W=48`（明显大于 heads）：  
  - `gcd(32,48)=16` → `sp=16`、`rp=3`  
  - 仍然可行：heads 只按 16 分，额外的 3 倍并行交给 ring-attn
- 你想用一个质数 `W=7`：  
  - `gcd(32,7)=1` → `sp=1`、`rp=7`  
  - 这时没有 Ulysses all2all（不分 heads），完全依赖 ring-attn 把序列切成 7 份（zigzag 后 14 个 chunks）

这就是 “no longer limited by number of heads” 的工程实现含义：**把 head 约束从 `W` 挪到 `gcd(num_heads,W)`。**

---

## 4. 两级并行在 attention 中如何衔接：先 Ulysses，再 ring-attention

### 4.1 attention 包装器：`DistributedAttention`

`swift/trainers/sequence_parallel/ulysses.py:DistributedAttention.forward()` 的核心逻辑有一句注释点题：

> `# gather ulysses first, ring-attention next`

具体行为：

1) **Ulysses（SP 维度）**：如果 `sp_world_size > 1`，对 Q/K/V 做 `_SeqAllToAll.apply(self.sp_group, ...)`  
   - 这一步将 “局部序列 + 全 heads” 变换为 “更长序列 + 局部 heads”
2) **Ring-attention（RP 维度）**：如果 `rp_world_size > 1`  
   - 不再走普通 flash-attn 路径，而是走 zigzag ring flash-attn varlen
3) 输出侧再做一次 all2all，把张量 layout 变回下游期望

### 4.2 为什么要 “先 Ulysses 再 ring”？

因为二者解决的是正交问题：

- Ulysses：解决 **heads 维度的分片**（但要求 head 可整分）
- ring-attention：解决 **序列维度的分片**（chunk 数不依赖 heads）

ms-swift 的做法是：

- 让 Ulysses 只吃掉 `sp_world_size=gcd(num_heads,W)` 这部分（永远合法）
- 再在 ring 维度把序列扩展到总并行度 `W=sp×rp`

---

## 5. ring-attention 为什么能 “任意 chunks”：zigzag + ring KV 旋转

ms-swift 的 ring-attention 是基于 flash-attn varlen 的 **block attention + ring 通信**，并用了 **zigzag** 来兼容 causal 并平衡算力。

### 5.1 zigzag 的输入重排（每个 rank 拿两段序列）

在 `SequenceParallel._split_packed()`（`swift/trainers/sequence_parallel/ulysses.py`）里：

- 每个 packed sequence 先被 `chunk(2 * rp_world_size)`
- 每个 rank 拿两个 chunk：`[rp_rank]` 与 `[2*rp_world_size-1-rp_rank]`

这会把序列切成 `2*rp_world_size` 段，并以 “前后配对” 的方式分配到每个 rank（典型 zigzag）。

对应代码位置：
- `swift/trainers/sequence_parallel/ulysses.py:_split_packed()`
- `swift/trainers/sequence_parallel/ulysses.py:pad_and_split_inputs()` 的 docstring 明确描述了 “rp_world_size*2 + zigzag”

### 5.2 ring kernel（flash-attn varlen + KV 轮转）

ring attention 的计算实现在：
- `swift/trainers/sequence_parallel/zigzag_ring_attn.py:zigzag_ring_flash_attn_varlen_forward/backward`

其核心套路：

1) 每步对本地 q 与当前持有的 k/v 做一次 flash-attn varlen block 计算
2) 用 `RingComm.send_recv_kv()`（P2P）把 k/v 在 ring 上轮转一圈
3) 对每步的 block 输出，用 `update_out_and_lse(...)` 做数值稳定的合并（log-sum-exp 语义）

关键通信原语：
- `swift/trainers/sequence_parallel/utils.py:RingComm`

### 5.3 兼容 packing/padding_free：用 position_ids → cu_seqlens

ring-attn 需要 varlen 的 `cu_seqlens/max_seqlen`。ms-swift 统一用 position_ids（packed 时每段从 0 重新计数）推导：
- `swift/utils/torch_utils.py:get_cu_seqlens_from_position_ids()`

并在 ring-attn 路径下对 QKV 做 mask（避免 padding 影响注意力）：
- `swift/trainers/sequence_parallel/ulysses.py:_mask_qkv()`

---

## 6. 训练侧输入如何切分（为什么要求 padding_free）

ring-attn 路径在 `SequenceParallel.prepare()` 里强制：

- `rp_world_size > 1` 必须 `--padding_free true`  
  - 代码位置：`swift/trainers/sequence_parallel/ulysses.py:prepare()`（抛 `NotImplementedError`）

原因来自 `pad_and_split_inputs()` 的设计：

- ring-attn 需要按 **每个 packed 子序列** 做 `2*W` 对齐 padding（否则 zigzag chunk 无法整分）
- 非 padding_free（batch>1 且存在常规 attention_mask）需要另一套 pad/split + mask 的复杂流程，ms-swift 在注释里明确暂未实现

---

## 7. 与 FlashAttention / SDPA 的兼容性边界

### 7.1 ring-attention 仅支持 flash-attn（varlen）

在 `ulysses.py` 的 attention patch 里：
- `local_sdpa_attn` 当 `rp_world_size > 1` 直接 `raise NotImplementedError('SDPA does not support Ring attention.')`
- ring-attn 路径显式调用 `flash_attn.flash_attn_interface._flash_attn_varlen_forward/_backward`

因此：
- **纯 text 且 rp=1**：可用 SDPA 或 flash-attn（取决于模型/环境）
- **启用 ring（rp>1）**：必须走 flash-attn varlen

这也与 FAQ 描述一致（见 `docs/source_en/Instruction/Frequently-asked-questions.md` 的 Q118）。

---

## 8. 总结：ms-swift 的系统设计亮点

1) **GCD 分解**把 “heads 必须可整分” 的限制从 `sequence_parallel_size` 上移走：  
   - Ulysses 只负责 `gcd(num_heads, W)` 这段
   - ring-attn 吃掉剩余因子，实现任意 chunk 数
2) **device mesh 统一管理 DP/RP/SP 三类 group**，避免手写复杂 rank 映射
3) **zigzag + ring KV 轮转**解决 causal ring-attn 的负载不均与正确性问题
4) **position_ids → cu_seqlens** 打通 packing/padding_free 的 varlen 语义，并用 `_mask_qkv` 保证 padding 不污染注意力

如果你要在其他训练框架复现同等能力，建议按相同思路拆分：

- `sp = gcd(num_heads, user_world_size)`
- `rp = user_world_size // sp`
- attention 内部先做 Ulysses all2all（只在 sp 维度），再做 ring-attn（只在 rp 维度）

