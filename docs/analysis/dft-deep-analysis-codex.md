# Dynamic Fine-Tuning (DFT) 在 ms-swift 中的实现深度分析（Codex）

> 本文基于 ms-swift 当前源码梳理 DFT（Dynamic Fine-Tuning）在 **SFT** 训练中的实现方式与工程兼容性。  
> 代码中该开关名为 `enable_dft_loss`，其 docstring 里也出现过 “DFD loss” 的表述（命名差异不影响实现本质）。

---

## 1. 功能入口与启用方式

- HF/Swift Trainer 路径：`swift/trainers/arguments.py` 定义 `TrainArgumentsMixin.enable_dft_loss: bool = False`
- Megatron-SWIFT 路径：`swift/megatron/argument/megatron_args.py` 定义 `enable_dft_loss: bool = False`

启用示例（SFT）：

```bash
swift sft ... --enable_dft_loss true
```

> 注意：如果你启用了 HF 的 `label_smoothing_factor`（从而启用 `label_smoother`），则 **最终用于反向传播的 loss 可能不再使用 DFT 加权的 per-token loss**（详见 §4.3）。

---

## 2. 数学形式：DFT 如何改写 Cross Entropy

ms-swift 的 DFT 本质是对 **token-level cross entropy** 做一个与模型当前置信度相关的权重缩放。

对一个监督 token（target id 为 `y`）：

- 标准 CE：  
  \[
  \ell = -\log p(y \mid x)
  \]
- DFT 权重：  
  \[
  w = \exp(-\ell) = p(y \mid x)
  \]
- DFT 加权后的 token loss：  
  \[
  \ell_{\text{dft}} = w \cdot \ell = p(y \mid x)\cdot (-\log p(y \mid x))
  \]

### 2.1 梯度层面的含义（关键）

ms-swift 在实现中对 `w` 使用了 `detach()`（见 §3.2），即 **不让梯度穿过权重分支**。因此对 logits 的梯度等价于：

> `grad(DFT-CE) = w * grad(CE)`，其中 `w = p(y|x)` 被当作常数。

这意味着：
- **难 token（`p(y|x)` 很小）** 的梯度会被显著缩小（更“保守”/更抗噪）
- 与 **Focal Loss（下调 easy 样本）** 相反，DFT 更像是下调 hard token 的贡献

---

## 3. Swift Trainer（Transformers/Accelerate）路径实现

### 3.1 触发条件：为何会走 per-token loss 路径

在 `swift/trainers/trainers.py` 的 `Seq2SeqTrainer.compute_loss()` 中，当满足下述任意条件，会把 `labels` 从 `inputs` 中弹出（使模型 forward 不再返回内置 loss），并改为 **手工计算 loss**：

- `self.args.enable_dft_loss`
- `loss_scale is not None`
- `self.args.enable_channel_loss`
- `self.template.sequence_parallel_size > 1`
- `label_smoother is not None` / `compute_loss_func is not None`（也会进入“手工 loss”分支，但最终 loss 的选择见 §4.3）

对应代码位置：`swift/trainers/trainers.py`（`compute_loss()` 内的条件判断）。

### 3.2 核心算子：`per_token_loss_func(...)`

DFT 的核心逻辑位于 `swift/trainers/utils.py:per_token_loss_func`：

1) 取 logits 并上采样到 float（避免精度问题）  
2) 对 labels 做 causal shift：`labels = torch.roll(labels, shifts=-1, dims=-1)`  
3) `F.cross_entropy(..., reduction="none", ignore_index=-100)` 得到 per-token loss  
4) **若启用 DFT**：  
   - `target_probs = exp(-loss)`（在 `torch.no_grad()` 下）
   - `loss *= target_probs`

关键片段（语义化摘录）：

```python
loss = F.cross_entropy(logits, labels, ignore_index=-100, reduction='none')
if enable_dft_loss:
    with torch.no_grad():
        target_probs = torch.exp(-loss)  # p(y|x)
    loss *= target_probs                # p(y|x) * (-log p(y|x))
```

### 3.3 与 `logits_to_keep` 的兼容（显存/速度优化不破坏）

ms-swift 默认会在满足条件时启用 `logits_to_keep`（见 `swift/trainers/mixin.py:get_use_logits_to_keep()`），并在 forward 前对 batch 做一次轻量变换（`swift/trainers/mixin.py:prepare_logits_to_keep()`）：

- 让 `labels` 只保留需要计算 loss 的尾部区间（或监督 token 子集）
- 把 `inputs["logits_to_keep"]` 传给支持该参数的模型 forward，从源头减少 logits 计算/保存

DFT 的实现只依赖 “最终用于 CE 的 logits 与 labels”，因此在 `labels/logits` 被裁剪后仍然成立：**DFT 不需要额外的张量对齐信息**。

---

## 4. 与其他训练特性的组合关系

### 4.1 与 `loss_scale`（token 权重）叠加

在 `swift/trainers/trainers.py:compute_loss()` 中，`loss_scale` 在 DFT 加权之后再相乘：

> 最终 token 权重 = `p(y|x)`（DFT） × `loss_scale`（模板/策略定义）

这也意味着：如果你同时开了 `--enable_channel_loss`，记录到 `loss_<channel>` 的值也是 **经过 DFT 与 loss_scale 共同加权后的 loss**。

### 4.2 与 Packing / Padding-Free（FlashAttention varlen）兼容

在 Swift Trainer 路径中，Packing / padding-free 会把一个 batch 展平为 “batch=1 的长序列”，并通过 `position_ids` 边界来做统计/切分（如 channel loss）。

DFT 本身只对 per-token loss 做逐元素缩放：
- 不依赖 batch 维度语义
- 不依赖 `cu_seqlens`
- 不引入额外通信

因此对 packing/padding-free 的兼容性天然良好；其性能开销相对 CE 可忽略（额外是 `exp` + `mul` 的 O(tokens)）。

### 4.3 与 `label_smoother` / 自定义 `compute_loss_func` 的交互（容易踩坑）

`Seq2SeqTrainer.compute_loss()` 的最终标量 loss 有多条路径：

- 若提供 `compute_loss_func`：由用户函数决定是否使用 `outputs.loss`（DFT 后的 per-token loss）
- 否则若 `label_smoother is None`：用 `outputs.loss.sum() / num_items_in_batch`（DFT 生效）
- 否则（启用 label smoothing）：走 `label_smoother(outputs, labels, shift_labels=True)`  
  - **此时 DFT 计算出来的 `outputs.loss` 通常不会被用作反向传播的 loss**

结论：想确保 DFT 生效，建议保持 `label_smoothing_factor=0`（默认）或在自定义 `compute_loss_func` 中显式使用 DFT 的 per-token loss。

---

## 5. Sequence Parallel（Swift 自身序列并行）路径

当 `self.template.sequence_parallel_size > 1` 时，会走 `per_token_loss_func_sp(...)`（`swift/trainers/utils.py`）：

- CE 计算可能使用 `ChunkedCrossEntropyLoss`（由 `CELOSS_PARALLEL_SIZE` 控制）以降低峰值显存
- DFT 权重同样是 `exp(-loss.detach())`，在本地逐 token 应用
- 随后通过 `GatherLoss.apply(...)` 等逻辑把 loss/labels 对齐到序列并行语义

工程含义：DFT 不引入新的 collective；仅复用既有 SP 的 loss 聚合流程。

---

## 6. Megatron-SWIFT 路径实现（与 ZeRO3/FSDP 思路不同，但同样高效）

Megatron-SWIFT 的 `loss_func(...)` 位于 `swift/megatron/trainers/trainer.py`，其输入 `output_tensor` 在该训练栈里通常已经是 per-token loss（或等价张量）。

实现逻辑（语义化）：

```python
losses = output_tensor.float()
loss_mask = labels != -100
if args.enable_dft_loss:
    losses = losses * torch.exp(-losses.detach())
```

随后再叠加 `loss_scale`、channel 统计，并按 Megatron 既有 DP/CP 规则做 all-reduce。

工程要点：
- DFT 的操作位置在 “loss 张量” 上，不触碰参数分片策略（TP/PP/CP/DP）
- 不新增通信：只是在既有损失聚合之前做一次逐元素缩放

---

## 7. 设计总结：为什么它不破坏分布式效率优化

从系统设计视角，ms-swift 的 DFT 有三个关键工程特征，使其对各种并行/显存优化策略天然友好：

1) **只做逐 token 的 pointwise 变换**：`loss *= exp(-loss.detach())`  
   - 计算量 O(tokens)，相对 CE（O(tokens·vocab)）几乎可忽略
2) **`detach()` 让权重不进入反向图**  
   - 避免二阶/额外梯度路径
   - 不引入额外通信依赖与图复杂度
3) **复用既有 loss reduction/同步机制**  
   - HF/Accelerate：仍然是标量 loss 的正常 backward + DDP/ZeRO 的梯度同步
   - Megatron：仍然走 Megatron 的 DP/CP loss 规约流程

---

## 8. 便于移植的伪代码（Trainer 路径）

```python
# logits: [B, T, V], labels: [B, T]
labels_shift = roll(labels, -1)                   # causal shift
loss_tok = CE(logits, labels_shift, reduction=none, ignore=-100)  # [B*T]
if enable_dft:
    w = exp(-detach(loss_tok))                    # p(y|x)
    loss_tok = loss_tok * w

loss_tok = loss_tok * loss_scale(optional)
loss = sum(loss_tok) / num_supervised_tokens      # scalar for backward
```

