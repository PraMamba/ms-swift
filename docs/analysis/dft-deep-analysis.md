# Dynamic Fine-Tuning (DFT) 深度分析

## 概述

Dynamic Fine-Tuning (DFT) 是 ms-swift 框架支持的一种自适应损失加权技术，通过动态调整每个 token 的损失权重来优化训练过程。DFT 的核心思想是：**让模型聚焦于"中等难度"的样本，避免在已学会的内容上浪费资源，同时也不被噪声或过难样本干扰**。

**论文参考**: https://arxiv.org/abs/2508.05629

## 数学原理

### 标准交叉熵损失

标准的 token 级别交叉熵损失定义为：

$$L_{CE} = -\log P(y|x)$$

其中 $P(y|x)$ 是模型预测正确 token 的概率。

### DFT 动态加权

DFT 在标准损失基础上引入动态权重：

$$L_{DFT} = L_{CE} \cdot \exp(-L_{CE})$$

这个权重函数 $w(L) = \exp(-L)$ 具有以下特性：

| 损失值 L | 权重 w = exp(-L) | 效果 |
|---------|-----------------|------|
| 0.0 (完美预测) | 1.0 | 保持原始权重 |
| 0.5 | 0.607 | 轻微降低 |
| 1.0 | 0.368 | 降低约 63% |
| 2.0 | 0.135 | 降低约 87% |
| 3.0 | 0.050 | 几乎忽略 |

### 权重函数分析

设 $f(L) = L \cdot \exp(-L)$，求其最大值：

$$f'(L) = \exp(-L) - L \cdot \exp(-L) = \exp(-L)(1 - L) = 0$$

解得 $L = 1$ 时，$f(L)$ 取得最大值 $f(1) = e^{-1} \approx 0.368$。

**理解 DFT 的核心机制**：

权重函数 $w(L) = \exp(-L)$ 确实使得低损失对应高权重。但关键是看**最终训练贡献** $f(L) = L \cdot w(L) = L \cdot \exp(-L)$：

| 场景 | 损失 L | 权重 w | 最终贡献 L×w | 效果 |
|-----|-------|--------|-------------|------|
| 已学会 | 0.1 | 0.90 | 0.09 | 贡献很小 |
| **最佳学习点** | **1.0** | **0.37** | **0.37** | **贡献最大** |
| 较困难 | 2.0 | 0.14 | 0.28 | 贡献中等 |
| 噪声/太难 | 3.0 | 0.05 | 0.15 | 贡献较小 |

这意味着 DFT 自动聚焦于"中等难度"的样本（L≈1），避免在已经学会的内容上浪费资源，同时也不被噪声样本干扰。

## 代码实现

### 核心函数：`per_token_loss_func`

**文件位置**: `swift/trainers/utils.py`

```python
def per_token_loss_func(outputs, labels, enable_dft_loss: bool = False, **kwargs):
    """计算 per-token 损失，支持 DFT 加权

    注意：此函数返回原始 per-token 损失张量，reduction 在 Trainer 中完成。
    loss_scale 的应用也在 Trainer 层级处理，而非此函数内部。
    """
    logits = outputs.logits
    # 上采样到 float32 避免精度问题
    logits = logits.float()
    # 标签向左移动一位（next token prediction）
    labels = torch.roll(labels, shifts=-1, dims=-1).view(-1)

    # Flatten the tokens
    logits = logits.view(-1, logits.shape[-1])
    # Enable model parallelism
    labels = labels.to(logits.device)

    # 计算标准交叉熵损失（per-token，不进行 reduction）
    loss = F.cross_entropy(logits, labels, ignore_index=-100, reduction='none')

    # DFT 动态加权
    if enable_dft_loss:
        with torch.no_grad():
            target_probs = torch.exp(-loss)  # 计算权重
        loss *= target_probs  # 应用权重

    return loss  # 返回 per-token 损失张量，shape: (batch_size * seq_len,)
```

**关键实现细节**：

1. **`torch.no_grad()` 包裹权重计算**：确保 DFT 权重不参与梯度计算，只作为缩放因子
2. **返回原始张量**：函数返回 per-token 损失张量，mean reduction 在 Trainer 中完成
3. **loss_scale 在 Trainer 层处理**：与 Channel Loss 等场景的 loss_scale 加权在 Trainer 中应用

### Sequence Parallel 版本：`per_token_loss_func_sp`

**文件位置**: `swift/trainers/utils.py`

```python
def per_token_loss_func_sp(outputs, labels, enable_dft_loss=False, **kwargs) -> torch.Tensor:
    """Sequence Parallel 版本的 per-token 损失

    关键特性：
    1. 支持 ChunkedCrossEntropyLoss 优化（通过 CELOSS_PARALLEL_SIZE 环境变量控制）
    2. 使用 GatherLoss 进行跨 SP rank 的损失聚合
    3. 支持 padding-free 训练的 position_ids 处理
    """
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs
    device = logits.device

    batch_size = logits.shape[0]
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.flatten().to(device)

    # 可选的分块交叉熵优化
    sploss_parallel_size = int(os.environ.get('CELOSS_PARALLEL_SIZE', '0'))
    if sploss_parallel_size > 0:
        from swift.trainers.sequence_parallel.utils import ChunkedCrossEntropyLoss
        loss = ChunkedCrossEntropyLoss.apply(logits, labels, sploss_parallel_size)
    else:
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits, labels)

    # DFT 动态加权
    if enable_dft_loss:
        with torch.no_grad():
            target_probs = torch.exp(-loss)
        loss *= target_probs

    # Sequence Parallel 损失聚合
    from swift.trainers.sequence_parallel import sequence_parallel
    position_ids = sequence_parallel.real_position_ids
    if position_ids is not None:
        position_ids = sequence_parallel.pad(position_ids, padding_value=-1, position_ids=position_ids)

    from swift.trainers.sequence_parallel.utils import GatherLoss
    loss, labels = GatherLoss.apply(loss.reshape(batch_size, -1), labels.reshape(batch_size, -1), 1, position_ids)

    # 处理 padding 位置
    if position_ids is not None and position_ids.min() == -1:
        _pos_mask = position_ids >= 0
        loss = loss[_pos_mask].contiguous()

    return loss
```

**环境变量**：

| 变量名 | 默认值 | 说明 |
|-------|-------|------|
| `CELOSS_PARALLEL_SIZE` | 0 | Sequence Parallel 模式下分块交叉熵的并行度，0 表示禁用 |

### Trainer 集成

**文件位置**: `swift/trainers/trainers.py` - `Seq2SeqTrainer.compute_loss()`

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    labels = inputs.pop('labels') if 'labels' in inputs else None
    loss_scale = inputs.pop('loss_scale', None)

    outputs = model(**inputs)

    # Step 1: 当启用 DFT、loss_scale 或 channel_loss 时，使用 per_token_loss_func
    if (self.args.enable_dft_loss or loss_scale is not None or
        self.args.enable_channel_loss or self.template.sequence_parallel_size > 1):

        if self.template.sequence_parallel_size > 1:
            # Sequence Parallel 模式
            outputs.loss = per_token_loss_func_sp(
                outputs, labels,
                enable_dft_loss=self.args.enable_dft_loss)
        else:
            # 标准模式
            outputs.loss = per_token_loss_func(
                outputs, labels,
                enable_dft_loss=self.args.enable_dft_loss)

        # Step 2: loss_scale 在 DFT 之后应用
        if loss_scale is not None:
            loss_scale = torch.roll(loss_scale, shifts=-1, dims=-1).view(-1)
            outputs.loss = outputs.loss * loss_scale

        # Step 3: Channel Loss 统计（如果启用）
        if self.args.enable_channel_loss and channels is not None:
            # ... 按 channel 统计损失 ...

        # Step 4: 最终 Reduction
        if num_items_in_batch is None:
            num_items_in_batch = (labels[:, 1:] != -100).sum()
        loss = outputs.loss.sum() / num_items_in_batch

    return (loss, outputs) if return_outputs else loss
```

**关键点**：
- DFT 权重先应用
- loss_scale 后应用
- 最终 reduction 在 Trainer 中完成，而非 loss 函数内部

### Megatron-SWIFT 实现

**文件位置**: `swift/megatron/trainers/trainer.py` - `MegatronTrainer.loss_func()`

```python
def loss_func(self,
              output_tensor: torch.Tensor,
              *,
              labels: torch.Tensor,
              loss_scale: Optional[torch.Tensor] = None,
              channels: Optional[List[str]] = None,
              packed_seq_params=None):
    args = get_args()

    losses = output_tensor.float()
    loss_mask = labels != -100

    # DFT 动态加权
    if args.enable_dft_loss:
        losses = losses * torch.exp(-losses.detach())

    # loss_scale 加权
    if loss_scale is not None:
        losses = losses * loss_scale

    # Channel Loss 统计
    if args.enable_channel_loss and channels is not None:
        # ... 按 channel 统计损失 ...

    # 聚合损失
    loss = torch.cat([torch.sum(losses * loss_mask).view(1), loss_mask.sum().view(1)])

    # Context Parallel 支持
    if args.context_parallel_size > 1 and not self.mcore_013:
        loss = all_reduce(loss, group=mpu.get_context_parallel_group())

    return (lm_loss, local_num_tokens, {'lm loss': reporting_loss})
```

**Megatron 版本特点**：
- 使用 `.detach()` 显式阻断梯度（与标准版的 `torch.no_grad()` 效果相同）
- 支持 Context Parallel (CP) 的损失聚合
- 与 Megatron-LM 的分布式训练完全兼容

**Megatron 高级特性**：

1. **Context Parallel 支持**
```python
if args.context_parallel_size > 1:
    loss = all_reduce(loss, group=mpu.get_context_parallel_group())
```

2. **NaN/Inf 检测**
```python
if args.check_for_nan_in_loss_and_grad:
    rerun_state_machine.validate_result(
        result=loss[0],
        rejection_func=torch.isnan,
        message='found NaN in local forward loss calculation',
        tolerance=0.0,
        fatal=True,
    )
```

3. **Spiky Loss 检测**
```python
if args.check_for_spiky_loss:
    SPIKY_LOSS_FACTOR = 10
    rerun_state_machine.validate_result(
        result=loss[0],
        rejection_func=partial(
            rerun_state_machine.is_unexpectedly_large,
            threshold=SPIKY_LOSS_FACTOR,
            context='loss',
        ),
        message='Spiky loss',
        tolerance=0.0,
        fatal=False,
    )
```

## 参数配置

### TrainArguments 定义

**文件位置**: `swift/trainers/arguments.py`

```python
@dataclass
class TrainArgumentsMixin:
    # ...
    enable_dft_loss: bool = False  # https://arxiv.org/abs/2508.05629
```

**参数说明**：

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| `enable_dft_loss` | bool | False | 是否启用 DFT 动态损失加权 |

### 使用方式

**命令行**：
```bash
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset AI-ModelScope/alpaca-gpt4-data-en \
    --enable_dft_loss true
```

**Python API**：
```python
from swift.llm import sft_main, TrainArguments

sft_main(TrainArguments(
    model='Qwen/Qwen2.5-7B-Instruct',
    dataset=['AI-ModelScope/alpaca-gpt4-data-en'],
    enable_dft_loss=True,
))
```

## 示例脚本

**文件位置**: `examples/train/full/dft.sh`

```bash
# 4*80G GPU 配置
# 实验参考: https://github.com/modelscope/ms-swift/pull/5355
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
    --model Qwen/Qwen2.5-Math-1.5B \
    --train_type full \
    --dataset AI-MO/NuminaMath-CoT#100000 \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --enable_dft_loss true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 32 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.1 \
    --deepspeed zero2 \
    --dataloader_num_workers 4
```

## 与其他特性的兼容性

### 1. 与 Packing (Sample Packing) 兼容

DFT 在 token 级别计算权重，与 packing 模式完全兼容：

```python
# 每个 token 独立计算损失和权重
loss = F.cross_entropy(logits, labels, ignore_index=-100, reduction='none')
if enable_dft_loss:
    loss *= torch.exp(-loss.detach())
```

### 2. 与 Loss Scale (Channel Loss) 兼容

DFT 权重和 loss_scale 可以同时使用，应用顺序为：

```python
# Step 1: DFT 权重先应用
if enable_dft_loss:
    loss *= torch.exp(-loss.detach())

# Step 2: loss_scale 后应用（在 Trainer 层级）
if loss_scale is not None:
    loss *= loss_scale
```

最终权重计算公式：`final_weight = exp(-L_CE) × loss_scale`

### 3. 与 Sequence Parallel 兼容

`per_token_loss_func_sp` 专门处理 Sequence Parallel 场景，确保 DFT 在分布式序列并行训练中正确工作。

### 4. 与 DeepSpeed ZeRO 兼容

DFT 在单个 forward pass 中完成，不需要额外的通信，与所有 ZeRO 阶段兼容。

### 5. 与 LoRA/QLoRA 兼容

DFT 只修改损失计算，不影响模型架构，与所有 PEFT 方法兼容。

### 6. 与 Label Smoothing 的交互

当同时启用 DFT 和 `label_smoothing_factor > 0` 时，存在潜在冲突：

```python
# trainers.py 中的逻辑
if self.label_smoother is None:
    loss = outputs.loss.sum() / num_items_in_batch  # 使用 DFT 计算的损失
else:
    loss = self.label_smoother(outputs, labels, shift_labels=True)  # 可能覆盖 DFT
```

**建议**：使用 DFT 时将 `label_smoothing_factor` 设为 0，或使用自定义 `compute_loss_func`。

## 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                       Training Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐     ┌──────────┐     ┌─────────────────┐          │
│  │  Input   │────>│  Model   │────>│    Logits       │          │
│  │  Tokens  │     │ Forward  │     │  (B, S, V)      │          │
│  └──────────┘     └──────────┘     └────────┬────────┘          │
│                                              │                   │
│                                              v                   │
│                                   ┌─────────────────┐            │
│                                   │ Cross Entropy   │            │
│                                   │ (per-token)     │            │
│                                   │ reduction='none'│            │
│                                   └────────┬────────┘            │
│                                            │                     │
│                                            v                     │
│                   ┌────────────────────────────────────────┐     │
│                   │              DFT Weighting             │     │
│                   │                                        │     │
│                   │  if enable_dft_loss:                   │     │
│                   │      with torch.no_grad():             │     │
│                   │          weight = exp(-loss)           │     │
│                   │      loss = loss * weight              │     │
│                   │                                        │     │
│                   │  ┌──────────────────────────────────┐  │     │
│                   │  │  Loss: 0.1 → Weight: 0.90        │  │     │
│                   │  │        → Contribution: 0.09      │  │     │
│                   │  │  Loss: 1.0 → Weight: 0.37        │  │     │
│                   │  │        → Contribution: 0.37 (MAX)│  │     │
│                   │  │  Loss: 2.0 → Weight: 0.14        │  │     │
│                   │  │        → Contribution: 0.28      │  │     │
│                   │  │  Loss: 3.0 → Weight: 0.05        │  │     │
│                   │  │        → Contribution: 0.15      │  │     │
│                   │  └──────────────────────────────────┘  │     │
│                   └────────────────┬───────────────────────┘     │
│                                    │                             │
│                                    v                             │
│                         ┌─────────────────┐                      │
│                         │   Loss Scale    │  (if channel_loss)   │
│                         │   Weighting     │                      │
│                         └────────┬────────┘                      │
│                                  │                               │
│                                  v                               │
│                         ┌─────────────────┐                      │
│                         │  Sum / N_tokens │                      │
│                         │  (in Trainer)   │                      │
│                         └────────┬────────┘                      │
│                                  │                               │
│                                  v                               │
│                         ┌─────────────────┐                      │
│                         │   Final Loss    │                      │
│                         │   for Backward  │                      │
│                         └─────────────────┘                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 代码调用流程

```
Trainer.training_step()
    │
    └── Seq2SeqTrainer.compute_loss()           # trainers.py
            │
            ├── model(**inputs)                 # Forward pass
            │
            ├── Check: enable_dft_loss or loss_scale or channel_loss?
            │   │
            │   ├── YES + SP: per_token_loss_func_sp()   # utils.py
            │   │       │
            │   │       ├── CrossEntropyLoss(reduction='none')
            │   │       │   or ChunkedCrossEntropyLoss (if CELOSS_PARALLEL_SIZE > 0)
            │   │       ├── exp(-loss) [DFT weight, no_grad]
            │   │       ├── GatherLoss.apply()           # SP aggregation
            │   │       └── return per-token losses
            │   │
            │   └── YES: per_token_loss_func()           # utils.py
            │           │
            │           ├── F.cross_entropy(reduction='none')
            │           ├── exp(-loss) [DFT weight, no_grad]
            │           └── return per-token losses
            │
            ├── Apply loss_scale (if any)       # After DFT
            │
            ├── Channel Loss statistics (if enabled)
            │
            └── loss = outputs.loss.sum() / num_items_in_batch
                                                # Final reduction in Trainer
```

## 性能影响

### 计算开销

DFT 引入的额外计算非常小：
- 一次 `torch.exp()` 操作
- 一次逐元素乘法

相比 forward 和 backward pass，开销可以忽略不计（< 1%）。

### 内存开销

- 额外存储一个与 loss 同形状的权重张量
- 在 `torch.no_grad()` 上下文中计算，不创建额外的计算图

### 梯度影响

**关键**：DFT 权重通过 `torch.no_grad()` 和 `.detach()` 与梯度计算隔离：

```python
with torch.no_grad():
    target_probs = torch.exp(-loss)  # 不追踪梯度
loss *= target_probs                  # 只作为缩放因子
```

这意味着 DFT 权重不会影响梯度的方向，只影响梯度的大小。

## 适用场景

### 推荐使用 DFT

1. **数据集质量参差不齐**：DFT 自动降低噪声样本的影响
2. **长尾分布数据**：避免在罕见模式上过拟合
3. **多任务混合训练**：不同任务难度不同时自动平衡
4. **继续预训练**：模型已经学会大部分知识，DFT 聚焦于新知识
5. **数学推理训练**：如示例脚本中的 NuminaMath-CoT 数据集

### 谨慎使用 DFT

1. **数据集很小**：可能导致欠拟合
2. **所有样本同等重要**：DFT 会降低简单样本权重
3. **需要精确控制损失权重**：使用 loss_scale 更可控

## 总结

ms-swift 的 DFT 实现具有以下特点：

1. **简洁优雅**：核心代码仅 3 行（权重计算 + 应用）
2. **零侵入**：不修改模型架构，只调整损失计算
3. **高度兼容**：与 packing、loss_scale、sequence parallel、Megatron 等特性完美配合
4. **性能友好**：几乎无额外开销
5. **易于使用**：单参数启用 `--enable_dft_loss true`

通过动态调整每个 token 的训练权重，DFT 让模型将更多注意力放在"有价值"的学习信号上，从而提高训练效率和最终效果。
