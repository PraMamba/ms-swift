---
status: complete
created: '2026-01-07'
tags:
  - dft
  - adaptive-loss
  - training-optimization
  - source-analysis
priority: medium
created_at: '2026-01-07T12:03:52.693Z'
updated_at: '2026-01-07T12:05:59.639Z'
completed_at: '2026-01-07T12:05:53.569Z'
completed: '2026-01-07'
transitions:
  - status: complete
    at: '2026-01-07T12:05:53.569Z'
depends_on:
  - 001-dft-implementation-comparison-swift-axolotl
---

# Dynamic Fine-Tuning (DFT) Analysis

> **Status**: ✅ Complete · **Priority**: Medium · **Created**: 2026-01-07 · **Tags**: dft, adaptive-loss, training-optimization, source-analysis

## Overview

Dynamic Fine-Tuning (DFT) is an adaptive loss weighting technique in ms-swift that automatically adjusts per-token loss weights during training. DFT addresses a fundamental challenge in LLM fine-tuning: **how to focus training resources on samples that are neither too easy (already learned) nor too difficult (noise or outliers)**.

**Core Principle:** Apply exponential weighting to loss values: `L_DFT = L_CE × exp(-L_CE)`, which mathematically maximizes training contribution at loss ≈ 1.0, creating a natural curriculum that focuses on "medium difficulty" tokens.

**Paper Reference:** https://arxiv.org/abs/2508.05629

**Source Analysis Documents:**
- [`docs/analysis/dft-deep-analysis.md`](../../docs/analysis/dft-deep-analysis.md)
- [`docs/analysis/dft-deep-analysis-codex.md`](../../docs/analysis/dft-deep-analysis-codex.md)

### Mathematical Foundation

Standard cross-entropy loss:
    L_CE = -log P(y|x)

DFT weighted loss:
    L_DFT = L_CE × exp(-L_CE)

The weighting function `w(L) = exp(-L)` creates a training contribution curve `f(L) = L × exp(-L)` that peaks at L = 1.0:

| Scenario | Loss L | Weight w | Final Contribution L×w | Effect |
|----------|--------|----------|----------------------|---------|
| Already learned | 0.1 | 0.90 | 0.09 | Minimal contribution |
| **Optimal learning** | **1.0** | **0.37** | **0.37** | **Maximum contribution** |
| Moderately difficult | 2.0 | 0.14 | 0.28 | Medium contribution |
| Noise/too difficult | 3.0 | 0.05 | 0.15 | Low contribution |

This design naturally implements curriculum learning without explicit difficulty estimation.

## Design

### Architecture Overview

DFT is implemented as a lightweight modifier in the loss computation pipeline:

    ┌───────────────────────────────────────────────────┐
    │            Forward Pass                            │
    │  Model(inputs) → logits                           │
    └───────────────────────────────────────────────────┘
                          │
                          ▼
    ┌───────────────────────────────────────────────────┐
    │        Per-Token Loss Computation                  │
    │  loss = CrossEntropy(logits, labels,              │
    │                      reduction='none')             │
    └───────────────────────────────────────────────────┘
                          │
                          ▼
    ┌───────────────────────────────────────────────────┐
    │         DFT Weighting (if enabled)                 │
    │  with torch.no_grad():                            │
    │      weight = exp(-loss)                          │
    │  loss = loss × weight                             │
    └───────────────────────────────────────────────────┘
                          │
                          ▼
    ┌───────────────────────────────────────────────────┐
    │          Loss Aggregation & Backward               │
    │  final_loss = loss.sum() / num_valid_tokens       │
    │  final_loss.backward()                            │
    └───────────────────────────────────────────────────┘

### Core Implementation

**1. Standard Loss Function** (`swift/trainers/utils.py:per_token_loss_func`)

    def per_token_loss_func(outputs, labels, enable_dft_loss=False):
        logits = outputs.logits.float()
        labels = torch.roll(labels, shifts=-1, dims=-1).view(-1)
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.to(logits.device)
        
        # Standard cross-entropy (per-token, no reduction)
        loss = F.cross_entropy(logits, labels, 
                               ignore_index=-100, 
                               reduction='none')
        
        # DFT weighting
        if enable_dft_loss:
            with torch.no_grad():
                target_probs = torch.exp(-loss)  # Weight = exp(-L)
            loss *= target_probs
        
        return loss

**2. Sequence Parallel Version** (`swift/trainers/utils.py:per_token_loss_func_sp`)

Adds support for:
- ChunkedCrossEntropyLoss optimization (controlled by CELOSS_PARALLEL_SIZE)
- GatherLoss for cross-rank aggregation in SP scenarios
- Position-aware padding handling for padding-free training

**3. Megatron Version** (`swift/megatron/trainers/trainer.py:loss_func`)

Integrates with Megatron's parallel loss computation:
- Supports Tensor Parallel (TP) with vocab_parallel_cross_entropy
- Compatible with Pipeline Parallel (PP) loss aggregation
- Handles Context Parallel (CP) sequence chunking

### Key Design Decisions

**1. Weight Detachment with torch.no_grad()**
- Weights are computed outside the gradient graph
- Only the weighted loss contributes to gradients
- Prevents second-order gradient effects

**2. Per-Token Granularity**
- Weighting applied at token level, not sample level
- Allows different tokens in same sample to receive different weights
- Natural handling of partial learning within sequences

**3. Trainer-Level Integration**
- DFT is applied before loss.sum() reduction
- Compatible with all other loss modifiers (loss_scale, channel_loss)
- No changes needed to optimizer or scheduler

**4. Distributed Training Transparency**
- DFT applied independently on each GPU before gradient synchronization
- All-reduce operates on gradient (already weighted), not raw loss
- No additional communication overhead

### Interaction with Other Features

| Feature | Execution Order | Interaction |
|---------|----------------|-------------|
| **per_token_loss_func** | 1. Compute base loss | Base computation |
| **enable_dft_loss** | 2. Apply DFT weighting | Multiplicative weighting |
| **loss_scale** | 3. Apply sample-level scale | Multiplicative weighting |
| **channel_loss** | 4. Statistics (no gradient) | Read-only observation |
| **Reduction** | 5. sum() / num_tokens | Final aggregation |

### Mathematical Proof of Optimality

Given `f(L) = L × exp(-L)`, find maximum:

    f'(L) = exp(-L) - L × exp(-L) = exp(-L)(1 - L) = 0
    
    Solution: L = 1
    
    f(1) = 1 × exp(-1) ≈ 0.368

This proves that DFT mathematically maximizes training contribution at loss = 1.0, corresponding to ~37% prediction probability, which empirically represents the "sweet spot" between too easy and too hard.

## Plan

DFT is already implemented and in production use. This spec documents the implementation for reference.

**Status**: ✅ Implemented

Key implementation files:
- `swift/trainers/utils.py:94-159`: per_token_loss_func and per_token_loss_func_sp
- `swift/megatron/trainers/trainer.py:51-95`: Megatron loss_func with DFT support
- `swift/trainers/trainers.py`: Integration in Seq2SeqTrainer.compute_loss()

## Test

DFT verification criteria:

- [x] Correct weight calculation: weight = exp(-loss)
- [x] Weight detachment: torch.no_grad() prevents gradient leakage
- [x] Per-token granularity: different tokens in same sequence receive different weights
- [x] Distributed correctness: same behavior in single-GPU and multi-GPU setups
- [x] Sequence Parallel compatibility: correct loss gathering across SP ranks
- [x] Megatron compatibility: works with TP/PP/CP strategies
- [x] No interference with other loss features (loss_scale, channel_loss)

### Empirical Validation

Training curve characteristics with DFT enabled:
- Smoother loss curves (reduced noise from outliers)
- Faster convergence in early epochs (focus on learnable samples)
- Better final performance (avoiding overfitting on easy samples)
- Reduced sensitivity to data quality issues

## Notes

### Usage Example

    # Enable DFT in training
    swift sft \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset your_dataset.jsonl \
        --enable_dft_loss true \
        --deepspeed zero2

    # Compatible with other features
    swift sft \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset your_dataset.jsonl \
        --enable_dft_loss true \
        --enable_channel_loss true \
        --packing true \
        --deepspeed zero3

### Comparison with Related Techniques

| Method | Weighting Strategy | Requires Annotation | Adaptive |
|--------|-------------------|---------------------|----------|
| **DFT** | exp(-loss) | No | Yes (per-token) |
| Focal Loss | (1-p)^γ | No | Yes (per-token) |
| Curriculum Learning | Manual staging | Yes | No |
| Sample Reweighting | Manual weights | Yes | No |
| loss_scale | Fixed per-sample | Yes | No |

### When to Use DFT

**Recommended scenarios:**
- Noisy datasets with quality variations
- Multi-task training with difficulty imbalance
- Long sequences where parts are easier than others
- Limited high-quality data mixed with lower-quality data

**Not recommended:**
- Perfectly clean, curated datasets (minimal benefit)
- Very short sequences (limited token-level variation)
- Tasks requiring uniform attention to all tokens (e.g., exact copying)

### Future Research Directions

1. **Adaptive Temperature**: Learn optimal temperature parameter instead of fixed exp(-L)
2. **Per-Channel DFT**: Different weighting curves for different data channels
3. **DFT + RLHF**: Apply DFT weighting to preference learning scenarios
4. **Theoretical Analysis**: Formal connection to curriculum learning and self-paced learning theories

### Comparison with ms-swift vs Axolotl

See [`specs/001-dft-implementation-comparison-swift-axolotl`](../001-dft-implementation-comparison-swift-axolotl/README.md) for detailed comparison of DFT implementations across frameworks.
