---
status: complete
created: '2025-12-31'
tags:
  - qwen3
  - template
  - configuration
  - model-comparison
priority: high
---

# Qwen3 Model Template Configuration Differences

> **Status**: ✅ Complete · **Priority**: High · **Created**: 2025-12-31 · **Tags**: qwen3, template, configuration, model-comparison

## Overview

This specification documents the **exact template configuration differences** between three Qwen3 2507 models when performing SFT training in ms-swift. All information is derived from source code analysis.

**Models Analyzed**:
1. **Qwen/Qwen3-4B-Instruct-2507** (Non-thinking Instruct model)
2. **Qwen/Qwen3-4B-Thinking-2507** (Thinking model)
3. **Qwen/Qwen3-30B-A3B-Thinking-2507** (MoE Thinking model)

---

## Quick Answer

### Are There Configuration Differences?

**YES** - The three models use **different template types** with distinct behaviors:

| Model | Model Type | Template Type | Template Class | Key Difference |
|-------|-----------|---------------|----------------|----------------|
| **Qwen3-4B-Instruct-2507** | `qwen3_nothinking` | `qwen3_nothinking` | `Template` (base) | No thinking prefix handling |
| **Qwen3-4B-Thinking-2507** | `qwen3_thinking` | `qwen3_thinking` | `ThinkingTemplate` | Adds `<think>\n` response prefix |
| **Qwen3-30B-A3B-Thinking-2507** | `qwen3_moe_thinking` | `qwen3_thinking` | `ThinkingTemplate` | Same as 4B-Thinking |

**Critical**: The 4B and 30B Thinking models use **identical template configuration** (`qwen3_thinking`) despite having different model types.

---

## Detailed Configuration Analysis

### 1. Qwen/Qwen3-4B-Instruct-2507

**Source References**:
- Model registration: `docs/source/Instruction/Supported-models-and-datasets.md:219`
- Template registration: `swift/llm/template/template/qwen.py:96`

#### Configuration

```python
# Model Type
model_type = 'qwen3_nothinking'
template_type = 'qwen3_nothinking'

# Template Registration (swift/llm/template/template/qwen.py:96)
register_template(QwenTemplateMeta(LLMTemplateType.qwen3_nothinking, default_system=None))
```

#### Template Behavior

**Template Class**: `Template` (base class via `QwenTemplateMeta` default)

**Key Characteristics**:
- ❌ Does NOT use `ThinkingTemplate`
- ❌ Does NOT add any thinking-related prefixes
- ❌ Does NOT have `no_think_prefix` processing
- ✅ Standard ChatML format only

**Training Data Handling**:
```jsonl
Input:  {"role": "assistant", "content": "The answer is 42"}
Output: {"role": "assistant", "content": "The answer is 42"}
```
No modifications to assistant responses.

**Use Case**: Standard instruction-following model without reasoning capability.

---

### 2. Qwen/Qwen3-4B-Thinking-2507

**Source References**:
- Model registration: `docs/source/Instruction/Supported-models-and-datasets.md:212`
- Template registration: `swift/llm/template/template/qwen.py:91-94`

#### Configuration

```python
# Model Type
model_type = 'qwen3_thinking'
template_type = 'qwen3_thinking'

# Template Registration (swift/llm/template/template/qwen.py:91-94)
register_template(
    QwenTemplateMeta(
        LLMTemplateType.qwen3_thinking,
        default_system=None,
        response_prefix='<think>\n',  # KEY DIFFERENCE
        template_cls=ThinkingTemplate))
```

#### Template Behavior

**Template Class**: `ThinkingTemplate` (`swift/llm/template/template/utils.py:35-72`)

**Key Characteristics**:
- ✅ Uses `ThinkingTemplate` base class
- ✅ `response_prefix = '<think>\n'`
- ✅ `no_think_prefix = ''` (empty by default)
- ✅ `with_answer = False`
- ✅ History thinking block removal during inference

**Training Data Handling**:
```jsonl
Input:  {"role": "assistant", "content": "<think>\nLet me calculate...\n</think>\nThe answer is 42"}
Output: Processed with thinking template logic (response_prefix applied)
```

**Inference Behavior** (`swift/llm/template/template/utils.py:58-71`):
```python
# During inference or with loss_scale='last_round':
# - Previous round's <think>...</think> content is deleted
# - Only keeps content after </think>
if not self.is_training or self.loss_scale.name in {'last_round', 'last_round_with_ignore_empty_think'}:
    for i, message in enumerate(messages):
        if message['role'] == 'assistant' and i != len(messages) - 1:
            message['content'] = self.history_think_prefix + message['content'].split('</think>')[-1].strip()
```

**Use Case**: Reasoning model that generates step-by-step thinking process.

---

### 3. Qwen/Qwen3-30B-A3B-Thinking-2507

**Source References**:
- Model registration: `docs/source/Instruction/Supported-models-and-datasets.md:234`
- Template registration: `swift/llm/template/template/qwen.py:91-94` (same as 4B-Thinking)

#### Configuration

```python
# Model Type
model_type = 'qwen3_moe_thinking'
template_type = 'qwen3_thinking'  # SAME AS 4B-THINKING

# Template Registration (IDENTICAL to 4B-Thinking)
register_template(
    QwenTemplateMeta(
        LLMTemplateType.qwen3_thinking,
        default_system=None,
        response_prefix='<think>\n',
        template_cls=ThinkingTemplate))
```

#### Template Behavior

**IDENTICAL to Qwen3-4B-Thinking-2507**

**Template Class**: `ThinkingTemplate`

**Key Characteristics**:
- ✅ Uses exact same `qwen3_thinking` template configuration
- ✅ Same `response_prefix = '<think>\n'`
- ✅ Same thinking block processing logic
- 📊 Only difference: `model_type` is `qwen3_moe_thinking` (for MoE architecture identification)

**Use Case**: Large-scale MoE reasoning model with identical template behavior to 4B-Thinking.

---

## Critical Comparison Table

### Template Configuration Matrix

| Feature | qwen3_nothinking | qwen3_thinking (4B & 30B-MoE) |
|---------|------------------|-------------------------------|
| **Template Class** | `Template` (base) | `ThinkingTemplate` |
| **response_prefix** | None | `'<think>\n'` |
| **no_think_prefix** | N/A | `''` (empty) |
| **with_answer** | N/A | `False` |
| **History Thinking Removal** | ❌ No | ✅ Yes (during inference) |
| **Auto-add Think Tags** | ❌ No | ❌ No (empty no_think_prefix) |
| **Thinking Block Processing** | ❌ No | ✅ Yes |

### Hybrid Model Comparison: qwen3 vs qwen3_thinking

There's a **third** template type worth noting:

**`qwen3` (Hybrid Thinking Model)** (`swift/llm/template/template/qwen.py:59-63`):
```python
class Qwen3Template(ThinkingTemplate):
    no_think_prefix = '<think>\n\n</think>\n\n'  # Auto-adds empty thinking block

register_template(QwenTemplateMeta(LLMTemplateType.qwen3,
                                   default_system=None,
                                   template_cls=Qwen3Template))
```

**Key Difference from qwen3_thinking**:
- `qwen3`: Has `no_think_prefix = '<think>\n\n</think>\n\n'` → Auto-adds empty thinking block
- `qwen3_thinking`: Has `no_think_prefix = ''` → Does NOT auto-add anything

---

## SFT Training Data Format Implications

### For Qwen3-4B-Instruct-2507 (`qwen3_nothinking`)

**Data Format**: Standard instruction format

```jsonl
{"messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
]}
```

**No special handling** - responses stay as-is.

---

### For Qwen3-4B-Thinking-2507 & Qwen3-30B-A3B-Thinking-2507 (`qwen3_thinking`)

**Data Format**: Must include explicit `<think>...</think>` structure

```jsonl
{"messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "<think>\nLet me add these numbers: 2 + 2 = 4\n</think>\n2+2 equals 4."}
]}
```

**Why?**
- `no_think_prefix = ''` means NO automatic empty think tag addition
- Template expects thinking content to already be in `<think>...</think>` format
- `response_prefix = '<think>\n'` only affects generation during inference

**If you provide plain text**:
```jsonl
{"messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}  # Missing <think> tags
]}
```

**Result**: Model will learn to generate direct answers WITHOUT thinking process!

---

## Training Command Differences

### Qwen3-4B-Instruct-2507

```bash
swift sft \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --dataset your_dataset.jsonl \
    # Standard training, no special flags needed
```

---

### Qwen3-4B-Thinking-2507

```bash
swift sft \
    --model Qwen/Qwen3-4B-Thinking-2507 \
    --dataset your_thinking_dataset.jsonl \
    # Data MUST have <think>...</think> structure
    # Optional: Use --loss_scale ignore_empty_think if mixing thinking/non-thinking data
```

---

### Qwen3-30B-A3B-Thinking-2507

```bash
# IDENTICAL to 4B-Thinking configuration
swift sft \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --dataset your_thinking_dataset.jsonl \
    # Same requirements as 4B-Thinking
```

---

## Common Pitfalls

### ❌ Pitfall 1: Using Qwen3-4B-Thinking with Plain Data

**Wrong**:
```jsonl
{"messages": [
    {"role": "user", "content": "Calculate 5*3"},
    {"role": "assistant", "content": "15"}  # No <think> tags
]}
```

**Problem**: Model learns to skip thinking process.

**Correct**:
```jsonl
{"messages": [
    {"role": "user", "content": "Calculate 5*3"},
    {"role": "assistant", "content": "<think>\n5 * 3 = 15\n</think>\n15"}
]}
```

---

### ❌ Pitfall 2: Confusing qwen3 with qwen3_thinking

- **`qwen3`** (hybrid): Auto-adds empty `<think>\n\n</think>\n\n` prefix
- **`qwen3_thinking`** (pure): Does NOT auto-add anything

**If you trained on qwen3 (hybrid model)**:
```python
# Qwen3Template has: no_think_prefix = '<think>\n\n</think>\n\n'
# Plain data becomes: '<think>\n\n</think>\n\nYour answer'
```

**If you train on qwen3_thinking**:
```python
# ThinkingTemplate has: no_think_prefix = ''
# Plain data stays: 'Your answer' (NO auto-prefix)
```

---

### ❌ Pitfall 3: Assuming 30B-MoE Has Different Template

**Wrong Assumption**: "30B-MoE is bigger, so it must have special template settings"

**Reality**: `qwen3_moe_thinking` and `qwen3_thinking` use **IDENTICAL** template configuration.

**The only difference**: Model type name (for architecture identification), not template behavior.

---

## Source Code References

All findings are based on exact source code locations:

### Template Registrations

```python
# File: swift/llm/template/template/qwen.py

# Line 63: qwen3 (hybrid thinking model)
register_template(QwenTemplateMeta(LLMTemplateType.qwen3,
                                   default_system=None,
                                   template_cls=Qwen3Template))

# Lines 91-94: qwen3_thinking (pure thinking model)
register_template(
    QwenTemplateMeta(
        LLMTemplateType.qwen3_thinking,
        default_system=None,
        response_prefix='<think>\n',
        template_cls=ThinkingTemplate))

# Line 96: qwen3_nothinking (standard instruct model)
register_template(QwenTemplateMeta(LLMTemplateType.qwen3_nothinking, default_system=None))
```

### ThinkingTemplate Implementation

```python
# File: swift/llm/template/template/utils.py:35-72

class ThinkingTemplate(Template):
    with_answer = False
    no_think_prefix = ''  # for hybrid thinking model
    history_think_prefix = ''
    add_no_think_prefix_after_tool = True

    def _swift_prepare_inputs(self, inputs):
        super()._swift_prepare_inputs(inputs)
        messages = inputs.messages

        # Auto-add no_think_prefix if non-empty (Line 45-56)
        if self.no_think_prefix and self.use_chat_template:
            pre_role = ''
            for message in messages:
                if message['role'] == 'assistant' and isinstance(message['content'], str):
                    if pre_role == 'tool' and not self.add_no_think_prefix_after_tool:
                        pass
                    elif not message['content'].startswith(('<think>', self.no_think_prefix)):
                        message['content'] = self.no_think_prefix + message['content']
                pre_role = message['role']

        # Remove thinking blocks in history during inference (Line 58-71)
        if not self.is_training or self.loss_scale.name in {'last_round', 'last_round_with_ignore_empty_think'}:
            for i, message in enumerate(messages):
                if message['role'] == 'assistant' and isinstance(message['content'], str) and i != len(messages) - 1:
                    if self.with_answer:
                        message['content'] = message['content'].split('<answer>')[-1].rstrip()
                        if message['content'].endswith('</answer>'):
                            message['content'] = message['content'][:-len('</answer>')]
                        message['content'] = message['content'].strip()
                    else:
                        message['content'] = self.history_think_prefix + message['content'].split(
                            '</think>')[-1].strip()
```

### Model Documentation

```markdown
# File: docs/source/Instruction/Supported-models-and-datasets.md

Line 212: |[Qwen/Qwen3-4B-Thinking-2507]|qwen3_thinking|qwen3_thinking|
Line 219: |[Qwen/Qwen3-4B-Instruct-2507]|qwen3_nothinking|qwen3_nothinking|
Line 234: |[Qwen/Qwen3-30B-A3B-Thinking-2507]|qwen3_moe_thinking|qwen3_thinking|
```

---

## Summary

### Three Template Types

| Template Type | Models Using It | Auto-add Think Prefix | Use Case |
|---------------|-----------------|----------------------|----------|
| `qwen3_nothinking` | 4B-Instruct, 30B-Instruct, 235B-Instruct | ❌ No | Standard instruction following |
| `qwen3_thinking` | 4B-Thinking, 30B-MoE-Thinking, 235B-Thinking | ❌ No (empty no_think_prefix) | Pure reasoning models |
| `qwen3` | Qwen3-8B, etc. | ✅ Yes (`<think>\n\n</think>\n\n`) | Hybrid thinking models |

### Key Takeaways

1. **Qwen3-4B-Instruct-2507**: Uses base `Template` class, no thinking support
2. **Qwen3-4B-Thinking-2507**: Uses `ThinkingTemplate`, expects explicit `<think>` tags in data
3. **Qwen3-30B-A3B-Thinking-2507**: **IDENTICAL** template config to 4B-Thinking
4. **Training data must match template expectations**: Thinking models need `<think>...</think>` structure
5. **qwen3_thinking does NOT auto-add empty think tags**: Unlike `qwen3` (hybrid), you must provide explicit thinking content

---

## Appendix: Full Template Hierarchy

```
Template (base class)
├── Qwen3Template (hybrid thinking)
│   └── no_think_prefix = '<think>\n\n</think>\n\n'
│       └── Used by: qwen3 (LLMTemplateType.qwen3)
│
├── ThinkingTemplate
│   ├── no_think_prefix = '' (empty)
│   ├── response_prefix = '' (override in registration)
│   └── Used by:
│       ├── qwen3_thinking (response_prefix='<think>\n')
│       ├── qwq (response_prefix='<think>\n')
│       └── deepseek-r1 variants
│
└── Template (default)
    └── Used by:
        ├── qwen3_nothinking
        ├── qwen3_coder
        └── Other non-thinking models
```
