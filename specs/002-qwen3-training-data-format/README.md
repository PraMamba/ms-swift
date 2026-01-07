---
status: complete
created: '2025-12-30'
tags:
  - qwen3
  - training
  - data-format
  - sft
  - grpo
  - documentation
priority: high
created_at: '2025-12-30T08:59:46.264Z'
updated_at: '2025-12-30T09:02:17.802Z'
completed_at: '2025-12-30T09:02:17.802Z'
completed: '2025-12-30'
transitions:
  - status: complete
    at: '2025-12-30T09:02:17.802Z'
---

# Qwen3 Model SFT and GRPO Training Data Format Requirements

> **Status**: ✅ Complete · **Priority**: High · **Created**: 2025-12-30 · **Tags**: qwen3, training, data-format, sft, grpo, documentation

## Overview

This specification documents the **exact input data format requirements** for training Qwen3 models in ms-swift, based entirely on source code analysis. It covers:

- **Standard message format** for Qwen3 models
- **SFT (Supervised Fine-Tuning)** data requirements
- **GRPO (Group Relative Policy Optimization)** data requirements
- **Data preprocessing and validation** mechanisms
- **Reward functions** for GRPO training
- **Configuration parameters** and practical examples

All information is derived from the ms-swift codebase and includes exact file paths for verification.

---

## ❓ Quick FAQ: Do I Need `<think>...</think><answer>...</answer>` in SFT Data?

### Question
**"Qwen3 SFT 数据格式里的 assistant response content 不需要设置 `<think>...</think><answer>...</answer>` 结构吗?"**

### Short Answer
**NO** - You do NOT need to add `<think>...</think><answer>...</answer>` structure to your SFT training data.

### Detailed Answer

**For SFT Training Data** (What you write):
```jsonl
✅ CORRECT - Plain text works fine:
{"messages": [{"role": "user", "content": "What is 2+2?"},
              {"role": "assistant", "content": "2+2 equals 4"}]}

✅ ALSO CORRECT - With empty think tags (to preserve reasoning ability):
{"messages": [{"role": "user", "content": "What is 2+2?"},
              {"role": "assistant", "content": "<think>\n\n</think>\n\n2+2 equals 4"}]}
```

**What Happens Internally**:
- Qwen3Template automatically adds `<think>\n\n</think>\n\n` prefix if your response doesn't start with `<think>`
- Source: `swift/llm/template/template/utils.py:45-56`
- Your plain response `"2+2 equals 4"` becomes `"<think>\n\n</think>\n\n2+2 equals 4"` during training

**For GRPO Training** (What the model generates):
- **Input data**: Same as SFT - no special structure required
- **Model output**: The `format` reward function encourages the model to OUTPUT `<think>...</think><answer>...</answer>` structure
- **Key distinction**: Input data is flexible, output format is enforced by rewards

**Three Training Strategies**:

| Strategy | Data Format | Use Case | Training Flag |
|----------|-------------|----------|---------------|
| **Option 1** | Plain responses: `"It equals 4"` | General SFT | No special flags needed |
| **Option 2** | Empty think: `"<think>\n\n</think>\n\nIt equals 4"` | Preserve reasoning ability | `--loss_scale ignore_empty_think` |
| **Option 3** | Plain + `/no_think`: `"What is 2+2? /no_think"` | Per-query control | No special flags needed |

**References**:
- See **Section 2.1** for complete details
- See **Section 5.2** for GRPO format reward function
- See **Section 8** for summary

---

## 1. Core Standard Format for Qwen3

### 1.1 Template Configuration

**Source**: `swift/llm/template/template/qwen.py:59-63`

```python
class Qwen3Template(ThinkingTemplate):
    no_think_prefix = '<think>\n\n</think>\n\n'

register_template(QwenTemplateMeta(LLMTemplateType.qwen3,
                                  default_system=None,
                                  template_cls=Qwen3Template))
```

**Key Characteristics**:
- Inherits from `ThinkingTemplate` base class
- **No default system prompt** (`default_system=None`)
- Supports thinking tags: `<think>` and `</think>`
- Default agent template: `'hermes'`

### 1.2 Standard Messages Format

**Source**: `docs/source_en/Customization/Custom-dataset.md:18-21`

The standard format uses **Messages format** (JSONL):

```jsonl
{"messages": [{"role": "system", "content": "<system>"}, {"role": "user", "content": "<query1>"}, {"role": "assistant", "content": "<response1>"}, {"role": "user", "content": "<query2>"}, {"role": "assistant", "content": "<response2>"}]}
```

**Valid Roles** (`swift/llm/dataset/preprocessor/core.py:75`):
- `system` - System instructions
- `user` - User queries
- `assistant` - Model responses
- `tool_call` - Tool invocation requests
- `tool_response` - Tool execution results
- `tool` - Tool definitions

---

## 2. SFT (Supervised Fine-Tuning) Data Format

### 2.1 Critical Question: Do Assistant Responses Need `<think>...</think><answer>...</answer>` Structure?

**Answer: NO, it is NOT required for SFT training data. The structure depends on your training objectives.**

**Three Training Approaches**:

#### Option 1: Standard Format (No Thinking Tags) - Recommended for Most Cases

**Source**: `docs/source_en/BestPractices/Qwen3-Best-Practice.md:79-91`

```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "Tell me tomorrow's weather"}, {"role": "assistant", "content": "Tomorrow's weather will be sunny"}]}
{"messages": [{"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "It equals 2"}]}
```

**What Happens Internally** (`swift/llm/template/template/utils.py:45-56`):

Qwen3Template automatically adds `no_think_prefix = '<think>\n\n</think>\n\n'` to assistant responses that don't start with `<think>` or the prefix itself:

```python
# If assistant content doesn't start with '<think>' or no_think_prefix
# The template prepends: '<think>\n\n</think>\n\n' + content
```

**Result**: Your simple response `"It equals 2"` becomes `"<think>\n\n</think>\n\nIt equals 2"` during training.

#### Option 2: Empty Think Tags + `--loss_scale ignore_empty_think` - Preserve Reasoning Ability

**Source**: `docs/source_en/BestPractices/Qwen3-Best-Practice.md:95-102`

When training with non-reasoning data but wanting to **preserve the model's thinking capability**:

```jsonl
{"messages": [{"role": "user", "content": "Where is the capital of Zhejiang?"}, {"role": "assistant", "content": "<think>\n\n</think>\n\nThe capital of Zhejiang is Hangzhou."}]}
```

**Training Command**:
```bash
swift sft \
    --model Qwen/Qwen3-8B \
    --dataset my_data.jsonl \
    --loss_scale ignore_empty_think \  # KEY: Ignore loss for empty think blocks
    ...
```

**How It Works** (`swift/plugin/loss_scale/loss_scale.py:145-150`):
- The `ignore_empty_think` loss scale uses regex matching: `<think>\\s*</think>\\s*`
- Any match gets `loss_scale = 0.0` (no gradient update)
- This prevents the model from "forgetting" how to do step-by-step reasoning

**Why This Matters**: Without this, training on direct-answer data can degrade the model's ability to perform complex reasoning tasks.

#### Option 3: Use `/no_think` Suffix in User Query

**Source**: `docs/source_en/BestPractices/Qwen3-Best-Practice.md:104-111`

```jsonl
{"messages": [{"role": "user", "content": "Where is the capital of Zhejiang? /no_think"}, {"role": "assistant", "content": "<think>\n\n</think>\n\nThe capital of Zhejiang is Hangzhou."}]}
```

The `/no_think` suffix signals the template to use the `no_think_prefix` behavior for this specific turn.

### 2.2 Standard SFT Format Requirements

**Format Requirements**:
- Each line is a complete JSON object (JSONL format)
- `messages` array contains conversation history
- Multi-turn conversations supported
- System message optional (Qwen3 has no default system)
- **Assistant responses do NOT require `<think>...</think>` structure for SFT**

### 2.2 Alternative Compatible Formats

**Source**: `swift/llm/dataset/preprocessor/core.py:363-364`

The preprocessor **automatically converts** these field names:

| Standard Field | Alternative Names |
|---------------|-------------------|
| `query` | `prompt`, `input`, `instruction`, `question`, `problem` |
| `response` | `answer`, `output`, `targets`, `target`, `answer_key`, `answers`, `solution`, `text`, `completion`, `content` |
| `system` | `system_prompt` |

**Example** - Simple query-response format:
```jsonl
{"query": "What is 1+1?", "response": "It equals 2"}
```

This is automatically converted to messages format internally.

### 2.3 Loss Control (Selective Training)

**Source**: `docs/source_en/Customization/Custom-dataset.md:64-68`

You can control which assistant responses contribute to training loss:

```jsonl
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi, how can I help you?", "loss": false}, {"role": "user", "content": "What is 1+1?"}, {"role": "assistant", "content": "It equals 2", "loss": true}]}
```

- `"loss": false` - Exclude this response from loss calculation
- `"loss": true` or omitted - Include in loss calculation

**Use Cases**:
- Skip training on greeting/acknowledgment responses
- Focus training on domain-specific content
- Exclude synthetic/templated responses

---

## 3. GRPO (Group Relative Policy Optimization) Data Format

### 3.1 Basic GRPO Format

**Source**: `docs/source_en/Customization/Custom-dataset.md:112-119`

**CRITICAL DIFFERENCE**: GRPO data contains messages **WITHOUT the final assistant response**:

```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "Tell me tomorrow's weather"}]}
{"messages": [{"role": "system", "content": "You are a useful and harmless math calculator"}, {"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1?"}]}
{"messages": [{"role": "user", "content": "What is your name?"}]}
```

**Why?** The model generates multiple completions per prompt, and reward functions evaluate them.

### 3.2 Additional Fields for Reward Functions

**Source**: `docs/source_en/Customization/Custom-dataset.md:119` (CRITICAL NOTE)

> GRPO will pass through **all additional field content** to the ORM (reward function), unlike other training methods that delete extra fields by default.

**Example with `solution` field**:
```jsonl
{"messages": [{"role": "user", "content": "What is 2+2?"}], "solution": "4"}
{"messages": [{"role": "user", "content": "Factor x^2-1"}], "solution": "(x-1)(x+1)"}
```

**Source**: `swift/llm/dataset/preprocessor/core.py:314-321`

The `solution` field is specifically retained for GRPO:
```python
# compat GRPO: The solution field will be retained.
if 'solution' in dataset.features:
    dataset = dataset.map(lambda x: {'__#solution': x['solution']}, **map_kwargs)
```

### 3.3 Real-World GRPO Example

**Source**: `swift/llm/dataset/dataset/llm.py:664-677`

XlamFunctionCalling GRPO dataset preprocessor:

```python
class XlamFunctionCallingGRPOPreprocessor(ResponsePreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = row['query']
        answers = row['response']
        if isinstance(answers, str):
            answers = json.loads(answers)
        answer = np.random.choice(answers)
        name = answer['name']
        args = json.dumps(answer['arguments'])
        response = f'Action: {name}\nAction Input: {args}'
        row = {'query': query,
               'response': response,
               'solution': response,  # Ground truth
               'tools': row['tools']}  # Additional field
        return super().preprocess(row)
```

**Custom Fields Passed to Reward Function**:
- `solution` - Expected correct answer
- `tools` - Available tool definitions
- Any other field you add

---

## 4. Data Preprocessing and Validation

### 4.1 Preprocessor Selection

**Source**: `swift/llm/dataset/preprocessor/core.py`

Three core preprocessors auto-detect format:

1. **MessagesPreprocessor** - For `messages` and ShareGPT format
2. **AlpacaPreprocessor** - For alpaca format
3. **ResponsePreprocessor** - For query/response format

**Source**: `swift/llm/dataset/preprocessor/core.py:460-514`

Format detection logic:
```python
def _is_sharegpt_format(message: Dict[str, str]) -> bool:
    # Detects ShareGPT format (from, value, etc.)

if self._is_sharegpt_format(messages[0]):
    # Convert ShareGPT to standard format
```

### 4.2 Role Validation

**Source**: `swift/llm/dataset/preprocessor/core.py:75`

```python
assert role in {'system', 'user', 'tool_call', 'tool_response', 'tool', 'assistant'}, \
    f'message: {message}'
```

Only these 6 roles are valid. Invalid roles cause assertion error.

### 4.3 Data Arguments Configuration

**Source**: `swift/llm/argument/base_args/data_args.py:58-64`

Key validation parameters:

```python
columns: Optional[str] = None  # JSON string for column mapping
strict: bool = True  # If True, raises error; if False, discards bad samples
remove_unused_columns: bool = True  # Default for SFT
# Note: GRPO sets remove_unused_columns=False automatically
```

**Column Mapping Example**:
```python
--columns '{"messages": "conversations", "query": "instruction"}'
```

Maps `conversations` field in your data to `messages`, etc.

---

## 5. Reward Functions for GRPO

### 5.1 Built-in Reward Functions

**Source**: `swift/plugin/orm.py:404-413`

```python
orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
}
```

### 5.2 Format Reward (for Qwen3 thinking format)

**Source**: `swift/plugin/orm.py:289-295`

```python
class Format(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        """Checks if completion has <think>...</think><answer>...</answer> format"""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE)
                   for content in completions]
        return [1.0 if match else 0.0 for match in matches]
```

**Returns**:
- `1.0` if completion matches format
- `0.0` otherwise

**CRITICAL DISTINCTION: SFT vs GRPO Data Format**:

| Aspect | SFT Training Data | GRPO Training/Inference Output |
|--------|-------------------|-------------------------------|
| **Input Data Structure** | ❌ Does NOT require `<think>...</think><answer>...</answer>` | ✅ Model GENERATES `<think>...</think><answer>...</answer>` |
| **Where Structure Matters** | Template auto-adds `<think>\n\n</think>\n\n` if missing | Reward function checks for this structure |
| **Purpose** | Training: what the model learns from | Evaluation: what the model produces |
| **Example** | `"content": "2+2 equals 4"` (plain text is fine) | Model outputs: `"<think>Let me calculate...</think><answer>4</answer>"` |

**Why This Design?**
- **SFT data flexibility**: You don't need to manually add think/answer tags to every training example
- **GRPO reward enforcement**: The `format` reward function trains the model to OUTPUT in the correct structure
- **Automatic handling**: Qwen3Template bridges the gap by adding empty think tags during SFT

### 5.3 Math Accuracy Reward

**Source**: `swift/plugin/orm.py:236-286`

```python
class MathAccuracy(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Args:
            completions: Generated responses
            solution: Ground truth answers (from dataset field)
        """
        rewards = []
        for content, sol in zip(completions, solution):
            # Extract answer from <answer>...</answer> tags
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            content_to_parse = content_match.group(1).strip() if content_match else content
            # Use math_verify package for symbolic verification
            # Returns 1.0 for correct, 0.0 for incorrect
```

**Requires**: `solution` field in dataset

### 5.4 Custom Reward Function

**Source**: `examples/train/grpo/plugin/plugin.py:40-92`

```python
from swift.plugin import ORM, orms

class CountdownORM(ORM):
    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Args:
            completions: Generated outputs (automatically passed)
            target: Expected answers (from dataset 'target' field)
            nums: Available numbers (from dataset 'nums' field)
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            # Parse completion
            match = re.search(r'<answer>(.*?)<\/answer>', completion)
            if not match:
                rewards.append(0.0)
                continue

            # Custom validation logic
            predicted = match.group(1).strip()
            reward = 1.0 if predicted == gt else 0.0
            rewards.append(reward)

        return rewards

# Register custom reward
orms['external_countdown'] = CountdownORM
```

**Usage**:
```bash
--reward_funcs external_countdown
```

**Key Points**:
- First arg is always `completions` (list of generated texts)
- Other args are **keyword arguments** from dataset fields
- Must return `List[float]` with same length as completions
- Register with `orms[name] = YourORMClass`

---

## 6. GRPO Training Configuration

### 6.1 Key GRPO Parameters

**Source**: `swift/llm/argument/rlhf_args.py:105-147`

```python
@dataclass
class GRPOArguments(GRPOArgumentsMixin):
    num_generations: int = 8          # G in GRPO paper
    reward_funcs: List[str] = []      # e.g., ['accuracy', 'format']
    reward_weights: List[float] = None  # Weights for each reward
    log_completions: bool = False     # Log generated completions
    num_iterations: int = 1           # Training iterations
    truncation_strategy: Literal['delete', 'left', 'right', 'split', None] = None
```

### 6.2 GRPO-Specific Initialization

**Source**: `swift/llm/argument/rlhf_args.py:307-336`

```python
def _init_grpo(self):
    if self.rlhf_type != 'grpo':
        return

    # CRITICAL: Extra fields passed to ORM
    self.remove_unused_columns = False

    if self.truncation_strategy is None:
        self.truncation_strategy = 'left'

    if self.beta is None:
        self.beta = 0.04  # Default KL divergence weight
```

**Important**: `remove_unused_columns=False` ensures custom dataset fields (like `solution`, `tools`) are preserved and passed to reward functions.

---

## 7. Practical Examples

### 7.1 Qwen3 SFT Training

```bash
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset my_dataset.jsonl \
    --num_train_epochs 3 \
    --learning_rate 2e-5
```

**Data format** (`my_dataset.jsonl`):
```jsonl
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "2+2 equals 4."}]}
{"messages": [{"role": "system", "content": "You are a math tutor"}, {"role": "user", "content": "Explain calculus"}, {"role": "assistant", "content": "Calculus is..."}]}
```

### 7.2 Qwen3 GRPO Training (Basic)

**Source**: `tests/train/test_grpo.py:20-35`

```python
from swift.llm import rlhf_main, RLHFArguments

result = rlhf_main(
    RLHFArguments(
        rlhf_type='grpo',
        model='Qwen/Qwen2.5-1.5B-Instruct',
        train_type='full',
        dataset=['AI-MO/NuminaMath-TIR#100'],
        split_dataset_ratio=0.1,
        system='You are a helpful math assistant',
        reward_funcs=['accuracy', 'format'],
        max_completion_length=4096,
        num_generations=2,
    ))
```

**Data format** (NuminaMath-TIR style):
```jsonl
{"messages": [{"role": "user", "content": "What is 15*23?"}], "solution": "345"}
{"messages": [{"role": "user", "content": "Factor x^2-4"}], "solution": "(x-2)(x+2)"}
```

### 7.3 Qwen3 GRPO Training (Advanced with vLLM)

**Source**: `examples/train/grpo/external/grpo_7b.sh`

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --reward_funcs accuracy format \
    --reward_weights 0.7 0.3 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --dataset AI-MO/NuminaMath-TIR#1000 \
    --max_completion_length 2048 \
    --num_generations 8 \
    --beta 0.04 \
    --log_completions true
```

**Key Points**:
- Multiple reward functions with custom weights
- vLLM for faster generation
- `num_generations=8` generates 8 completions per prompt
- `beta=0.04` controls KL divergence penalty

### 7.4 Custom GRPO Dataset Example

```jsonl
{"messages": [{"role": "user", "content": "Solve: 2x + 3 = 7"}], "solution": "x = 2", "difficulty": "easy"}
{"messages": [{"role": "user", "content": "Prove: sqrt(2) is irrational"}], "solution": "Use proof by contradiction...", "difficulty": "hard"}
```

**Custom Reward Function**:
```python
class MathGradedORM(ORM):
    def __call__(self, completions, solution, difficulty, **kwargs):
        rewards = []
        for comp, sol, diff in zip(completions, solution, difficulty):
            # Check correctness
            correct = is_equivalent(comp, sol)

            # Weight by difficulty
            if correct:
                reward = 1.5 if diff == 'hard' else 1.0
            else:
                reward = 0.0

            rewards.append(reward)
        return rewards

orms['math_graded'] = MathGradedORM
```

---

## 8. Key Findings Summary

### Critical Answer: `<think>...</think><answer>...</answer>` Structure in SFT Data

**Question**: Do Qwen3 SFT assistant responses need `<think>...</think><answer>...</answer>` structure?

**Answer**: **NO - Not required in input data. The template handles it automatically.**

**Detailed Explanation**:

1. **What You Write in SFT Data** (Input):
   ```jsonl
   {"messages": [{"role": "user", "content": "What is 2+2?"},
                 {"role": "assistant", "content": "It equals 4"}]}
   ```
   ✅ Plain text is perfectly fine

2. **What Qwen3Template Does** (Automatic Processing):
   - Adds `<think>\n\n</think>\n\n` prefix automatically
   - Your data becomes: `"<think>\n\n</think>\n\nIt equals 4"`
   - Source: `swift/llm/template/template/utils.py:45-56`

3. **Three Training Strategies**:
   - **Option A**: Plain responses → Template adds empty think prefix automatically
   - **Option B**: Add `<think>\n\n</think>\n\n` yourself + use `--loss_scale ignore_empty_think` to preserve reasoning ability
   - **Option C**: Use `/no_think` suffix in user queries

4. **GRPO vs SFT Distinction**:
   - **SFT data (input)**: Does NOT need think/answer structure
   - **GRPO output (generated)**: Model learns to OUTPUT think/answer structure via reward function
   - The `format` reward function (Section 5.2) enforces structure on **model outputs**, not training inputs

### SFT Requirements
1. ✅ Standard messages format with roles: system, user, assistant
2. ✅ Multi-turn conversations supported
3. ✅ Optional loss control per assistant message
4. ✅ Auto-conversion from alternative field names
5. ✅ System message optional for Qwen3
6. ✅ **Assistant responses do NOT require `<think>...</think>` tags - template adds them**

### GRPO Requirements
1. ✅ Messages format **WITHOUT final assistant response**
2. ✅ Additional fields (e.g., `solution`) preserved and passed to reward functions
3. ✅ Reward functions receive `completions` + custom dataset fields
4. ✅ Multiple reward functions can be combined with weights
5. ✅ `remove_unused_columns=False` automatically set

### Critical Configuration
- **SFT**: `remove_unused_columns=True` (default)
- **GRPO**: `remove_unused_columns=False` (automatic)
- **Validation**: `strict=True` raises errors, `strict=False` discards bad samples
- **GRPO Beta**: Default `0.04` for KL divergence

### Common Pitfalls

**SFT Data Pitfalls**:
1. ❌ **MISCONCEPTION**: "I must add `<think>...</think><answer>...</answer>` to every assistant response"
   - ✅ **REALITY**: Template adds `<think>\n\n</think>\n\n` automatically. Plain text responses work fine.
2. ❌ Training with plain responses without `--loss_scale ignore_empty_think` when you want to preserve reasoning ability
   - ✅ **FIX**: Use `--loss_scale ignore_empty_think` + manually add empty think tags to preserve the model's thinking capability
3. ❌ Confusing SFT data requirements with GRPO output requirements
   - ✅ **REMEMBER**: SFT data is flexible (template handles it), GRPO output is strict (reward function enforces it)

**GRPO Data Pitfalls**:
1. ❌ Including final assistant response in GRPO data
2. ❌ Forgetting to add `solution` field for accuracy rewards
3. ❌ Using invalid roles (only 6 valid: system/user/assistant/tool_call/tool_response/tool)
4. ❌ Not registering custom reward functions with `orms[name] = Class`

---

## Source Code References

All findings are derived from these source files:

| File Path | Key Content |
|-----------|-------------|
| `swift/llm/template/template/qwen.py:59-63` | Qwen3 template configuration |
| `docs/source_en/Customization/Custom-dataset.md` | Data format documentation |
| `swift/llm/dataset/preprocessor/core.py` | Preprocessing and validation logic |
| `swift/llm/argument/base_args/data_args.py:58-64` | Data arguments configuration |
| `swift/llm/argument/rlhf_args.py:105-147` | GRPO parameters |
| `swift/plugin/orm.py` | Built-in reward functions |
| `examples/train/grpo/plugin/plugin.py` | Custom reward function example |
| `tests/train/test_grpo.py:20-35` | GRPO training example |
| `swift/llm/dataset/dataset/llm.py:664-677` | Real-world GRPO preprocessor |
