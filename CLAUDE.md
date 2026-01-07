# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**ms-swift** (ModelScope SWIFT - Scalable lightWeight Infrastructure for Fine-Tuning) is a comprehensive framework for training, fine-tuning, and deploying large language models (LLMs) and multimodal models. It supports 600+ text models and 300+ multimodal models with capabilities spanning the entire ML pipeline from pre-training through deployment.

## Common Commands

### Installation & Setup

```bash
# Install from source (development mode)
pip install -e .

# Install all optional dependencies
bash requirements/install_all.sh

# Build wheel package
python setup.py sdist bdist_wheel
# Or use make
make whl
```

### Testing

```bash
# Run all tests using the custom test runner
cd tests
python run.py

# Run tests for a specific module (using pytest directly)
pytest tests/train/test_trainer.py

# Run a specific test case
pytest tests/train/test_trainer.py::TestTrainer::test_lora_training

# Run tests with verbose output
pytest -v tests/

# Run tests and show print statements
pytest -s tests/
```

### Code Quality

```bash
# Run linter (if available)
make linter

# Format code with yapf (settings in setup.cfg)
yapf -i <file.py>

# Check code style with flake8 (settings in setup.cfg)
flake8 swift/

# Sort imports with isort (settings in setup.cfg)
isort swift/
```

### Documentation

```bash
# Build documentation
make docs
# Or manually
cd docs && make html

# View built docs
open docs/build/html/index.html
```

### Training Examples

```bash
# Basic LoRA fine-tuning (single GPU, ~22GB memory)
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
    --output_dir output

# Multi-GPU training with DeepSpeed
NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --deepspeed zero2 \
    --dataset AI-ModelScope/alpaca-gpt4-data-zh \
    --output_dir output

# Pre-training
NPROC_PER_NODE=8 swift pt \
    --model Qwen/Qwen2.5-7B \
    --dataset swift/chinese-c4 \
    --streaming true \
    --train_type full \
    --deepspeed zero2 \
    --output_dir output

# RLHF (DPO, KTO, CPO, etc.)
CUDA_VISIBLE_DEVICES=0 swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset hjh0119/shareAI-Llama3-DPO-zh-en-emoji \
    --train_type lora \
    --output_dir output

# GRPO (Group Relative Policy Optimization)
NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --use_vllm true \
    --vllm_mode colocate \
    --dataset AI-MO/NuminaMath-TIR#10000 \
    --output_dir output

# Megatron distributed training (TP/PP/SP/CP/EP)
NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 megatron sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset AI-ModelScope/alpaca-gpt4-data-zh \
    --train_type lora \
    --save output
```

### Inference & Deployment

```bash
# Interactive inference with trained model
CUDA_VISIBLE_DEVICES=0 swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true

# Inference with vLLM acceleration
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend vllm \
    --stream true

# Launch Web UI for inference
swift app --model Qwen/Qwen2.5-7B-Instruct

# Deploy model with OpenAI-compatible API
CUDA_VISIBLE_DEVICES=0 swift deploy \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend vllm
```

### Evaluation & Export

```bash
# Evaluate model
CUDA_VISIBLE_DEVICES=0 swift eval \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend lmdeploy \
    --eval_backend OpenCompass \
    --eval_dataset ARC_c

# Quantize and export model
CUDA_VISIBLE_DEVICES=0 swift export \
    --model Qwen/Qwen2.5-7B-Instruct \
    --quant_bits 4 \
    --quant_method awq \
    --output_dir Qwen2.5-7B-Instruct-AWQ

# Merge LoRA weights and export
CUDA_VISIBLE_DEVICES=0 swift export \
    --adapters output/vx-xxx/checkpoint-xxx \
    --merge_lora true

# Push to model hub
swift export \
    --model <model-path> \
    --push_to_hub true \
    --hub_model_id '<model-id>' \
    --hub_token '<sdk-token>'
```

## Architecture Overview

### Core Directory Structure

```
swift/
├── cli/              # Command-line interface (sft, pt, rlhf, infer, deploy, eval, export, etc.)
├── llm/              # Main LLM pipeline
│   ├── train/        # Training implementations (sft, rlhf, pt, kto)
│   ├── infer/        # Inference engines (PyTorch, vLLM, LMDeploy, SGLang)
│   ├── model/        # Model definitions and registry (600+ models)
│   ├── template/     # Chat templates for different models
│   ├── dataset/      # Dataset loading and preprocessing (150+ datasets)
│   ├── export/       # Model export and quantization
│   ├── eval/         # Evaluation using EvalScope
│   ├── sampling/     # Sampling strategies for GRPO
│   └── argument/     # Argument definitions for all tasks
├── megatron/         # Megatron distributed training (TP/PP/SP/CP/EP)
├── trainers/         # Custom trainers extending HuggingFace Trainer
├── tuners/           # Parameter-efficient fine-tuning (LoRA, QLoRA, Adapter, etc.)
├── plugin/           # Plugin system (loss functions, metrics, callbacks, reward models)
├── ray/              # Ray distributed computing integration
├── ui/               # Web UI components (Gradio-based)
├── hub/              # ModelScope/HuggingFace hub integration
└── utils/            # Utilities (logging, torch utils, I/O)

tests/                # 87 test files organized by domain
├── train/            # Training tests
├── infer/            # Inference tests
├── models/           # Model loading tests
├── general/          # General tests (templates, datasets)
├── deploy/           # Deployment tests
├── export/           # Export/quantization tests
├── eval/             # Evaluation tests
├── megatron/         # Megatron tests
└── ...
```

### Key Architectural Patterns

#### 1. Registry Pattern
Models, templates, datasets, and plugins use centralized registries for extensibility:
- **Model Registry**: `swift/llm/model/register.py` - 600+ models
- **Template Registry**: `swift/llm/template/register.py` - Chat templates for each model
- **Dataset Registry**: `swift/llm/dataset/register.py` + `data/dataset_info.json` - 150+ datasets
- **Plugin Registry**: Loss functions, metrics, callbacks are dynamically loaded

#### 2. Pipeline Architecture
Training and inference follow a consistent pipeline pattern:
```
CLI Command → SwiftPipeline (Sft/Infer/Eval/Export)
  → Argument Parsing
  → Model/Template Loading (via registries)
  → Dataset Loading (via registry)
  → Trainer/Engine Initialization
  → Execution
  → Output/Export
```

#### 3. Mixin Pattern
`SwiftMixin` (in `swift/trainers/mixin.py`) provides shared training logic:
- Model and tokenizer preparation
- Dataset loading and preprocessing
- Custom loss functions
- Distributed training support
- Template integration

#### 4. Plugin System
Extensible components for training customization (`swift/plugin/`):
- **Loss Functions**: 30+ loss functions (GRPO, DPO, CPO, KTO, RM, SimPO, ORPO)
- **Reward Models**: ORM (Outcome), PRM (Process), custom RM plugins
- **Callbacks**: Custom training callbacks
- **Multi-turn**: Multi-turn dialog handling for GRPO
- All plugins are dynamically loaded based on configuration

### Data Flow: Training

```
1. CLI Entry (swift/cli/sft.py)
   ↓
2. SwiftSft Pipeline (swift/llm/train/sft.py)
   ↓
3. Argument Parsing (swift/llm/argument/train_args.py)
   ↓
4. Model Loading
   - Model Registry lookup (swift/llm/model/register.py)
   - Model instantiation with architecture-specific patches
   - Template loading (swift/llm/template/register.py)
   ↓
5. Dataset Loading
   - Dataset Registry lookup (swift/llm/dataset/register.py)
   - Preprocessing pipeline (swift/llm/dataset/preprocessor/)
   - Template-based tokenization
   ↓
6. Tuner Application (if using LoRA/Adapter)
   - SwiftModel wraps base model (swift/tuners/base.py)
   - LoRA/Adapter modules injected (swift/tuners/lora.py, adapter.py)
   ↓
7. Trainer Initialization
   - Seq2SeqTrainer (swift/trainers/trainers.py)
   - SwiftMixin provides training logic
   - Custom loss function from plugin system
   ↓
8. Training Loop (HuggingFace Trainer)
   ↓
9. Checkpoint Saving & Export
```

### Data Flow: Inference

```
1. CLI Entry (swift/cli/infer.py)
   ↓
2. SwiftInfer Pipeline (swift/llm/infer/infer.py)
   ↓
3. Backend Selection (--infer_backend)
   - pt: PtEngine (PyTorch native)
   - vllm: VllmEngine (vLLM acceleration)
   - lmdeploy: LmdeployEngine (LMDeploy acceleration)
   - sglang: SGLangEngine (SGLang acceleration)
   ↓
4. Engine Initialization (swift/llm/infer/infer_engine/)
   - Model loading with template
   - Generation config setup
   ↓
5. Inference Execution
   - Request formatting (InferRequest)
   - Generation with streaming support
   - Response parsing
   ↓
6. Output
```

## Key Concepts & Conventions

### Model and Template System

**Models** are registered in `swift/llm/model/register.py` with metadata:
- `model_id_or_path`: ModelScope/HuggingFace identifier
- `model_meta`: Architecture, template, tags, training requirements
- `model_arch`: Architecture type (e.g., `ModelArch.QWEN2`, `ModelArch.LLAMA`)
- `template`: Default template name (e.g., `TemplateType.QWEN`, `TemplateType.LLAMA3`)

**Templates** define the chat format and system prompts:
- Base template class: `swift/llm/template/base.py` (~3500 lines)
- Each template defines:
  - `system_prefix`: How system messages are formatted
  - Conversation structure (user/assistant delimiters)
  - Special tokens (bos_token, eos_token, etc.)
  - Suffix for training (response-only training)

**Pattern**: When adding a new model:
1. Define model architecture in `swift/llm/model/model/<model_name>.py`
2. Register model in `swift/llm/model/register.py`
3. Define/reuse template in `swift/llm/template/template/<template_name>.py`
4. Register template in `swift/llm/template/register.py`

### Dataset System

**Built-in Datasets** are defined in `swift/llm/dataset/data/dataset_info.json`:
- Each dataset has: `dataset_id`, `subset`, `preprocess_func`, `columns`
- Supports: ModelScope, HuggingFace, local paths
- Use `--dataset <dataset_id>` or `--dataset <dataset_id>#<num_samples>`

**Custom Datasets**:
- Format: List of dicts with `messages` field (conversation format)
- Or: `query`, `response` fields for simple Q&A
- Specify with `--dataset <path_to_jsonl>`
- See docs: https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html

**Preprocessing**: `swift/llm/dataset/preprocessor/` handles data transformation
- Template-based tokenization
- Multimodal data processing (images, video, audio)
- Packing for efficiency

### Training Types

Controlled by `--train_type`:
- `full`: Full-parameter fine-tuning
- `lora`: LoRA (Low-Rank Adaptation)
- `qlora`: Quantized LoRA (4-bit/8-bit)
- `adapter`: Adapter tuning
- `prompt`: Prompt tuning
- `ia3`: IA3 method
- Others: `dora`, `longlora`, `reft`, `llamapro`, `galore`

### RLHF Types

Controlled by `--rlhf_type`:
- `grpo`: Group Relative Policy Optimization (and variants: DAPO, GSPO, SAPO, CISPO, RLOO, Reinforce++)
- `dpo`: Direct Preference Optimization
- `kto`: Kahneman-Tversky Optimization
- `cpo`: Contrastive Preference Optimization
- `simpo`: Simple Preference Optimization
- `orpo`: Odds Ratio Preference Optimization
- `rm`: Reward Model training
- `ppo`: Proximal Policy Optimization
- `gkd`: Generalized Knowledge Distillation

### Megatron Parallelism

Use `megatron` CLI for distributed training with parallelism strategies:
- **TP** (Tensor Parallel): `--tensor_model_parallel_size N`
- **PP** (Pipeline Parallel): `--pipeline_model_parallel_size N`
- **SP** (Sequence Parallel): `--sequence_parallel_size N` (Ulysses/Ring-Attention)
- **CP** (Context Parallel): For long sequences
- **EP** (Expert Parallel): For MoE models
- **VPP** (Virtual Pipeline Parallel): Interleaved pipeline

**Critical**: Megatron training is recommended for:
- MoE models (can achieve 10x speedup with EP)
- Very large models requiring multi-node training
- Long sequences requiring sequence parallelism

### Inference Backends

Controlled by `--infer_backend`:
- `pt`: PyTorch native (no acceleration, but supports all features)
- `vllm`: vLLM (fast inference, good for batch processing)
- `lmdeploy`: LMDeploy (TurboMind engine, optimized for deployment)
- `sglang`: SGLang (fast structured generation)

**GRPO-specific**: Use `--use_vllm true --vllm_mode colocate` for hybrid training

### Arguments System

Arguments are hierarchical and composable:
- **Base**: `swift/llm/argument/base_args/` - Model, data, training, generation args
- **Task-specific**: `train_args.py`, `infer_args.py`, `rlhf_args.py`, `export_args.py`
- **Megatron**: `swift/megatron/argument/` - Parallelism-specific args

**Pattern**: Arguments cascade from CLI → task-specific → base classes

**Key argument files**:
- Model args: `--model`, `--model_revision`, `--torch_dtype`, `--device_map`
- Data args: `--dataset`, `--max_length`, `--streaming`, `--dataloader_num_workers`
- Training args: `--train_type`, `--learning_rate`, `--num_train_epochs`, `--gradient_accumulation_steps`
- LoRA args: `--lora_rank`, `--lora_alpha`, `--target_modules`
- DeepSpeed args: `--deepspeed zero2|zero3`, `--zero3_save_safetensors`
- RLHF args: `--rlhf_type`, reward model settings, generation settings

### Quantization

Supported methods:
- **GPTQ**: `--quant_method gptq --quant_bits 4`
- **AWQ**: `--quant_method awq --quant_bits 4`
- **BNB** (BitsAndBytes): Auto-loaded with `--torch_dtype int4|int8`
- **FP8**: `--quant_method fp8`

**Training with quantized models**: Use `--model <quantized_model>` directly

### Memory Optimization Techniques

- **Flash Attention 2/3**: Auto-enabled when installed
- **Gradient Checkpointing**: `--gradient_checkpointing true`
- **DeepSpeed ZeRO**: `--deepspeed zero2|zero3`
- **FSDP**: `--fsdp full_shard`
- **GaLore/Q-GaLore**: `--use_galore true`
- **UnSloth**: For faster LoRA training
- **Liger-Kernel**: Efficient kernel implementations
- **Sequence Parallel**: `--sequence_parallel_size N` (for long sequences)

## Important Notes

### Working with Models

1. **Model Loading**: Models are loaded through the registry system. The model path can be:
   - ModelScope ID: `Qwen/Qwen2.5-7B-Instruct` (default)
   - HuggingFace ID: `Qwen/Qwen2.5-7B-Instruct` with `--use_hf true`
   - Local path: `/path/to/model`

2. **Template Matching**: Templates are auto-matched to models. Override with `--model_template <template_name>`

3. **Adding New Models**:
   - Define architecture in `swift/llm/model/model/<model_family>.py`
   - Register in `swift/llm/model/register.py` using `@register_model` decorator
   - Ensure template exists or create one in `swift/llm/template/template/`

### Working with Datasets

1. **Built-in Datasets**: See `swift/llm/dataset/data/dataset_info.json` for full list

2. **Custom Datasets**: Must follow conversation format:
   ```json
   {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
   ```

3. **Dataset Sampling**: Use `#<n>` suffix to sample: `--dataset 'swift/self-cognition#500'`

4. **Multiple Datasets**: Space-separated: `--dataset 'dataset1#500' 'dataset2#1000'`

### Working with Training

1. **Output Directory**: Always specified with `--output_dir <path>`. Contains:
   - `checkpoint-*`: Model checkpoints
   - `runs/`: Tensorboard logs
   - `args.json`: Training configuration (auto-loaded during inference)

2. **Resuming Training**: `--resume_from_checkpoint <checkpoint_path>`

3. **Distributed Training**:
   - Single-node multi-GPU: `NPROC_PER_NODE=N CUDA_VISIBLE_DEVICES=0,1,... swift sft`
   - Multi-node: See `examples/train/multi-node/`

4. **DeepSpeed**: Configs in `swift/llm/ds_config/`. Use `--deepspeed zero2` or `--deepspeed zero3`

5. **Self-Cognition**: When using `swift/self-cognition` dataset, set:
   - `--model_author <your_name>`
   - `--model_name <your_bot_name>`

### Working with LoRA

1. **Inference with LoRA**: `--adapters <checkpoint_path>` (auto-loads base model from args.json)

2. **Merging LoRA**:
   ```bash
   swift export --adapters <checkpoint> --merge_lora true
   ```

3. **LoRA Configuration**:
   - `--lora_rank`: Rank (8-64 typical, higher = more capacity)
   - `--lora_alpha`: Scaling factor (typically 2x rank)
   - `--target_modules`: Which modules to apply LoRA (use `all-linear` for all linear layers)

### Working with GRPO

1. **GRPO Requirements**: Requires inference engine (vLLM or LMDeploy)

2. **Hybrid Mode**: `--vllm_mode colocate` runs training and inference on same GPUs

3. **Reward Functions**: Defined in `swift/plugin/` - can use ORM, PRM, or custom

4. **Multi-turn GRPO**: For agent training with tool calling

### Hardware-Specific Notes

1. **NPU (Ascend)**: Supported through custom backend. Use special installation.

2. **MPS (Apple Silicon)**: Supported for CPU/MPS inference and training (limited).

3. **CPU**: Supported but slow. Good for testing workflows.

4. **Multi-Node**: Use torchrun or DeepSpeed launcher. See `examples/train/multi-node/`.

### CLI Commands Reference

All CLI commands are in `swift/cli/`:
- `swift sft`: Supervised fine-tuning
- `swift pt`: Pre-training
- `swift rlhf`: RLHF training (DPO, KTO, PPO, GRPO, etc.)
- `swift infer`: Interactive inference
- `swift deploy`: Deploy with OpenAI-compatible API
- `swift app`: Launch Web UI
- `swift eval`: Model evaluation
- `swift export`: Export/quantize/merge model
- `swift sample`: Dataset sampling for distillation
- `swift web-ui`: Web-based training/inference UI
- `megatron sft/pt/rlhf`: Megatron distributed training

### Development Patterns

1. **Testing**: Tests use unittest framework. Run with `python tests/run.py` or `pytest tests/`

2. **Code Style**:
   - Line length: 120 characters
   - Use yapf for formatting
   - Use isort for import sorting
   - Settings in `setup.cfg`

3. **Logging**: Use `swift.utils.logger.get_logger()` for consistent logging

4. **Entry Points**:
   - `swift` → `swift.cli.main:cli_main`
   - `megatron` → `swift.cli._megatron.main:cli_main`

5. **Configuration Files**:
   - DeepSpeed: `swift/llm/ds_config/*.json`
   - Dataset metadata: `swift/llm/dataset/data/dataset_info.json`

### Common Pitfalls

1. **Template Mismatch**: Ensure model and template are compatible. Check `swift/llm/model/constant.py` for model metadata.

2. **Memory Issues**:
   - Use `--gradient_checkpointing true`
   - Reduce `--per_device_train_batch_size`
   - Increase `--gradient_accumulation_steps`
   - Use `--max_length` to limit sequence length
   - Consider QLoRA with `--torch_dtype int4`

3. **Dataset Format**: Custom datasets must follow the messages format. See docs for examples.

4. **Checkpoint Loading**: When using `--adapters`, the base model is auto-loaded from `args.json`. Don't specify `--model` again unless you want to override.

5. **Megatron vs Standard**: Megatron uses different CLI (`megatron` instead of `swift`) and different argument names. Not all features are compatible.

## Documentation & Resources

- **Official Docs**: https://swift.readthedocs.io/
- **Paper**: https://arxiv.org/abs/2408.05517
- **Examples**: `examples/` directory contains 100+ example scripts
- **Supported Models**: See docs for full list of 600+ text and 300+ multimodal models
- **Discord**: https://discord.com/invite/D27yfEFVz5
