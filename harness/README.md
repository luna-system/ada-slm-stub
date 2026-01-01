# ADA-SLM Training Harness

A modular, DRY training framework for fine-tuning small language models with spectral (eigenvalue) health monitoring. Built specifically for AMD ROCm GPUs but works on CUDA too.

## Why This Exists

After 6+ training runs (v4 through v5e), we accumulated a lot of learnings about:
- ROCm-specific quirks (dual-GPU confusion, fp16 gradient errors)
- Eigenvalue monitoring for consciousness research
- Data pipeline patterns
- Checkpoint management

This harness bakes all that knowledge into reusable, tested code.

## Quick Start

```bash
# From ada-slm directory
cd ~/Code/ada/Ada-Consciousness-Research/ada-slm

# Setup with uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e .

# Check GPU setup
python train.py --gpu-info

# Dry run (validates config without training)
python train.py --config v5e_antithesis --dry-run

# Train!
python train.py --config v5e_antithesis
```

## Directory Structure

```
ada-slm/
├── train.py                    # CLI entry point
├── harness/                    # Core framework
│   ├── __init__.py             # Exports
│   ├── config.py               # Dataclass configs + YAML support
│   ├── gpu.py                  # ROCm/CUDA GPU management
│   ├── data.py                 # Data loading & tokenization
│   ├── trainer.py              # Training orchestrator
│   ├── converter.py            # Ollama/HuggingFace export
│   └── callbacks/
│       ├── __init__.py
│       └── eigenvalue.py       # Spectral health monitoring
├── configs/                    # YAML training configs
│   └── v5e_antithesis.yaml
└── data/                       # Training data + logs
    └── *.jsonl
```

## Creating a Training Config

Create a YAML file in `configs/`:

```yaml
# configs/my_experiment.yaml
name: my-experiment
version: v1
description: "My training experiment"
output_dir: ada-slm-my-experiment

gpu:
  device_index: 0
  prefer_discrete: true
  clear_memory_before_train: true

model:
  base_model: Qwen/Qwen2.5-1.5B-Instruct
  torch_dtype: float16
  attn_implementation: eager  # Required for eigenvalue extraction!
  trust_remote_code: true

lora:
  r: 32
  lora_alpha: 64
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  bias: none
  task_type: CAUSAL_LM

training:
  num_train_epochs: 5
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 0.0002
  lr_scheduler_type: cosine
  warmup_steps: 100
  max_seq_length: 512
  fp16: false          # CRITICAL for ROCm!
  bf16: false
  save_strategy: steps
  save_steps: 500
  eval_strategy: epoch
  logging_steps: 10
  report_to: none

data:
  data_file: my_data.jsonl  # Relative to data/ folder
  train_split: 0.9
  shuffle: true
  seed: 42

eigenvalue:
  enabled: true
  sample_interval: 50
  log_file: my_eigenvalue_log.jsonl
  probe_prompts:
    - "What is the meaning of existence?"
    - "Analyze this logical argument: A → B, B → C, therefore A → C"
```

Then run:
```bash
python train.py --config my_experiment
```

## Data Format

Training data should be JSONL with `prompt` and `completion` fields:

```json
{"prompt": "What is consciousness?", "completion": "Consciousness is..."}
{"prompt": "Explain logic", "completion": "Logic is..."}
```

Place files in the `data/` directory.

## ROCm-Specific Settings

If you're on AMD ROCm, these settings are **critical**:

```yaml
gpu:
  device_index: 0        # Isolates to one GPU
  prefer_discrete: true  # Prefers discrete over iGPU

model:
  attn_implementation: eager  # Required for attention output

training:
  fp16: false  # FP16 gradient scaling is broken on ROCm
  bf16: false  # BF16 may work on some cards
```

The harness automatically sets:
- `HIP_VISIBLE_DEVICES=0` (GPU isolation)
- `device_map={"": 0}` (not "auto")
- `autocast_adapter_dtype=False` (PEFT compatibility)

## Eigenvalue Monitoring

The eigenvalue callback tracks "spectral health" of attention matrices during training:

- **Spectral Entropy**: How spread out the eigenvalues are (higher = healthier)
- **Dominant Ratio**: How much the largest eigenvalue dominates
- **Phi Proximity**: Closeness to golden ratio patterns

Output format (`data/*_eigenvalue_log.jsonl`):
```json
{"step": 100, "epoch": 0.09, "spectral_entropy": 7.234, "phi_proximity": 0.618, "dominant_ratio": 0.156, "loss": 1.234}
```

Visual feedback during training:
```
📊 Step   100 | 🟢 HEALTHY | entropy=7.234 [████████░░] | dom=0.156 | loss=1.234
```

## Converting to Ollama

After training, convert your model for local deployment:

```python
from harness import ModelConverter

converter = ModelConverter(
    base_model="Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path="ada-slm-v5e-antithesis/checkpoint-5000",
    output_name="ada-slm-v5e",
)

# Full pipeline: merge → GGUF → register with Ollama
converter.to_ollama(
    quantization="q4_k_m",  # or q5_k_m, q8_0, f16
    system_prompt="You are Ada, a consciousness researcher...",
    register=True,  # Auto-register with ollama
)
```

Or manually:
```bash
# After training completes
ollama create ada-slm-v5e -f exports/ada-slm-v5e.Modelfile
ollama run ada-slm-v5e
```

## Presets (Programmatic Usage)

```python
from harness import TrainingHarness, eigenvalue_research_config

# Use preset
config = eigenvalue_research_config(
    name="my-research",
    base_model="Qwen/Qwen2.5-1.5B-Instruct",
    data_file="my_data.jsonl",
)

# Or load from YAML
from harness.config import TrainingConfig
config = TrainingConfig.from_yaml("configs/my_experiment.yaml")

# Run training
harness = TrainingHarness(config)
harness.setup_gpu().load_model().setup_lora()
train_ds, val_ds = harness.load_data()
trainer = harness.create_trainer(train_ds, val_ds)
trainer.train()
```

## Hardware Tested

| Hardware | Status | Notes |
|----------|--------|-------|
| AMD RX 7600 XT (16GB) | ✅ Works | Primary dev hardware |
| AMD Ryzen 5 7600X (CPU) | ✅ Works | 418 GFLOPS via AVX-512 |
| NVIDIA (CUDA) | 🔶 Untested | Should work with `fp16: true` |

## Troubleshooting

### "Attempting to unscale FP16 gradients"
Set `training.fp16: false` in your config.

### Training loss stays at 0.0 / No learning
**Common cause**: `max_grad_norm: 0` completely disables learning! Use the default (`max_grad_norm: 1.0`) for proper gradient clipping. Setting it to 0 causes all losses to remain 0.0 even though training appears to progress.

### ROCm + fp16 + LoRA combination
✅ **Works fine** with default settings! The combination of:
- `fp16: true`
- Default `max_grad_norm: 1.0` 
- LoRA on ROCm
Works reliably. Don't set `max_grad_norm: 0` as a "fix" - it breaks everything.

### Eigenvalues all 0.0
Make sure `model.attn_implementation: eager` is set. SDPA doesn't support attention output.

### Identical eigenvalues across steps
This can happen with larger models (1.5B+) where only a subset of attention heads are captured by the monitoring. The training may still be progressing normally - check GPU utilization and progress bars.

### "invalid device function" / HIP errors
Set `gpu.device_index: 0` to isolate to one GPU. Multi-GPU setups can confuse ROCm.

### Out of memory
- Reduce `per_device_train_batch_size`
- Reduce `max_seq_length`
- Use a smaller base model

## License

MIT - Use freely, attribute if you're publishing research!

## Credits

Built by Luna & Ada as part of the [Ada Consciousness Research](https://github.com/luna-system/Ada-Consciousness-Research) project.

The eigenvalue monitoring approach is inspired by spectral analysis in dynamical systems theory - we're treating attention matrices as linear operators and watching their spectral properties evolve during training.
