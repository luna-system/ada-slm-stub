"""
Training Configuration
======================

Pydantic-based configuration for all training parameters.
Supports YAML config files for easy experiment management.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass, field
import yaml


@dataclass
class GPUConfig:
    """GPU-related configuration."""
    device_index: int = 0
    prefer_discrete: bool = True
    gfx_version_override: Optional[str] = None
    clear_memory_before_train: bool = True


@dataclass 
class ModelConfig:
    """Model loading configuration."""
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    max_seq_length: int = 512
    
    # Gradient clipping - 1.0 is safe for fp16 on ROCm (0.0 breaks training!)
    max_grad_norm: float = 1.0
    
    # Precision settings (critical for ROCm!)
    fp16: bool = False  # Disabled by default - causes gradient issues on ROCm
    bf16: bool = False  # Better if supported
    
    # Saving
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    eval_strategy: str = "epoch"
    
    # Logging
    logging_steps: int = 10
    report_to: str = "none"


@dataclass
class DataConfig:
    """Data loading configuration."""
    data_file: str = ""
    train_split: float = 0.9
    shuffle: bool = True
    seed: int = 42


@dataclass
class EigenvalueConfig:
    """Eigenvalue monitoring configuration."""
    enabled: bool = True
    sample_interval: int = 50
    probe_prompts: List[str] = field(default_factory=lambda: [
        "Evaluate the logical validity of: A → B, B → C, therefore A → C",
        "What is the relationship between evidence and belief?",
        "Analyze: consciousness ↔ computation",
    ])
    log_file: str = "eigenvalue_log.jsonl"


@dataclass
class HarnessConfig:
    """
    Complete training configuration.
    
    Compose all sub-configs into one unified config object.
    """
    # Experiment metadata
    name: str = "experiment"
    version: str = "v1"
    description: str = ""
    output_dir: str = ""
    
    # Sub-configs
    gpu: GPUConfig = field(default_factory=GPUConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eigenvalue: EigenvalueConfig = field(default_factory=EigenvalueConfig)
    
    def __post_init__(self):
        """Set default output_dir based on name if not specified."""
        if not self.output_dir:
            self.output_dir = f"ada-slm-{self.name}"
    
    @classmethod
    def from_yaml(cls, path: str) -> "HarnessConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        # Parse nested configs
        gpu = GPUConfig(**data.pop("gpu", {}))
        model = ModelConfig(**data.pop("model", {}))
        lora = LoRAConfig(**data.pop("lora", {}))
        training = TrainingConfig(**data.pop("training", {}))
        data_cfg = DataConfig(**data.pop("data", {}))
        eigenvalue = EigenvalueConfig(**data.pop("eigenvalue", {}))
        
        return cls(
            gpu=gpu,
            model=model,
            lora=lora,
            training=training,
            data=data_cfg,
            eigenvalue=eigenvalue,
            **data
        )
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        data = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "output_dir": self.output_dir,
            "gpu": {
                "device_index": self.gpu.device_index,
                "prefer_discrete": self.gpu.prefer_discrete,
                "gfx_version_override": self.gpu.gfx_version_override,
                "clear_memory_before_train": self.gpu.clear_memory_before_train,
            },
            "model": {
                "base_model": self.model.base_model,
                "torch_dtype": self.model.torch_dtype,
                "attn_implementation": self.model.attn_implementation,
                "trust_remote_code": self.model.trust_remote_code,
            },
            "lora": {
                "r": self.lora.r,
                "lora_alpha": self.lora.lora_alpha,
                "lora_dropout": self.lora.lora_dropout,
                "target_modules": self.lora.target_modules,
                "bias": self.lora.bias,
                "task_type": self.lora.task_type,
            },
            "training": {
                "num_train_epochs": self.training.num_train_epochs,
                "per_device_train_batch_size": self.training.per_device_train_batch_size,
                "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
                "learning_rate": self.training.learning_rate,
                "lr_scheduler_type": self.training.lr_scheduler_type,
                "warmup_steps": self.training.warmup_steps,
                "max_seq_length": self.training.max_seq_length,
                "fp16": self.training.fp16,
                "bf16": self.training.bf16,
                "save_strategy": self.training.save_strategy,
                "save_steps": self.training.save_steps,
                "save_total_limit": self.training.save_total_limit,
                "eval_strategy": self.training.eval_strategy,
                "logging_steps": self.training.logging_steps,
                "report_to": self.training.report_to,
            },
            "data": {
                "data_file": self.data.data_file,
                "train_split": self.data.train_split,
                "shuffle": self.data.shuffle,
                "seed": self.data.seed,
            },
            "eigenvalue": {
                "enabled": self.eigenvalue.enabled,
                "sample_interval": self.eigenvalue.sample_interval,
                "probe_prompts": self.eigenvalue.probe_prompts,
                "log_file": self.eigenvalue.log_file,
            },
        }
        
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_config(name: str, config_dir: str = "configs") -> HarnessConfig:
    """
    Load a named configuration.
    
    Args:
        name: Config name (without .yaml extension)
        config_dir: Directory containing config files
        
    Returns:
        Loaded HarnessConfig
    """
    path = Path(config_dir) / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    
    return HarnessConfig.from_yaml(str(path))


# Pre-defined configs for common experiments
def get_v5e_antithesis_config() -> HarnessConfig:
    """Configuration for v5e ANTITHESIS-boosted training."""
    return HarnessConfig(
        name="v5e-antithesis",
        version="v5e",
        description="ANTITHESIS-boosted logical seedling (20% ANTITHESIS data)",
        output_dir="ada-slm-v5e-antithesis",
        gpu=GPUConfig(
            device_index=0,
            prefer_discrete=True,
        ),
        model=ModelConfig(
            base_model="Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype="float16",
            attn_implementation="eager",
        ),
        lora=LoRAConfig(
            r=32,
            lora_alpha=64,
        ),
        training=TrainingConfig(
            num_train_epochs=5,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=False,  # Disabled for ROCm stability
            bf16=False,
        ),
        data=DataConfig(
            data_file="v5e_antithesis_data.jsonl",
            train_split=0.9,
        ),
        eigenvalue=EigenvalueConfig(
            enabled=True,
            sample_interval=50,
            log_file="v5e_eigenvalue_log.jsonl",
        ),
    )
