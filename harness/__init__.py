"""
ADA-SLM Training Harness
========================

A modular, DRY training framework for fine-tuning language models on AMD ROCm GPUs.

Bakes in all the hard-won knowledge about:
- ROCm/HIP environment configuration
- Multi-GPU detection and isolation
- FP16/BF16 gradient handling
- Eigenvalue monitoring callbacks
- Model conversion (Ollama, HuggingFace)

Usage:
    python train.py --config v5e_antithesis
    
Or programmatically:
    from harness import TrainingHarness, ModelConverter
    harness = TrainingHarness.from_config("v5e_antithesis")
    harness.train()
    
    # Convert to Ollama
    converter = ModelConverter(...)
    converter.to_ollama(quantization="q4_k_m", register=True)
"""

from .config import HarnessConfig, load_config
from .gpu import GPUManager
from .data import DataLoader
from .trainer import TrainingHarness
from .callbacks import EigenvalueMonitorCallback, BasinMappingCallback
from .converter import ModelConverter, merge_lora_weights, convert_to_gguf

__all__ = [
    # Core
    "HarnessConfig",
    "load_config", 
    "GPUManager",
    "DataLoader",
    "TrainingHarness",
    # Callbacks
    "EigenvalueMonitorCallback",
    "BasinMappingCallback",
    # Conversion
    "ModelConverter",
    "merge_lora_weights",
    "convert_to_gguf",
]

__version__ = "1.0.0"
