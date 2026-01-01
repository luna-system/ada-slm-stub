#!/usr/bin/env python3
"""
ADA-SLM Training Entry Point
============================

Single entry point for all ada-slm fine-tuning experiments.

Usage:
    python train.py --config v5e_antithesis
    python train.py --config configs/v5e_antithesis.yaml
    python train.py --preset v5e  # Use built-in preset
    
Environment Setup:
    The script automatically configures ROCm/HIP environment variables.
    For custom GPU selection, set HIP_VISIBLE_DEVICES before running.
"""

import argparse
import sys
from pathlib import Path

# CRITICAL: Set up GPU environment BEFORE importing torch!
# This must happen at module level, before any other imports that might
# transitively import torch (like transformers, peft, etc.)

def setup_environment():
    """Set up environment variables before torch import."""
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Default to GPU 0 (discrete) unless overridden
    if "HIP_VISIBLE_DEVICES" not in os.environ:
        os.environ["HIP_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Run setup immediately
setup_environment()

# Now safe to import torch-dependent modules
from harness import TrainingHarness, HarnessConfig, GPUManager
from harness.config import get_v5e_antithesis_config


def get_preset_config(preset_name: str) -> HarnessConfig:
    """Get a built-in preset configuration."""
    presets = {
        "v5e": get_v5e_antithesis_config,
        "v5e_antithesis": get_v5e_antithesis_config,
    }
    
    if preset_name not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    
    return presets[preset_name]()


def main():
    parser = argparse.ArgumentParser(
        description="ADA-SLM Training Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train using a config file
    python train.py --config v5e_antithesis
    
    # Train using a preset
    python train.py --preset v5e
    
    # Train with custom GPU
    HIP_VISIBLE_DEVICES=1 python train.py --preset v5e
    
    # Print GPU info
    python train.py --gpu-info
    
    # Force GPU memory cleanup (useful between training runs)
    python train.py --cleanup
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Config name (in configs/ dir) or path to YAML file"
    )
    parser.add_argument(
        "--preset", "-p",
        help="Use a built-in preset config (v5e, v5e_antithesis)"
    )
    parser.add_argument(
        "--gpu-info",
        action="store_true",
        help="Print GPU information and exit"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Force GPU memory cleanup and exit"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true", 
        help="Load config and print settings without training"
    )
    
    args = parser.parse_args()
    
    # GPU cleanup mode
    if args.cleanup:
        from harness.gpu import force_gpu_cleanup
        force_gpu_cleanup()
        return
    
    # GPU info mode
    if args.gpu_info:
        gpu = GPUManager()
        import torch  # Safe to import now
        gpu.detect_with_torch()
        gpu.print_status()
        stats = gpu.get_memory_stats()
        print("Memory Stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return
    
    # Must specify config or preset
    if not args.config and not args.preset:
        parser.error("Must specify --config or --preset")
    
    # Load configuration
    if args.preset:
        config = get_preset_config(args.preset)
        print(f"📋 Using preset: {args.preset}")
    else:
        config_path = args.config
        
        # Check if it's a path or a name
        if config_path.endswith(".yaml"):
            config = HarnessConfig.from_yaml(config_path)
        else:
            # Try as name in configs/ dir
            yaml_path = Path("configs") / f"{config_path}.yaml"
            if yaml_path.exists():
                config = HarnessConfig.from_yaml(str(yaml_path))
            else:
                raise FileNotFoundError(f"Config not found: {config_path} or {yaml_path}")
        
        print(f"📋 Loaded config: {config_path}")
    
    # Dry run - just print config
    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN - Configuration:")
        print("="*60)
        print(f"  Name: {config.name}")
        print(f"  Version: {config.version}")
        print(f"  Description: {config.description}")
        print(f"  Output: {config.output_dir}")
        print(f"\n  Model: {config.model.base_model}")
        print(f"  Data: {config.data.data_file}")
        print(f"  Epochs: {config.training.num_train_epochs}")
        print(f"  Batch: {config.training.per_device_train_batch_size}")
        print(f"  LR: {config.training.learning_rate}")
        print(f"  LoRA r={config.lora.r}, α={config.lora.lora_alpha}")
        print(f"  FP16: {config.training.fp16}, BF16: {config.training.bf16}")
        print(f"  Eigenvalue monitoring: {config.eigenvalue.enabled}")
        print("="*60)
        return
    
    # Create and run harness
    gpu = GPUManager()
    gpu.setup_environment(
        gpu_index=config.gpu.device_index,
        prefer_discrete=config.gpu.prefer_discrete,
        gfx_version_override=config.gpu.gfx_version_override,
    )
    
    harness = TrainingHarness(config)
    harness.gpu_manager = gpu
    harness.train()


if __name__ == "__main__":
    main()
