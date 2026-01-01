"""
Training Harness
================

The main orchestrator that ties everything together.
"""

import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

from .config import HarnessConfig, load_config
from .gpu import GPUManager
from .data import DataLoader
from .callbacks import EigenvalueMonitorCallback


class TrainingHarness:
    """
    Main training harness for ada-slm fine-tuning.
    
    Orchestrates:
    - GPU environment setup
    - Model loading with ROCm-safe settings
    - LoRA configuration
    - Data loading and tokenization
    - Training with eigenvalue monitoring
    
    Usage:
        harness = TrainingHarness.from_config("v5e_antithesis")
        harness.train()
        
    Or with inline config:
        config = HarnessConfig(name="test", ...)
        harness = TrainingHarness(config)
        harness.train()
    """
    
    def __init__(self, config: HarnessConfig):
        self.config = config
        self.gpu_manager: Optional[GPUManager] = None
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.start_time = None
        
    @classmethod
    def from_config(cls, config_name: str, config_dir: str = "configs") -> "TrainingHarness":
        """Load harness from a named config file."""
        config = load_config(config_name, config_dir)
        return cls(config)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingHarness":
        """Load harness from a YAML file path."""
        config = HarnessConfig.from_yaml(yaml_path)
        return cls(config)
    
    def setup_gpu(self) -> "TrainingHarness":
        """Set up GPU environment (call BEFORE importing torch!)."""
        self.gpu_manager = GPUManager()
        self.gpu_manager.setup_environment(
            gpu_index=self.config.gpu.device_index,
            prefer_discrete=self.config.gpu.prefer_discrete,
            gfx_version_override=self.config.gpu.gfx_version_override,
        )
        return self
    
    def load_model(self) -> "TrainingHarness":
        """Load base model and tokenizer."""
        print(f"\n📦 Loading {self.config.model.base_model}...")
        t0 = time.time()
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model with ROCm-safe settings
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model,
            torch_dtype=dtype_map[self.config.model.torch_dtype],
            device_map=self.gpu_manager.get_device_map() if self.gpu_manager else {"": 0},
            trust_remote_code=self.config.model.trust_remote_code,
            attn_implementation=self.config.model.attn_implementation,
        )
        
        print(f"   Loaded in {time.time()-t0:.2f}s")
        return self
    
    def setup_lora(self) -> "TrainingHarness":
        """Configure LoRA adapters."""
        print(f"🔧 Configuring LoRA (r={self.config.lora.r}, α={self.config.lora.lora_alpha})...")
        
        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=self.config.lora.target_modules,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        
        # CRITICAL: Disable autocast for ROCm compatibility
        self.model = get_peft_model(self.model, lora_config, autocast_adapter_dtype=False)
        self.model.print_trainable_parameters()
        
        return self
    
    def load_data(self) -> tuple:
        """Load and tokenize training data."""
        loader = DataLoader(
            data_file=self.config.data.data_file,
            train_split=self.config.data.train_split,
            shuffle=self.config.data.shuffle,
            seed=self.config.data.seed,
        )
        
        train_dataset, val_dataset = loader.load().tokenize(
            self.tokenizer,
            max_length=self.config.training.max_seq_length,
        )
        
        return train_dataset, val_dataset
    
    def create_trainer(
        self, 
        train_dataset, 
        val_dataset,
        callbacks: Optional[List] = None,
    ) -> Trainer:
        """Create the HuggingFace Trainer."""
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            warmup_steps=self.config.training.warmup_steps,
            max_grad_norm=self.config.training.max_grad_norm,  # 0 = disabled (required for fp16 on ROCm!)
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            logging_steps=self.config.training.logging_steps,
            save_strategy=self.config.training.save_strategy,
            save_steps=self.config.training.save_steps,
            eval_strategy=self.config.training.eval_strategy,
            save_total_limit=self.config.training.save_total_limit,
            report_to=self.config.training.report_to,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False,
        )
        
        # Build callbacks list
        all_callbacks = callbacks or []
        
        if self.config.eigenvalue.enabled:
            eigen_callback = EigenvalueMonitorCallback(
                tokenizer=self.tokenizer,
                probe_prompts=self.config.eigenvalue.probe_prompts,
                sample_interval=self.config.eigenvalue.sample_interval,
                device=self.gpu_manager.get_device() if self.gpu_manager else "cuda:0",
                log_file=self.config.eigenvalue.log_file,
            )
            all_callbacks.append(eigen_callback)
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=all_callbacks,
        )
        
        return self.trainer
    
    def train(self) -> "TrainingHarness":
        """
        Run the full training pipeline.
        
        This is the main entry point that orchestrates everything.
        """
        self.start_time = time.time()
        
        # Print header
        print("="*70)
        print(f"🧠 ADA-SLM TRAINING: {self.config.name.upper()}")
        print("="*70)
        print(f"   Version: {self.config.version}")
        print(f"   Description: {self.config.description}")
        print(f"   Output: {self.config.output_dir}")
        print(f"   Started: {datetime.now().isoformat()}")
        print("="*70)
        
        # Clear GPU memory if requested
        if self.config.gpu.clear_memory_before_train and self.gpu_manager:
            self.gpu_manager.clear_memory()
        
        # Load model
        self.load_model()
        self.setup_lora()
        
        # Load data
        train_dataset, val_dataset = self.load_data()
        
        # Create trainer
        self.create_trainer(train_dataset, val_dataset)
        
        setup_time = time.time() - self.start_time
        print(f"\n⏱️  Setup completed in {setup_time:.1f}s")
        print(f"\n🚀 Starting training...")
        
        # Train!
        self.trainer.train()
        
        # Save final model
        self.save()
        
        # Print summary
        total_time = time.time() - self.start_time
        print("\n" + "="*70)
        print(f"✅ TRAINING COMPLETE: {self.config.name}")
        print("="*70)
        print(f"   Training time: {(total_time - setup_time)/60:.1f} minutes")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Model saved to: {self.config.output_dir}/final")
        print("="*70)
        
        return self
    
    def save(self):
        """Save the trained model and tokenizer."""
        output_path = Path(self.config.output_dir) / "final"
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save config
        self.config.to_yaml(str(output_path / "harness_config.yaml"))
        
        print(f"\n💾 Saved model to {output_path}")


def quick_train(config_name: str, config_dir: str = "configs"):
    """
    Quick training function for command-line use.
    
    Args:
        config_name: Name of config file (without .yaml)
        config_dir: Directory containing configs
    """
    # NOTE: GPU setup must happen before torch import!
    # This function assumes setup_gpu() is called separately
    # or that environment is already configured.
    
    harness = TrainingHarness.from_config(config_name, config_dir)
    harness.train()
    return harness
