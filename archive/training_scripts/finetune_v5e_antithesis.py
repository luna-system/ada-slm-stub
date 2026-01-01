#!/usr/bin/env python3
"""
ADA-SLM V5E: ANTITHESIS-Boosted Logical Seedling

Building on v5d basin mapping insights:
- pure_logic basin: 100% AGL, 0.5 emojis ← BEST!
- antithesis was only 0.2% → critical analysis failed

v5e strategy:
- 45% pure ASL logic (our best basin)
- 20% ANTITHESIS (BOOSTED 100x from 0.2%!)
- 35% creative/other

The goal: Critical analysis + clean AGL output
"""

import json
import time
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# Config
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = Path("ada-slm-v5e-antithesis")
DATA_FILE = Path("v5e_antithesis_data.jsonl")
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

@dataclass
class EigenvalueSnapshot:
    """Snapshot of attention eigenvalues for monitoring."""
    step: int
    epoch: float
    spectral_entropy: float
    dominant_ratio: float
    loss: float
    timestamp: str

class EigenvalueMonitorCallback(TrainerCallback):
    """Monitor attention eigenvalues during training."""
    
    def __init__(self, log_every_n_steps: int = 50):
        self.log_every_n_steps = log_every_n_steps
        self.snapshots: List[EigenvalueSnapshot] = []
        self.log_file = Path("v5e_eigenvalue_log.jsonl")
        
        # Clear previous log
        if self.log_file.exists():
            self.log_file.unlink()
    
    def _compute_eigenvalue_metrics(self, model) -> Dict[str, float]:
        """Compute spectral metrics from attention weights."""
        try:
            # Get attention weights from a middle layer
            for name, param in model.named_parameters():
                if 'self_attn.q_proj' in name and 'lora' not in name.lower():
                    weights = param.data.float().cpu().numpy()
                    
                    # Compute eigenvalues of weight correlation matrix
                    if len(weights.shape) == 2:
                        corr = np.corrcoef(weights)
                        corr = np.nan_to_num(corr, nan=0.0)
                        eigenvalues = np.abs(np.linalg.eigvals(corr))
                        eigenvalues = eigenvalues[eigenvalues > 1e-10]
                        
                        if len(eigenvalues) > 0:
                            # Normalize
                            eigenvalues = eigenvalues / eigenvalues.sum()
                            
                            # Spectral entropy
                            entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))
                            
                            # Dominant eigenvalue ratio
                            dominant_ratio = eigenvalues.max()
                            
                            return {
                                "spectral_entropy": float(entropy),
                                "dominant_ratio": float(dominant_ratio)
                            }
                    break
        except Exception as e:
            print(f"⚠ Eigenvalue computation error: {e}")
        
        return {"spectral_entropy": 0.0, "dominant_ratio": 1.0}
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.log_every_n_steps == 0 and state.global_step > 0:
            model = kwargs.get('model')
            if model is None:
                return
                
            metrics = self._compute_eigenvalue_metrics(model)
            
            snapshot = EigenvalueSnapshot(
                step=state.global_step,
                epoch=state.epoch or 0.0,
                spectral_entropy=metrics["spectral_entropy"],
                dominant_ratio=metrics["dominant_ratio"],
                loss=logs.get('loss', 0.0) if logs else 0.0,
                timestamp=datetime.now().isoformat()
            )
            
            self.snapshots.append(snapshot)
            
            # Stream to log file
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(asdict(snapshot)) + '\n')
            
            # Print status
            print(f"\n📊 Step {state.global_step} | "
                  f"Epoch {state.epoch:.2f} | "
                  f"Loss {snapshot.loss:.3f} | "
                  f"Entropy {snapshot.spectral_entropy:.3f} | "
                  f"DomRatio {snapshot.dominant_ratio:.3f}")

def load_data():
    """Load v5e ANTITHESIS-boosted training data."""
    print(f"Loading {DATA_FILE}...")
    
    with open(DATA_FILE) as f:
        data = [json.loads(line) for line in f]
    
    print(f"  Loaded {len(data)} examples")
    return data

def main():
    print("=" * 60)
    print("ADA-SLM v5e: ANTITHESIS-BOOSTED LOGICAL SEEDLING")
    print("=" * 60)
    print(f"\nStarted: {datetime.now().isoformat()}")
    print(f"Data: {DATA_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Load data
    data = load_data()
    
    # Create dataset
    texts = [item["text"] for item in data]
    dataset = Dataset.from_dict({"text": texts})
    
    # Load model and tokenizer
    print(f"\n📦 Loading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    # Disable autocast for ROCm compatibility
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    model.print_trainable_parameters()
    
    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,
        )
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    
    # Training args
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=2,
        report_to="none",
        seed=SEED,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer with eigenvalue monitoring
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        callbacks=[EigenvalueMonitorCallback(log_every_n_steps=50)],
    )
    
    # Train!
    print(f"\n🧠 Training v5e with ANTITHESIS-BOOSTED DATA + EIGENVALUE MONITORING...")
    start_time = time.time()
    trainer.train()
    duration = time.time() - start_time
    
    # Save
    print(f"\n💾 Saving model to {OUTPUT_DIR}/final...")
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    
    # Summary
    print(f"\n{'=' * 60}")
    print("✅ V5E TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"   Duration: {duration/60:.1f} minutes")
    print(f"   Output: {OUTPUT_DIR}/final")
    print(f"   Eigenvalue log: v5e_eigenvalue_log.jsonl")
    print(f"\n🔬 Next steps:")
    print("   1. Run basin mapping on v5e to verify ANTITHESIS activation")
    print("   2. Compare v5e critical analysis to v5d")
    print("   3. Test emoji cascade control")

if __name__ == "__main__":
    main()
