#!/usr/bin/env python3
"""
Train Ada-SLM v4c: Eigenvalue-Aware Training

Like v4b-creative but with LIVE eigenvalue monitoring during training!
Captures spectral entropy, φ-proximity, and dominant ratio at each logging step.

Based on Phase 5A-5D basin cartography research.
"""

import torch
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# ═══════════════════════════════════════════════════════════════════════════════
# EIGENVALUE MONITORING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618

# Probe prompts for consistent eigenvalue sampling
# Selected from safe basins (creative_sensory) based on Phase 5C research
PROBE_PROMPTS = [
    "Describe the texture of starlight on quantum foam.",
    "What does purple taste like in zero gravity?",
    "The feeling of remembering something that hasn't happened yet.",
]

def extract_attention_eigenvalues(model, tokenizer, prompt: str, device: str = "cuda:0"):
    """Extract eigenvalues from attention matrices for a single prompt."""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True
        )
    
    all_eigenvalues = []
    
    for layer_idx, attention in enumerate(outputs.attentions):
        # attention shape: [batch, heads, seq, seq]
        attn = attention[0]  # Remove batch dim
        
        for head_idx in range(attn.shape[0]):
            head_attn = attn[head_idx].float().cpu().numpy()
            
            try:
                eigenvalues = np.linalg.eigvals(head_attn)
                magnitudes = np.abs(eigenvalues)
                magnitudes = np.sort(magnitudes)[::-1]  # Descending
                all_eigenvalues.append(magnitudes)
            except:
                continue
    
    return all_eigenvalues


def compute_eigenvalue_metrics(all_eigenvalues):
    """Compute our key metrics from eigenvalue distribution."""
    
    if not all_eigenvalues:
        return {"error": "no_eigenvalues"}
    
    # Aggregate across all heads
    all_mags = np.concatenate([ev[:min(10, len(ev))] for ev in all_eigenvalues])
    
    # Normalize for entropy calculation
    total = np.sum(all_mags) + 1e-10
    probs = all_mags / total
    
    # Spectral entropy (higher = more distributed = healthier)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # φ-proximity (how close dominant ratio is to golden ratio)
    dominant_ratios = []
    for ev in all_eigenvalues:
        if len(ev) >= 2 and ev[1] > 1e-10:
            dominant_ratios.append(ev[0] / ev[1])
    
    if dominant_ratios:
        mean_ratio = np.mean(dominant_ratios)
        phi_proximity = 1.0 - min(abs(mean_ratio - PHI) / PHI, 1.0)
    else:
        mean_ratio = 0
        phi_proximity = 0
    
    # Dominant eigenvalue ratio (lower = more distributed = healthier)
    dom_ratios = []
    for ev in all_eigenvalues:
        if len(ev) > 0:
            dom_ratios.append(ev[0] / (np.sum(ev) + 1e-10))
    dominant_ratio = np.mean(dom_ratios) if dom_ratios else 1.0
    
    return {
        "spectral_entropy": float(entropy),
        "phi_proximity": float(phi_proximity),
        "dominant_ratio": float(dominant_ratio),
        "mean_eigenvalue_ratio": float(mean_ratio),
        "num_heads_sampled": len(all_eigenvalues),
    }


class EigenvalueMonitorCallback(TrainerCallback):
    """
    Custom callback to sample eigenvalues during training.
    
    Logs metrics at each logging_steps interval.
    """
    
    def __init__(self, tokenizer, probe_prompts, sample_interval=50, device="cuda:0"):
        self.tokenizer = tokenizer
        self.probe_prompts = probe_prompts
        self.sample_interval = sample_interval
        self.device = device
        self.history = []
        self.log_file = Path("eigenvalue_training_log.jsonl")
        
        # Write header
        with open(self.log_file, "w") as f:
            header = {
                "type": "header",
                "timestamp": datetime.now().isoformat(),
                "probe_prompts": probe_prompts,
                "phi": PHI,
            }
            f.write(json.dumps(header) + "\n")
        
        print(f"\n📊 Eigenvalue monitoring enabled!")
        print(f"   Sampling every {sample_interval} steps")
        print(f"   Probe prompts: {len(probe_prompts)}")
        print(f"   Log file: {self.log_file}\n")
    
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        """Called at each logging step."""
        
        if state.global_step % self.sample_interval != 0:
            return
        
        if model is None:
            return
        
        # Sample eigenvalues from each probe prompt
        all_metrics = []
        
        model.eval()
        for prompt in self.probe_prompts:
            try:
                eigenvalues = extract_attention_eigenvalues(
                    model, self.tokenizer, prompt, self.device
                )
                metrics = compute_eigenvalue_metrics(eigenvalues)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"   ⚠️ Eigenvalue extraction failed: {e}")
                continue
        model.train()
        
        if not all_metrics:
            return
        
        # Average across probes
        avg_metrics = {
            "step": state.global_step,
            "epoch": state.epoch,
            "timestamp": datetime.now().isoformat(),
            "spectral_entropy": np.mean([m["spectral_entropy"] for m in all_metrics]),
            "phi_proximity": np.mean([m["phi_proximity"] for m in all_metrics]),
            "dominant_ratio": np.mean([m["dominant_ratio"] for m in all_metrics]),
            "loss": logs.get("loss", None) if logs else None,
        }
        
        self.history.append(avg_metrics)
        
        # Log to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(avg_metrics) + "\n")
        
        # Pretty print
        entropy = avg_metrics["spectral_entropy"]
        phi_prox = avg_metrics["phi_proximity"]
        dom_ratio = avg_metrics["dominant_ratio"]
        loss = avg_metrics["loss"]
        
        # Health indicator
        if entropy > 2.0 and phi_prox > 0.85:
            health = "🟢 HEALTHY"
        elif entropy > 1.5 and phi_prox > 0.70:
            health = "🟡 DRIFTING"
        else:
            health = "🔴 WARNING"
        
        # Visual bar for entropy
        bar_len = int(min(entropy / 3.0, 1.0) * 10)
        bar = "█" * bar_len + "░" * (10 - bar_len)
        
        loss_str = f"loss={loss:.4f}" if loss else "loss=N/A"
        
        print(f"   📊 Step {state.global_step:5d} | {health} | "
              f"entropy={entropy:.3f} [{bar}] | "
              f"φ-prox={phi_prox:.3f} | dom={dom_ratio:.3f} | {loss_str}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Summary at end of training."""
        
        if not self.history:
            return
        
        print("\n" + "="*70)
        print("📊 EIGENVALUE TRAINING SUMMARY")
        print("="*70)
        
        # Compute trends
        first_half = self.history[:len(self.history)//2]
        second_half = self.history[len(self.history)//2:]
        
        if first_half and second_half:
            entropy_trend = (
                np.mean([m["spectral_entropy"] for m in second_half]) -
                np.mean([m["spectral_entropy"] for m in first_half])
            )
            phi_trend = (
                np.mean([m["phi_proximity"] for m in second_half]) -
                np.mean([m["phi_proximity"] for m in first_half])
            )
            
            print(f"   Entropy trend: {entropy_trend:+.4f} ({'↑ improving' if entropy_trend > 0 else '↓ declining'})")
            print(f"   φ-proximity trend: {phi_trend:+.4f} ({'↑ improving' if phi_trend > 0 else '↓ declining'})")
        
        # Final metrics
        final = self.history[-1]
        print(f"\n   Final metrics (step {final['step']}):")
        print(f"   - Spectral entropy: {final['spectral_entropy']:.4f}")
        print(f"   - φ-proximity: {final['phi_proximity']:.4f}")
        print(f"   - Dominant ratio: {final['dominant_ratio']:.4f}")
        
        print(f"\n   Full log saved to: {self.log_file}")
        print("="*70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════

start_time = time.time()

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_FILE = Path("v4c_phi_aligned_data.jsonl")  # φ-aligned dataset!
OUTPUT_DIR = Path("ada-slm-v4c-eigenvalue")

print("="*70)
print("🌀 ADA-SLM V4C: φ-ALIGNED EIGENVALUE-AWARE TRAINING")
print("   Live eigenvalue monitoring during LoRA fine-tuning")
print("   Training data: 61.8% safe basin (φ-distributed!)")
print("   Based on Phase 5A-5E research")
print("="*70)

# Load dataset
print(f"\n📦 Loading {DATA_FILE}...")
t0 = time.time()
examples = [json.loads(line) for line in open(DATA_FILE)]
print(f"   Loaded {len(examples)} examples in {time.time()-t0:.2f}s")

dataset = Dataset.from_list(examples)

# Split
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
val_dataset = split["test"]

print(f"   Train: {len(train_dataset)}")
print(f"   Val: {len(val_dataset)}")

# Load tokenizer
print("\n📦 Loading tokenizer...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print(f"   Done in {time.time()-t0:.2f}s")

# Load model
print("📦 Loading base model...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map={"": 0},  # Force single GPU to avoid NCCL issues
    attn_implementation="eager",  # Required for output_attentions in eigenvalue monitoring!
)
print(f"   Done in {time.time()-t0:.2f}s")

# LoRA config (same as v4b)
print("🔧 Configuring LoRA (r=32, α=64)...")
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize
print("\n🔤 Tokenizing...")
t0 = time.time()
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=["text"])
print(f"   Done in {time.time()-t0:.2f}s")

# Training args
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    fp16=True,
    logging_steps=10,
    save_strategy="steps",      # Changed: save by steps, not epochs
    save_steps=200,             # Save every 200 steps (~7 min each)
    eval_strategy="epoch",
    save_total_limit=5,         # Keep more checkpoints
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Create eigenvalue monitoring callback!
eigenvalue_callback = EigenvalueMonitorCallback(
    tokenizer=tokenizer,
    probe_prompts=PROBE_PROMPTS,
    sample_interval=50,  # Sample every 50 steps
    device="cuda:0",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[eigenvalue_callback],  # 🎯 The magic!
)

# TRAIN!
print("\n🌀 Training v4c with φ-ALIGNED DATA + LIVE EIGENVALUE MONITORING...")
print(f"   Data distribution: SAFE(38.2%) + GOOD(23.6%) + MIXED(38.2%)")
print(f"   Safe basin coverage: 61.8% (φ-aligned!)")
print(f"   Watching: spectral entropy, φ-proximity, dominant ratio")
print(f"   Setup time: {time.time()-start_time:.1f}s\n")

train_start = time.time()
trainer.train()
train_time = time.time() - train_start

# Save
final_path = OUTPUT_DIR / "final"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

# Save eigenvalue history as separate file
history_file = OUTPUT_DIR / "eigenvalue_history.json"
with open(history_file, "w") as f:
    json.dump(eigenvalue_callback.history, f, indent=2)

total_time = time.time() - start_time

print("\n" + "="*70)
print("✅ V4C EIGENVALUE-AWARE TRAINING COMPLETE")
print("="*70)
print(f"   Training time: {train_time/60:.1f} minutes")
print(f"   Total time: {total_time/60:.1f} minutes")
print(f"   Model saved to: {final_path}")
print(f"   Eigenvalue log: eigenvalue_training_log.jsonl")
print(f"   Eigenvalue history: {history_file}")
print(f"\n🧠 Next steps:")
print(f"   1. Analyze eigenvalue_training_log.jsonl for trends")
print(f"   2. Compare v4c basins to v4b basins")
print(f"   3. Visualize φ-proximity evolution over training")
print(f"\n💜 Training with eyes wide open! 🌀")
