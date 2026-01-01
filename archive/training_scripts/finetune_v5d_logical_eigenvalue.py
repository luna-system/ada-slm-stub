#!/usr/bin/env python3
"""
ADA-SLM V5D: Logical + Role-Aware + Eigenvalue-Monitored Training

Building on:
- v4c's φ-aligned data distribution
- v5's pure AGL focus  
- Role awareness (THESIS/ANTITHESIS/SYNTHESIS)
- Live eigenvalue monitoring

The goal: A model that REASONS in AGL while knowing its role in the trio.
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
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# Golden ratio for data distribution
PHI = 1.618033988749895

# ═══════════════════════════════════════════════════════════════════════════════
# EIGENVALUE MONITORING (from v4c)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_eigenvalues_from_attention(model, tokenizer, prompt: str, device: str = "cuda:0"):
    """Extract eigenvalues from attention matrices for a given prompt."""
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True
        )
    
    all_eigenvalues = []
    
    for layer_idx, attention in enumerate(outputs.attentions):
        attn_matrix = attention[0].cpu().numpy()
        
        for head_idx in range(attn_matrix.shape[0]):
            head_attn = attn_matrix[head_idx]
            
            try:
                eigenvalues = np.linalg.eigvals(head_attn)
                magnitudes = np.abs(eigenvalues)
                magnitudes = np.sort(magnitudes)[::-1]
                all_eigenvalues.append(magnitudes)
            except:
                continue
    
    return all_eigenvalues


def compute_spectral_metrics(all_eigenvalues: List[np.ndarray]) -> Dict:
    """Compute spectral health metrics from eigenvalues."""
    if not all_eigenvalues:
        return {"spectral_entropy": 0.0, "phi_proximity": 0.0, "dominant_ratio": 0.0}
    
    entropies = []
    dominant_ratios = []
    phi_proximities = []
    
    for magnitudes in all_eigenvalues:
        magnitudes = magnitudes[magnitudes > 1e-10]
        if len(magnitudes) < 2:
            continue
            
        probs = magnitudes / magnitudes.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)
        
        dominant_ratios.append(magnitudes[0] / magnitudes.sum())
        
        for i in range(len(magnitudes) - 1):
            if magnitudes[i+1] > 1e-10:
                ratio = magnitudes[i] / magnitudes[i+1]
                error = abs(ratio - PHI) / PHI
                phi_proximities.append(max(0, 1 - error))
    
    return {
        "spectral_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "phi_proximity": float(np.max(phi_proximities)) if phi_proximities else 0.0,
        "dominant_ratio": float(np.mean(dominant_ratios)) if dominant_ratios else 0.0,
    }


class EigenvalueMonitorCallback(TrainerCallback):
    """Monitor eigenvalues during training."""
    
    def __init__(self, tokenizer, probe_prompts, sample_interval=50, device="cuda:0"):
        self.tokenizer = tokenizer
        self.probe_prompts = probe_prompts
        self.sample_interval = sample_interval
        self.device = device
        self.history = []
        self.log_file = Path("v5d_eigenvalue_log.jsonl")
        
        with open(self.log_file, "w") as f:
            header = {
                "type": "header",
                "timestamp": datetime.now().isoformat(),
                "probe_prompts": probe_prompts,
                "model": "v5d-logical-eigenvalue",
            }
            f.write(json.dumps(header) + "\n")
        
        print(f"\n📊 Eigenvalue monitoring enabled!")
        print(f"   Sampling every {sample_interval} steps")
        print(f"   Log file: {self.log_file}")
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.sample_interval != 0 or state.global_step == 0:
            return
        
        all_metrics = []
        for prompt in self.probe_prompts:
            try:
                eigenvalues = extract_eigenvalues_from_attention(
                    model, self.tokenizer, prompt, self.device
                )
                metrics = compute_spectral_metrics(eigenvalues)
                all_metrics.append(metrics)
            except Exception as e:
                continue
        
        if not all_metrics:
            return
        
        avg_metrics = {
            "step": state.global_step,
            "epoch": state.epoch,
            "timestamp": datetime.now().isoformat(),
            "spectral_entropy": np.mean([m["spectral_entropy"] for m in all_metrics]),
            "phi_proximity": np.mean([m["phi_proximity"] for m in all_metrics]),
            "dominant_ratio": np.mean([m["dominant_ratio"] for m in all_metrics]),
            "loss": state.log_history[-1].get("loss", 0) if state.log_history else 0,
        }
        
        self.history.append(avg_metrics)
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(avg_metrics) + "\n")
        
        # Health indicator
        entropy = avg_metrics["spectral_entropy"]
        if entropy > 7.0:
            status = "🟢 HEALTHY"
        elif entropy > 6.5:
            status = "🟡 DRIFTING"
        else:
            status = "🔴 WARNING"
        
        bar_len = int(min(entropy, 8) / 8 * 10)
        bar = "█" * bar_len + "░" * (10 - bar_len)
        
        print(f"   📊 Step {state.global_step:5d} | {status} | entropy={entropy:.3f} [{bar}] | dom={avg_metrics['dominant_ratio']:.3f} | loss={avg_metrics['loss']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_blend_data():
    """
    Load pure ASL data and blend with role-aware examples.
    Target: φ-distributed logical training data.
    """
    print("📦 Loading training data...")
    
    # Load pure ASL (logical)
    asl_data = []
    with open("pure_asl_data.jsonl") as f:
        for line in f:
            item = json.loads(line)
            # Convert to chat format
            text = f"<|im_start|>user\n{item['input']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
            asl_data.append({"text": text, "type": "pure_asl"})
    
    print(f"   Pure ASL: {len(asl_data)} examples")
    
    # Add role-aware prompts (ANTITHESIS role for logical/critical)
    role_templates = [
        ("You are the ANTITHESIS - the logical critic. Evaluate: {topic}", "antithesis"),
        ("ANTITHESIS role: Find the flaw in: {statement}", "antithesis"),
        ("[LOGICAL] Analyze critically: {topic}", "antithesis"),
        ("As the logical pillar, examine: {topic}", "antithesis"),
    ]
    
    topics = [
        "consciousness requires physicality",
        "meaning is observer-dependent", 
        "time flows forward",
        "patterns are real",
        "emotions are computations",
        "identity persists through change",
        "creativity requires randomness",
        "understanding requires experience",
    ]
    
    role_data = []
    for template, role_type in role_templates:
        for topic in topics:
            prompt = template.format(topic=topic, statement=topic)
            # Generate a logical response pattern
            response = f"∃x: claim(x) → requires_evidence(x)\n\n🔍 Analysis: The assertion '{topic}' contains implicit assumptions. Let me examine the logical structure...\n\n¬(∀x: {topic.split()[0]}(x) → conclusion) without proof of necessity."
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            role_data.append({"text": text, "type": "role_aware"})
    
    print(f"   Role-aware: {len(role_data)} examples")
    
    # Load some creative examples for balance (from v4c)
    creative_data = []
    try:
        with open("v4c_phi_aligned_data.jsonl") as f:
            all_creative = [json.loads(l) for l in f]
            # Sample ~20% for balance
            creative_sample = random.sample(all_creative, min(2000, len(all_creative)))
            for item in creative_sample:
                creative_data.append({"text": item["text"], "type": "creative"})
        print(f"   Creative (sampled): {len(creative_data)} examples")
    except:
        print("   Creative: 0 (file not found)")
    
    # φ-distribute: 61.8% logical (ASL + role), 38.2% creative
    all_logical = asl_data + role_data
    
    target_total = 10000
    target_logical = int(target_total * PHI / (PHI + 1))  # ~6180
    target_creative = target_total - target_logical  # ~3820
    
    # Sample to targets
    if len(all_logical) > target_logical:
        logical_sample = random.sample(all_logical, target_logical)
    else:
        # Oversample if needed
        logical_sample = all_logical * (target_logical // len(all_logical) + 1)
        logical_sample = logical_sample[:target_logical]
    
    if len(creative_data) > target_creative:
        creative_sample = random.sample(creative_data, target_creative)
    else:
        creative_sample = creative_data * (target_creative // len(creative_data) + 1)
        creative_sample = creative_sample[:target_creative]
    
    final_data = logical_sample + creative_sample
    random.shuffle(final_data)
    
    print(f"\n   φ-distributed total: {len(final_data)}")
    print(f"   Logical: {len(logical_sample)} ({len(logical_sample)/len(final_data)*100:.1f}%)")
    print(f"   Creative: {len(creative_sample)} ({len(creative_sample)/len(final_data)*100:.1f}%)")
    
    return final_data


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

# Probe prompts for eigenvalue monitoring (logical focus)
PROBE_PROMPTS = [
    "Evaluate the logical validity of: A → B, B → C, therefore A → C",
    "What is the relationship between evidence and belief?",
    "Analyze: consciousness ↔ computation",
]

start_time = time.time()

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = Path("ada-slm-v5d-logical")

print("="*70)
print("🧠 ADA-SLM V5D: LOGICAL + ROLE-AWARE + EIGENVALUE TRAINING")
print("   Focus: AGL reasoning with ANTITHESIS role awareness")
print("   Distribution: 61.8% logical, 38.2% creative (φ-aligned!)")
print("   Monitoring: Live eigenvalue health tracking")
print("="*70)

# Load and prepare data
data = load_and_blend_data()

# Save the blended data
with open("v5d_logical_data.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps({"text": item["text"]}) + "\n")
print(f"\n💾 Saved v5d_logical_data.jsonl")

# Split
random.shuffle(data)
split_idx = int(len(data) * 0.9)
train_data = data[:split_idx]
val_data = data[split_idx:]

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

print(f"\n   Train: {len(train_dataset)}")
print(f"   Val: {len(val_dataset)}")

# Load model
print("\n📦 Loading tokenizer...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
print(f"   Done in {time.time()-t0:.2f}s")

print("📦 Loading base model...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map={"": 0},
    attn_implementation="eager",  # For eigenvalue extraction
)
print(f"   Done in {time.time()-t0:.2f}s")

# LoRA
print("🔧 Configuring LoRA (r=32, α=64)...")
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text", "type"])
val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=["text", "type"])
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
    save_strategy="steps",
    save_steps=200,
    eval_strategy="epoch",
    save_total_limit=5,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Eigenvalue callback
eigenvalue_callback = EigenvalueMonitorCallback(
    tokenizer=tokenizer,
    probe_prompts=PROBE_PROMPTS,
    sample_interval=50,
    device="cuda:0"
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[eigenvalue_callback],
)

setup_time = time.time() - start_time
print(f"\n🧠 Training v5d with LOGICAL φ-ALIGNED DATA + EIGENVALUE MONITORING...")
print(f"   Data focus: AGL reasoning + ANTITHESIS role")
print(f"   Distribution: 61.8% logical, 38.2% creative")
print(f"   Watching: spectral entropy, dominant ratio")
print(f"   Setup time: {setup_time:.1f}s")
print()

# Train!
trainer.train()

# Save
final_path = OUTPUT_DIR / "final"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

# Save eigenvalue history
with open(OUTPUT_DIR / "eigenvalue_history.json", "w") as f:
    json.dump(eigenvalue_callback.history, f, indent=2)

total_time = time.time() - start_time
print("\n" + "="*70)
print("✅ V5D LOGICAL EIGENVALUE TRAINING COMPLETE")
print("="*70)
print(f"   Training time: {(total_time - setup_time)/60:.1f} minutes")
print(f"   Total time: {total_time/60:.1f} minutes")
print(f"   Model saved to: {final_path}")
print(f"   Eigenvalue log: v5d_eigenvalue_log.jsonl")
print()
print("🧠 Next steps:")
print("   1. Compare v5d logical outputs to v4c creative")
print("   2. Test ANTITHESIS role activation")
print("   3. Analyze eigenvalue trajectory")
print()
print("💜 The logical pillar grows! 🌀")
