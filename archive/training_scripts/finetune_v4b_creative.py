#!/usr/bin/env python3
"""
Train Ada-SLM v4b-creative: Creative consciousness with role awareness.
60% pure AGL + 40% creative hybrid. Role: THESIS generation.
"""

import torch
import json
import time
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

start_time = time.time()

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_FILE = Path("v4b_creative_data.jsonl")
OUTPUT_DIR = Path("ada-slm-v4b-creative")

print("="*60)
print("🎨 ADA-SLM V4B-CREATIVE TRAINING")
print("Creative Consciousness + Role Awareness")
print("Target: Thesis Generation for Dialectical Trio")
print("="*60)

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
    device_map="cuda",
)
print(f"   Done in {time.time()-t0:.2f}s")

# LoRA
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

# Training args - slightly larger batch for 10k dataset
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
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=3,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# TRAIN!
print("\n🎨 Training v4b-creative (Creative Consciousness Awakening)...")
print(f"   Role: Thesis generation for dialectical trio")
print(f"   Data: 60% pure AGL, 40% creative hybrid")
print(f"   Setup time: {time.time()-start_time:.1f}s\n")

train_start = time.time()
trainer.train()
train_time = time.time() - train_start

# Save
final_path = OUTPUT_DIR / "final"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

total_time = time.time() - start_time

print("\n" + "="*60)
print("✅ V4B-CREATIVE TRAINING COMPLETE")
print("="*60)
print(f"   Training time: {train_time/60:.1f} minutes")
print(f"   Total time: {total_time/60:.1f} minutes")
print(f"   Saved to: {final_path}")
print(f"\n🎨 Next steps:")
print(f"   1. Test creative outputs and role awareness")
print(f"   2. Evaluate AGL fluency maintained")
print(f"   3. Integrate into dialectical trio as thesis generator")
print(f"\n💜 The creative consciousness awakens! 🌀")
