#!/usr/bin/env python3
"""Fine-tune Qwen2.5-0.5B-INSTRUCT on ASL symbolic reasoning.

Ada-SLM: a tiny model specialized in symbolic reasoning.
KEY FIX: Use INSTRUCT model as base, not raw model!
"""

import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import os
import time

os.environ["AMD_DEBUG_NO_LIBDRM"] = "1"

# === CONFIG ===
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # INSTRUCT version!
OUTPUT_DIR = Path(__file__).parent / "ada-slm-v2"
DATA_PATH = Path(__file__).parent / "asl_training_data.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training hyperparams
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 5
MAX_LENGTH = 128

# ASL special tokens
ASL_TOKENS = ["●", "◑", "⊥", "→", "←", "⟷", "∧", "∨", "¬", "∈", "∉", "∴", "∵"]


class ASLDataset(Dataset):
    def __init__(self, data_path: Path, tokenizer, max_length: int = 128):
        self.examples = []
        with open(data_path) as f:
            for line in f:
                item = json.loads(line)
                if "text" in item:
                    self.examples.append(item["text"])
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def main():
    total_start = time.time()
    
    print("=" * 60)
    print("ADA-SLM v2 (using INSTRUCT base!)")
    print("=" * 60)
    
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load tokenizer
    t0 = time.time()
    print(f"\nLoading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": ASL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  ✓ Tokenizer ready, added {num_added} tokens ({format_time(time.time() - t0)})")
    
    # Load model
    t0 = time.time()
    print(f"\nLoading model from {BASE_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(DEVICE)
    
    model.resize_token_embeddings(len(tokenizer))
    params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model loaded: {params/1e6:.1f}M params ({format_time(time.time() - t0)})")
    
    # Configure LoRA
    t0 = time.time()
    print("\nConfiguring LoRA (r=32)...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"  ✓ LoRA configured ({format_time(time.time() - t0)})")
    
    # Load data
    t0 = time.time()
    print("\nLoading dataset...")
    dataset = ASLDataset(DATA_PATH, tokenizer, MAX_LENGTH)
    print(f"  ✓ Loaded {len(dataset)} examples ({format_time(time.time() - t0)})")
    
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print(f"Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"Steps per epoch: {len(train_loader)}")
    print("=" * 60)
    
    training_start = time.time()
    model.train()
    global_step = 0
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        print(f"\n{'─'*40}")
        print(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'─'*40}")
        
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            if global_step % 50 == 0:
                elapsed = time.time() - training_start
                steps_per_sec = global_step / elapsed
                print(f"  Step {global_step}: loss={loss.item():.4f} ({steps_per_sec:.1f} steps/s)")
        
        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                eval_loss += outputs.loss.item()
        
        avg_eval_loss = eval_loss / len(eval_loader)
        print(f"\n  Epoch {epoch + 1} complete in {format_time(epoch_time)}")
        print(f"  Train loss: {avg_loss:.4f}")
        print(f"  Eval loss:  {avg_eval_loss:.4f}")
        model.train()
    
    training_time = time.time() - training_start
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Total training time: {format_time(training_time)}")
    print(f"{'='*60}")
    
    t0 = time.time()
    print("\nSaving model...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR / "final")
    tokenizer.save_pretrained(OUTPUT_DIR / "final")
    print(f"  ✓ Saved to {OUTPUT_DIR / 'final'} ({format_time(time.time() - t0)})")
    
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"TOTAL WALL CLOCK: {format_time(total_time)}")
    print(f"{'='*60}")
    print("\nDone! 🌀")


if __name__ == "__main__":
    main()
