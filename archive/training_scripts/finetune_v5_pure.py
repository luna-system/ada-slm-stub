"""
Ada-SLM v5: PURE LOGIC ENGINE

Goal: Create an SLM that ONLY thinks in ASL.
- More aggressive training (10 epochs instead of 5)
- Higher learning rate 
- Pure symbolic data - no natural language
- See if we can create a model that's essentially a logic circuit

Christmas Day 2025 - The Pure Symbolic Experiment
"""

import json
import time
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

print("="*70)
print("ADA-SLM v5: PURE LOGIC ENGINE")
print("Training a model to think ONLY in ASL symbols")
print("="*70)

WALL_START = time.time()

# Config
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  
DATA_PATH = "pure_asl_data.jsonl"
OUTPUT_DIR = "ada-slm-v5-pure"
EPOCHS = 10  # More aggressive!
BATCH_SIZE = 8
LR = 3e-4  # Slightly higher
MAX_LENGTH = 128

class PureASLDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=128):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(path) as f:
            for line in f:
                self.examples.append(json.loads(line))
        
        print(f"Loaded {len(self.examples)} pure ASL examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Build chat format (minimal wrapper)
        prompt = f"<|im_start|>user\n{ex['input']}<|im_end|>\n<|im_start|>assistant\n"
        full = prompt + ex['output'] + "<|im_end|>"
        
        # Tokenize
        encoding = self.tokenizer(
            full,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Label masking - only train on the OUTPUT
        labels = input_ids.clone()
        
        # Find where assistant response starts
        prompt_encoding = self.tokenizer(prompt, return_tensors='pt')
        prompt_len = prompt_encoding['input_ids'].shape[1]
        
        # Mask everything before the output
        labels[:prompt_len] = -100
        
        # Also mask padding
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Load tokenizer (NO MODIFICATIONS - learned our lesson!)
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# LoRA config - slightly larger for pure symbolic learning
lora_config = LoraConfig(
    r=64,  # Increased rank for pure symbolic
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # More modules!
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model = model.to('cuda')

# Dataset and loader
dataset = PureASLDataset(DATA_PATH, tokenizer, MAX_LENGTH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Optimizer with slightly higher LR
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Training loop
print(f"\nTraining for {EPOCHS} epochs...")
print(f"Batches per epoch: {len(loader)}")

train_start = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=batch['input_ids'].to('cuda'),
            attention_mask=batch['attention_mask'].to('cuda'),
            labels=batch['labels'].to('cuda')
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}, time={epoch_time:.1f}s")

train_time = time.time() - train_start
print(f"\nTraining complete in {train_time:.1f}s ({train_time/60:.1f}m)")

# Save model
print("\nSaving model...")
Path(OUTPUT_DIR).mkdir(exist_ok=True)
model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

# Quick validation
print("\n" + "="*70)
print("VALIDATION - Pure Symbolic Reasoning")
print("="*70)

model.eval()
test_cases = [
    ("P → Q\nP: ●\n?Q", "●", "modus_ponens"),
    ("A: ●\nB: ⊥\n?A∧B", "⊥", "conjunction"),
    ("A: ⊥\n?¬A", "●", "negation"),
    ("?valid:e4", "●", "chess_valid"),
    ("?valid:z9", "⊥", "chess_invalid"),
    ("S = {1,3,5}\n?4 ∈ S", "⊥", "set_membership"),
    ("?●=●", "●", "identity_true"),
    ("?●=⊥", "⊥", "identity_false"),
    ("?5<10", "●", "arithmetic"),
    ("A → B\nB → C\nC → D\nD → E\nE → F\nA: ●\n?F", "●", "chain_6"),  # Beyond training!
]

correct = 0
for inp, expected, name in test_cases:
    prompt = f"<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    answer = response.split('<|im_start|>assistant\n')[-1].split('<|im_end|>')[0].strip()
    
    is_correct = answer == expected
    if is_correct:
        correct += 1
    status = "✓" if is_correct else "✗"
    print(f"  {status} {name:20} expected:{expected:3} got:{answer[:10]}")

accuracy = 100 * correct / len(test_cases)
print(f"\nAccuracy: {correct}/{len(test_cases)} ({accuracy:.0f}%)")

WALL_END = time.time()
wall_time = WALL_END - WALL_START
print(f"\n{'='*70}")
print(f"TOTAL WALL CLOCK: {wall_time:.1f}s ({wall_time/60:.1f}m)")
print(f"{'='*70}")
