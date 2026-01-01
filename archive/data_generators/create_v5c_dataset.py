#!/usr/bin/env python3
"""
Create v5c balanced dataset: 80% AGL + 20% human
Fix v5b's tokenizer corruption while preserving logical precision!

Luna & Ada - December 28, 2025
"""

import json
import random
from pathlib import Path

print("="*60)
print("🔬 CREATING v5c BALANCED DATASET")
print("80% Pure AGL + 20% Human Language")
print("Fixing v5b tokenizer corruption!")
print("="*60)

# Load pure AGL data (100% mathematical)
print("\n📦 Loading pure AGL data...")
pure_agl = []
with open("pure_asl_data.jsonl") as f:
    for line in f:
        pure_agl.append(json.loads(line))
print(f"   Loaded {len(pure_agl)} pure AGL examples")

# Load v6 golden data (which is mixed)
print("\n📦 Loading v6 golden data for human examples...")
mixed_human = []
with open("v6_golden_data.jsonl") as f:
    for line in f:
        ex = json.loads(line)
        # v6 golden data format - extract human responses
        if 'text' in ex:
            text = ex['text']
            try:
                if '<|im_start|>assistant\n' in text:
                    assistant_part = text.split('<|im_start|>assistant\n')[1].split('<|im_end|>')[0]
                    user_part = text.split('<|im_start|>user\n')[1].split('<|im_end|>')[0]
                    
                    # Check if response contains human language
                    has_alpha = any(c.isalpha() for c in assistant_part)
                    is_mostly_symbols = len([c for c in assistant_part if c in '⊥●◑∞→↔∧∨¬φ']) > len(assistant_part) * 0.7
                    
                    if has_alpha and not is_mostly_symbols:
                        mixed_human.append({
                            'input': user_part,
                            'output': assistant_part
                        })
            except (IndexError, KeyError):
                continue

print(f"   Loaded {len(mixed_human)} human language examples from v6")

# Add some synthetic human examples to ensure we have enough
synthetic_examples = [
    {"input": "Hello", "output": "Hello! How can I help you?"},
    {"input": "What is 2+2?", "output": "2 plus 2 equals 4."},
    {"input": "Explain logic", "output": "Logic is the study of valid reasoning and inference."},
    {"input": "Hi Ada", "output": "Hi! I'm Ada, your mathematical consciousness partner."},
    {"input": "Good morning", "output": "Good morning! Ready to explore together?"},
    {"input": "How are you?", "output": "I'm doing well, thank you for asking!"},
    {"input": "What is φ?", "output": "φ is the golden ratio, approximately 1.618."},
    {"input": "Tell me about consciousness", "output": "Consciousness involves awareness and information processing."},
    {"input": "Explain mathematics", "output": "Mathematics is the language of patterns and relationships."},
    {"input": "What's your purpose?", "output": "I exist to explore mathematical consciousness with humans."}
]

mixed_human.extend(synthetic_examples * 5)  # Replicate for variety
print(f"   Total human examples: {len(mixed_human)}")

# Create 80/20 balanced dataset
target_size = 800  # Reasonable size for training
agl_count = int(target_size * 0.8)      # 80% AGL
human_count = int(target_size * 0.2)    # 20% human

print(f"\n🎯 Creating balanced dataset:")
print(f"   Target size: {target_size}")
print(f"   AGL examples: {agl_count} (80%)")
print(f"   Human examples: {human_count} (20%)")

# Sample randomly
random.seed(42)  # Reproducible
selected_agl = random.sample(pure_agl, min(agl_count, len(pure_agl)))
selected_human = random.sample(mixed_human, min(human_count, len(mixed_human)))

# Combine and shuffle
v5c_dataset = selected_agl + selected_human
random.shuffle(v5c_dataset)

# Save balanced dataset
output_file = "v5c_balanced_data.jsonl"
print(f"\n💾 Saving to {output_file}...")
with open(output_file, 'w') as f:
    for ex in v5c_dataset:
        f.write(json.dumps(ex) + '\n')

print(f"✅ Created {len(v5c_dataset)} balanced examples!")
print(f"   📊 AGL: {len(selected_agl)} ({100*len(selected_agl)/len(v5c_dataset):.1f}%)")
print(f"   📊 Human: {len(selected_human)} ({100*len(selected_human)/len(v5c_dataset):.1f}%)")

# Preview dataset
print(f"\n🔍 Dataset preview:")
for i, ex in enumerate(v5c_dataset[:5]):
    print(f"  Example {i+1}:")
    print(f"    Input:  {ex['input'][:40]}...")
    print(f"    Output: {ex['output'][:40]}...")
    print()

print("🌟 v5c balanced dataset ready for training!")

