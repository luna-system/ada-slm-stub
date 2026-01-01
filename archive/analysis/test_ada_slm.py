#!/usr/bin/env python3
"""Test the fine-tuned Ada-SLM model."""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

os.environ["PYTHONNOUSERSITE"] = "1"

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
FINETUNED_PATH = Path(__file__).parent / "ada-slm-v0" / "final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ASL special tokens (must match training)
ASL_TOKENS = ["●", "◑", "⊥", "→", "←", "⟷", "∧", "∨", "¬", "∈", "∉", "∴", "∵"]

TEST_PROMPTS = [
    {
        "name": "modus_ponens",
        "prompt": "P → Q\nP: ●\n?Q",
        "expected": "●",
    },
    {
        "name": "uncertainty",
        "prompt": "A: ◑\nA → B\n?B",
        "expected": "◑",
    },
    {
        "name": "chess_valid",
        "prompt": "?move:e4\nfile∈{a,b,c,d,e,f,g,h}\nrank∈{1,2,3,4,5,6,7,8}",
        "expected": "●valid",
    },
    {
        "name": "chess_invalid",
        "prompt": "?move:e9\nfile∈{a,b,c,d,e,f,g,h}\nrank∈{1,2,3,4,5,6,7,8}",
        "expected": "⊥invalid",
    },
    {
        "name": "contradiction",
        "prompt": "P: ●\n¬P: ●\n?consistent",
        "expected": "⊥",
    },
]


def main():
    print("=" * 60)
    print("TESTING ADA-SLM")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(FINETUNED_PATH), trust_remote_code=True)
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Load LoRA weights
    print("Loading fine-tuned LoRA weights...")
    model = PeftModel.from_pretrained(base_model, str(FINETUNED_PATH))
    model = model.to(DEVICE)
    model.eval()
    
    print("\n" + "=" * 60)
    print("TESTING")
    print("=" * 60)
    
    correct = 0
    total = len(TEST_PROMPTS)
    
    for test in TEST_PROMPTS:
        # Format as chat
        prompt = f"<|im_start|>user\n{test['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Extract just the assistant's response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
            response = response.split("<|im_end|>")[0].strip()
        
        got_it = test["expected"] in response
        if got_it:
            correct += 1
        
        status = "✓" if got_it else "✗"
        print(f"\n{status} {test['name']}")
        print(f"  Input: {test['prompt'][:50]}...")
        print(f"  Expected: {test['expected']}")
        print(f"  Got: {response[:100]}")
    
    print("\n" + "=" * 60)
    print(f"SCORE: {correct}/{total} ({100*correct/total:.0f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
