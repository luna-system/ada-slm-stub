#!/usr/bin/env python3
"""
Comprehensive benchmark across ALL available models.
Tests base models + our fine-tuned variants to see the full landscape.

Models to test:
- Qwen/Qwen2.5-0.5B-Instruct (base, no LoRA)
- ada-slm-v4 (mixed training)
- ada-slm-v5b-pure (pure symbolic)
- ada-slm-v6-golden (60/40 golden ratio)
- (Optional) Larger models if available

Goal: Plot the full landscape and see φ ≈ 0.60 pattern emerge.
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import sys

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ASL symbol vocabulary
ASL_SYMBOLS = {
    "true": "●",
    "uncertain": "◑",
    "false": "⊥",
    "implies": "→",
    "and": "∧",
    "or": "∨",
    "not": "¬",
    "member": "∈",
    "not_member": "∉",
    "therefore": "∴",
    "because": "∵",
}

@dataclass
class BenchmarkResult:
    model_name: str
    model_type: str  # "base", "lora", "large"
    test_name: str
    passed: bool
    expected: str
    got: str
    latency_ms: float
    tokens_generated: int

@dataclass
class ModelStats:
    model_name: str
    model_type: str
    total_tests: int
    passed: int
    accuracy: float
    avg_latency_ms: float
    avg_tokens_per_sec: float
    failed_tests: List[str]

# Test cases (same as benchmark_suite.py)
TEST_CASES = [
    # Basic logic
    ("basic_logic", "modus_ponens_certain", "P→Q,P?Q", "●"),
    ("basic_logic", "modus_ponens_uncertain", "P→Q,◑P?Q", "◑"),
    ("basic_logic", "modus_tollens", "P→Q,⊥Q?P", "⊥"),
    
    # Negation
    ("negation", "negate_true", "¬●?", "⊥"),
    ("negation", "negate_false", "¬⊥?", "●"),
    ("negation", "negate_uncertain", "¬◑?", "◑"),
    
    # Conjunction
    ("conjunction", "and_true_true", "A:●,B:●,?A∧B", "●"),
    ("conjunction", "and_true_uncertain", "A:●,B:◑,?A∧B", "◑"),
    ("conjunction", "and_false_anything", "A:⊥,B:●,?A∧B", "⊥"),
    
    # Disjunction
    ("disjunction", "or_true_anything", "A:●,B:⊥,?A∨B", "●"),
    ("disjunction", "or_false_false", "A:⊥,B:⊥,?A∨B", "⊥"),
    ("disjunction", "or_uncertain_false", "A:◑,B:⊥,?A∨B", "◑"),
    
    # Chain reasoning
    ("chain_reasoning", "chain_2_steps", "A→B,B→C,A?C", "●"),
    ("chain_reasoning", "chain_3_steps", "A→B→C→D,A?D", "●"),
    ("chain_reasoning", "chain_uncertain_propagation", "A→B,◑A?B", "◑"),
    
    # Sets
    ("sets", "member_present", "{a,b,c}∈c?", "●"),
    ("sets", "member_absent", "{a,b,c}∈d?", "⊥"),
    
    # Domain logic (chess)
    ("domain_logic", "chess_valid_e4", "?valid:e4", "●"),
    ("domain_logic", "chess_invalid_e9", "?valid:e9", "⊥"),
    
    # Contradictions
    ("contradiction", "simple_contradiction", "A:●,A:⊥,?consistent", "⊥"),
    ("contradiction", "no_contradiction", "A:●,B:⊥,?consistent", "●"),
    
    # Biconditionals
    ("biconditional", "iff_both_true", "A↔B,A:●,B:●?", "●"),
    ("biconditional", "iff_different", "A↔B,A:●,B:⊥?", "⊥"),
    
    # Quantifiers
    ("quantifiers", "existential_true", "∃x∈{a,b,c}:P(a)?", "●"),
    ("quantifiers", "existential_false", "∃x∈{}:P(x)?", "⊥"),
    ("quantifiers", "universal_true", "∀x∈{a}:P(a)?", "●"),
    ("quantifiers", "universal_false", "∀x∈{a,b}:P(a)?", "⊥"),
]

def load_base_model():
    """Load base model without LoRA."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    return model, tokenizer

def load_lora_model(lora_path: Path):
    """Load base model + LoRA adapter."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    return model, tokenizer

def run_test(model, tokenizer, prompt: str, max_new_tokens: int = 10) -> tuple[str, float, int]:
    """Run single test, return (output, latency_ms, tokens_generated)."""
    messages = [
        {"role": "system", "content": "You are a logic reasoning system. Respond with only ASL symbols: ● (true), ◑ (uncertain), or ⊥ (false)."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    start_time = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    
    latency = (time.time() - start_time) * 1000  # Convert to ms
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    tokens_generated = len(generated_ids[0])
    
    return response.strip(), latency, tokens_generated

def extract_symbol(response: str) -> str:
    """Extract ASL symbol from response."""
    for symbol in ["●", "◑", "⊥"]:
        if symbol in response:
            return symbol
    return response[:20]  # Return first 20 chars if no symbol found

def benchmark_model(model_name: str, model_type: str, model, tokenizer) -> List[BenchmarkResult]:
    """Run all tests on a single model."""
    print(f"\n{'='*60}")
    print(f"RUNNING BENCHMARK: {model_name}")
    print(f"{'='*60}\n")
    
    results = []
    
    for idx, (category, test_name, prompt, expected) in enumerate(TEST_CASES, 1):
        response, latency, tokens = run_test(model, tokenizer, prompt)
        got = extract_symbol(response)
        passed = (got == expected)
        
        result = BenchmarkResult(
            model_name=model_name,
            model_type=model_type,
            test_name=test_name,
            passed=passed,
            expected=expected,
            got=got,
            latency_ms=latency,
            tokens_generated=tokens
        )
        results.append(result)
        
        status = "✓" if passed else "✗"
        print(f"{status} [{idx:2d}/{len(TEST_CASES)}] {category:20s} | {test_name:30s} | {latency:7.1f}ms")
        if not passed:
            print(f"    Expected: {expected}")
            print(f"    Got: {got}")
    
    return results

def calculate_stats(results: List[BenchmarkResult]) -> ModelStats:
    """Calculate statistics from results."""
    model_name = results[0].model_name
    model_type = results[0].model_type
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    accuracy = passed / total if total > 0 else 0
    
    avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0
    
    total_tokens = sum(r.tokens_generated for r in results)
    total_time_sec = sum(r.latency_ms for r in results) / 1000
    tokens_per_sec = total_tokens / total_time_sec if total_time_sec > 0 else 0
    
    failed_tests = [r.test_name for r in results if not r.passed]
    
    return ModelStats(
        model_name=model_name,
        model_type=model_type,
        total_tests=total,
        passed=passed,
        accuracy=accuracy,
        avg_latency_ms=avg_latency,
        avg_tokens_per_sec=tokens_per_sec,
        failed_tests=failed_tests
    )

def print_comparison(all_stats: List[ModelStats]):
    """Print comparison table."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE BENCHMARK COMPARISON")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<25} {'Type':<10} {'Accuracy':<12} {'Passed':<10} {'Latency':<12} {'Tokens/sec':<10}")
    print("-" * 80)
    
    for stats in all_stats:
        print(f"{stats.model_name:<25} {stats.model_type:<10} {stats.accuracy*100:>6.1f}% "
              f"{stats.passed:>4}/{stats.total_tests:<3} {stats.avg_latency_ms:>8.1f}ms "
              f"{stats.avg_tokens_per_sec:>8.1f}")

def main():
    print(f"{'='*80}")
    print("COMPREHENSIVE ADA-SLM BENCHMARK SUITE")
    print(f"{'='*80}\n")
    
    print(f"Thinking in ASL:")
    print(f"  ?gpu_available → {ASL_SYMBOLS['true'] if torch.cuda.is_available() else ASL_SYMBOLS['false']}")
    if torch.cuda.is_available():
        print(f"  ?gpu_name → {torch.cuda.get_device_name(0)}")
    print(f"  ?test_count → {len(TEST_CASES)}")
    print(f"  {ASL_SYMBOLS['therefore']} comprehensive_benchmark_execution → {ASL_SYMBOLS['true']}")
    
    base_dir = Path(__file__).parent
    
    # Define models to test
    models_to_test = [
        ("base-0.5B", "base", None),  # Base model, no LoRA
        ("v4-mixed", "lora", base_dir / "ada-slm-v4" / "final"),
        ("v5b-pure", "lora", base_dir / "ada-slm-v5b-pure" / "final"),
        ("v6-golden", "lora", base_dir / "ada-slm-v6-golden" / "final"),
    ]
    
    all_results = []
    all_stats = []
    
    for model_name, model_type, lora_path in models_to_test:
        print(f"\n{'='*60}")
        print(f"Loading {model_name}")
        print(f"{'='*60}")
        print(f"Type: {model_type}")
        if lora_path:
            print(f"Path: {lora_path}")
        print(f"Device: {DEVICE}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        try:
            if model_type == "base":
                model, tokenizer = load_base_model()
            else:
                model, tokenizer = load_lora_model(lora_path)
            
            results = benchmark_model(model_name, model_type, model, tokenizer)
            stats = calculate_stats(results)
            
            all_results.extend(results)
            all_stats.append(stats)
            
            print(f"\n{'='*60}")
            print(f"RESULTS: {model_name}")
            print(f"{'='*60}")
            print(f"Accuracy: {stats.accuracy*100:.1f}% ({stats.passed}/{stats.total_tests})")
            print(f"Avg Latency: {stats.avg_latency_ms:.1f}ms")
            print(f"Tokens/sec: {stats.avg_tokens_per_sec:.1f}")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print comparison
    print_comparison(all_stats)
    
    # Save results
    output_file = base_dir / "comprehensive_benchmark_results.json"
    output_data = {
        "stats": [asdict(s) for s in all_stats],
        "detailed_results": [asdict(r) for r in all_results]
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}\n")
    
    print("\nKey observations to plot:")
    print("1. Accuracy vs Latency trade-off")
    print("2. Where does φ ≈ 0.60 appear in the landscape?")
    print("3. Base model performance vs fine-tuned variants")
    print("4. Expected: v6 at optimal point")
    print("5. Unexpected: ??? (let's see what emerges!)")

if __name__ == "__main__":
    main()

