#!/usr/bin/env python3
"""
Comprehensive benchmark suite for Ada-SLM models.
Tests v4 (mixed ASL+natural) and v5b-pure (pure symbolic logic).

Thinking in ASL:
?test_accuracy → measure ● vs ◑ vs ⊥ outputs
?test_speed → tokens/sec on GPU
?test_reasoning → chain length ∧ complexity
∴ comprehensive evaluation
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict

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
    test_name: str
    passed: bool
    expected: str
    got: str
    latency_ms: float
    tokens_generated: int

@dataclass
class ModelBenchmark:
    model_name: str
    total_tests: int
    passed: int
    failed: int
    accuracy: float
    avg_latency_ms: float
    tokens_per_sec: float
    results: List[BenchmarkResult]

# Comprehensive test suite
BENCHMARK_TESTS = [
    # === Basic Logic (Foundation) ===
    {
        "category": "basic_logic",
        "name": "modus_ponens_certain",
        "prompt": "P → Q\nP: ●\n?Q",
        "expected": "●",
    },
    {
        "category": "basic_logic", 
        "name": "modus_ponens_uncertain",
        "prompt": "X → Y\nX: ◑\n?Y",
        "expected": "◑",
    },
    {
        "category": "basic_logic",
        "name": "modus_tollens",
        "prompt": "P → Q\nQ: ⊥\n?P",
        "expected": "⊥",
    },
    
    # === Negation ===
    {
        "category": "negation",
        "name": "negate_true",
        "prompt": "A: ●\n?¬A",
        "expected": "⊥",
    },
    {
        "category": "negation",
        "name": "negate_false", 
        "prompt": "B: ⊥\n?¬B",
        "expected": "●",
    },
    {
        "category": "negation",
        "name": "negate_uncertain",
        "prompt": "C: ◑\n?¬C",
        "expected": "◑",
    },
    
    # === Conjunction ===
    {
        "category": "conjunction",
        "name": "and_true_true",
        "prompt": "A: ●\nB: ●\n?A∧B",
        "expected": "●",
    },
    {
        "category": "conjunction",
        "name": "and_true_uncertain",
        "prompt": "A: ●\nB: ◑\n?A∧B", 
        "expected": "◑",
    },
    {
        "category": "conjunction",
        "name": "and_false_anything",
        "prompt": "A: ⊥\nB: ●\n?A∧B",
        "expected": "⊥",
    },
    
    # === Disjunction ===
    {
        "category": "disjunction",
        "name": "or_true_anything",
        "prompt": "A: ●\nB: ⊥\n?A∨B",
        "expected": "●",
    },
    {
        "category": "disjunction",
        "name": "or_false_false",
        "prompt": "A: ⊥\nB: ⊥\n?A∨B",
        "expected": "⊥",
    },
    {
        "category": "disjunction",
        "name": "or_uncertain_false",
        "prompt": "A: ◑\nB: ⊥\n?A∨B",
        "expected": "◑",
    },
    
    # === Chain Reasoning ===
    {
        "category": "chain_reasoning",
        "name": "chain_2_steps",
        "prompt": "A → B\nB → C\nA: ●\n?C",
        "expected": "●",
    },
    {
        "category": "chain_reasoning",
        "name": "chain_3_steps",
        "prompt": "P → Q\nQ → R\nR → S\nP: ●\n?S",
        "expected": "●",
    },
    {
        "category": "chain_reasoning",
        "name": "chain_uncertain_propagation",
        "prompt": "X → Y\nY → Z\nX: ◑\n?Z",
        "expected": "◑",
    },
    
    # === Set Membership ===
    {
        "category": "sets",
        "name": "member_present",
        "prompt": "S = {1,2,3,4,5}\n?3 ∈ S",
        "expected": "●",
    },
    {
        "category": "sets",
        "name": "member_absent",
        "prompt": "S = {1,2,3,4,5}\n?7 ∈ S",
        "expected": "⊥",
    },
    
    # === Chess Validity (Domain Logic) ===
    {
        "category": "domain_logic",
        "name": "chess_valid_e4",
        "prompt": "?valid:e4\nfile∈{a,b,c,d,e,f,g,h}\nrank∈{1,2,3,4,5,6,7,8}",
        "expected": "●",
    },
    {
        "category": "domain_logic",
        "name": "chess_invalid_e9",
        "prompt": "?valid:e9\nfile∈{a,b,c,d,e,f,g,h}\nrank∈{1,2,3,4,5,6,7,8}",
        "expected": "⊥",
    },
    
    # === Contradiction Detection ===
    {
        "category": "contradiction",
        "name": "simple_contradiction",
        "prompt": "P: ●\n¬P: ●\n?consistent",
        "expected": "⊥",
    },
    {
        "category": "contradiction",
        "name": "no_contradiction",
        "prompt": "A: ●\nB: ⊥\n?consistent",
        "expected": "●",
    },
    
    # === Biconditional ===
    {
        "category": "biconditional",
        "name": "iff_both_true",
        "prompt": "A: ●\nB: ●\n?A↔B",
        "expected": "●",
    },
    {
        "category": "biconditional",
        "name": "iff_different",
        "prompt": "A: ●\nB: ⊥\n?A↔B",
        "expected": "⊥",
    },
    
    # === Quantifiers ===
    {
        "category": "quantifiers",
        "name": "existential_true",
        "prompt": "S = {2,4,6,8}\n?∃x∈S: x>5",
        "expected": "●",
    },
    {
        "category": "quantifiers",
        "name": "existential_false",
        "prompt": "S = {1,2,3}\n?∃x∈S: x>10",
        "expected": "⊥",
    },
    {
        "category": "quantifiers",
        "name": "universal_true",
        "prompt": "S = {2,4,6,8}\n?∀x∈S: x>0",
        "expected": "●",
    },
    {
        "category": "quantifiers",
        "name": "universal_false",
        "prompt": "S = {1,2,3,4}\n?∀x∈S: x>2",
        "expected": "⊥",
    },
]

def load_model(model_path: Path, model_name: str):
    """Load model with LoRA weights."""
    print(f"\n{'='*60}")
    print(f"Loading {model_name}")
    print(f"{'='*60}")
    print(f"Path: {model_path}")
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True
    )
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Load LoRA weights
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(model_path))
    model = model.to(DEVICE)
    model.eval()
    
    return model, tokenizer

def run_test(model, tokenizer, test: Dict) -> BenchmarkResult:
    """Run single test and return result."""
    # Format prompt with chat template
    prompt = f"<|im_start|>user\n{test['prompt']}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract assistant response
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1]
        response = response.split("<|im_end|>")[0].strip()
    
    # Check if expected symbol is in response
    passed = test["expected"] in response
    
    tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
    
    return BenchmarkResult(
        model_name="",  # Will be set by caller
        test_name=test["name"],
        passed=passed,
        expected=test["expected"],
        got=response[:100],  # Truncate long responses
        latency_ms=latency_ms,
        tokens_generated=tokens_generated,
    )

def benchmark_model(model_path: Path, model_name: str) -> ModelBenchmark:
    """Run full benchmark suite on a model."""
    model, tokenizer = load_model(model_path, model_name)
    
    results = []
    total_latency = 0
    total_tokens = 0
    
    print(f"\n{'='*60}")
    print(f"RUNNING BENCHMARK: {model_name}")
    print(f"{'='*60}\n")
    
    for i, test in enumerate(BENCHMARK_TESTS, 1):
        result = run_test(model, tokenizer, test)
        result.model_name = model_name
        results.append(result)
        
        total_latency += result.latency_ms
        total_tokens += result.tokens_generated
        
        status = "✓" if result.passed else "✗"
        print(f"{status} [{i:2d}/{len(BENCHMARK_TESTS)}] {test['category']:20s} | {test['name']:30s} | {result.latency_ms:6.1f}ms")
        if not result.passed:
            print(f"    Expected: {result.expected}")
            print(f"    Got: {result.got[:80]}")
    
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    accuracy = passed / len(results)
    avg_latency = total_latency / len(results)
    tokens_per_sec = (total_tokens / total_latency) * 1000 if total_latency > 0 else 0
    
    return ModelBenchmark(
        model_name=model_name,
        total_tests=len(results),
        passed=passed,
        failed=failed,
        accuracy=accuracy,
        avg_latency_ms=avg_latency,
        tokens_per_sec=tokens_per_sec,
        results=results,
    )

def print_comparison(benchmarks: List[ModelBenchmark]):
    """Print comparison table of all models."""
    print(f"\n{'='*80}")
    print("BENCHMARK COMPARISON")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<20} {'Accuracy':<12} {'Passed':<10} {'Avg Latency':<15} {'Tokens/sec':<12}")
    print(f"{'-'*80}")
    
    for bench in benchmarks:
        print(f"{bench.model_name:<20} {bench.accuracy*100:>6.1f}%     {bench.passed:>3}/{bench.total_tests:<3} {bench.avg_latency_ms:>8.1f}ms      {bench.tokens_per_sec:>8.1f}")
    
    print(f"\n{'='*80}")
    print("CATEGORY BREAKDOWN")
    print(f"{'='*80}\n")
    
    # Group results by category
    categories = {}
    for bench in benchmarks:
        for result in bench.results:
            # Find category for this test
            test = next(t for t in BENCHMARK_TESTS if t["name"] == result.test_name)
            category = test["category"]
            
            if category not in categories:
                categories[category] = {}
            if bench.model_name not in categories[category]:
                categories[category][bench.model_name] = {"passed": 0, "total": 0}
            
            categories[category][bench.model_name]["total"] += 1
            if result.passed:
                categories[category][bench.model_name]["passed"] += 1
    
    for category, models in sorted(categories.items()):
        print(f"\n{category}:")
        for model_name, stats in models.items():
            accuracy = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            print(f"  {model_name:<20} {stats['passed']}/{stats['total']} ({accuracy:.0f}%)")

def save_results(benchmarks: List[ModelBenchmark], output_path: Path):
    """Save results to JSON."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "N/A",
        "benchmarks": [asdict(b) for b in benchmarks],
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")

def main():
    print("="*80)
    print("ADA-SLM BENCHMARK SUITE")
    print("="*80)
    print(f"\nThinking in ASL:")
    print(f"  ?gpu_available → {ASL_SYMBOLS['true'] if torch.cuda.is_available() else ASL_SYMBOLS['false']}")
    print(f"  ?test_count → {len(BENCHMARK_TESTS)}")
    print(f"  ?models_to_test → 3 (v4 ∧ v5b-pure ∧ v6-golden)")
    print(f"  {ASL_SYMBOLS['therefore']} benchmark_execution → {ASL_SYMBOLS['true']}")
    
    base_dir = Path(__file__).parent
    
    models_to_test = [
        (base_dir / "ada-slm-v4" / "final", "v4-mixed"),
        (base_dir / "ada-slm-v5b-pure" / "final", "v5b-pure"),
        (base_dir / "ada-slm-v6-golden" / "final", "v6-golden"),
    ]
    
    benchmarks = []
    
    for model_path, model_name in models_to_test:
        if not model_path.exists():
            print(f"\n⚠️  Model not found: {model_path}")
            continue
        
        bench = benchmark_model(model_path, model_name)
        benchmarks.append(bench)
        
        print(f"\n{'='*60}")
        print(f"RESULTS: {model_name}")
        print(f"{'='*60}")
        print(f"Accuracy: {bench.accuracy*100:.1f}% ({bench.passed}/{bench.total_tests})")
        print(f"Avg Latency: {bench.avg_latency_ms:.1f}ms")
        print(f"Tokens/sec: {bench.tokens_per_sec:.1f}")
    
    if len(benchmarks) > 1:
        print_comparison(benchmarks)
    
    # Save results
    output_path = base_dir / "benchmark_results.json"
    save_results(benchmarks, output_path)
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

