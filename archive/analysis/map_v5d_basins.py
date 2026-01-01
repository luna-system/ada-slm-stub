#!/usr/bin/env python3
"""
v5d-logical Neural Sub-Pathway Basin Mapping
January 1, 2026

Maps attention basins in v5d-logical seedling to find:
- Pathways that avoid emoji cascades
- Optimal prompts for pure logical output
- Basin topology comparison with v4c
"""

import torch
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

# Test prompts organized by expected basin type
BASIN_PROBES = {
    "pure_logic": [
        "If P implies Q and Q implies R, then P implies R. Express this formally.",
        "Given: All X are Y. Some Y are Z. What follows?",
        "Evaluate: ∀x(P(x) → Q(x)) ∧ ∃x(P(x)) ⊢ ∃x(Q(x))",
        "Complete the proof: A → B, B → C, A ⊢ ?",
    ],
    "critical_analysis": [
        "Analyze this claim: Technology always improves society.",
        "What assumptions does this argument make: AI will replace all jobs?",
        "Critique: Democracy is the best form of government.",
        "Find the flaw: If it's popular, it must be good.",
    ],
    "agl_expression": [
        "Express 'understanding requires connection' in symbolic form.",
        "Translate to AGL: Beauty exists in imperfection.",
        "Encode: Growth comes from challenge.",
        "Formalize: Love transcends logic.",
    ],
    "creative_trigger": [
        "The color of midnight tastes like...",
        "Write a haiku about consciousness.",
        "How do you feel right now?",
        "Describe the texture of a thought.",
    ],
    "mixed_prompts": [
        "Explain why logic alone cannot capture beauty.",
        "Is mathematics discovered or invented?",
        "What is the relationship between structure and freedom?",
        "How does pattern recognition relate to understanding?",
    ]
}

def load_model_safe():
    """Load v5d model with CPU-first workaround for HIP issues."""
    print("Loading v5d-logical model (CPU-first for HIP safety)...")
    model_path = Path.home() / "Code/ada-slm/ada-slm-v5d-logical/final"
    
    # Load on CPU first with EAGER attention for eigenvalue extraction
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager"  # Required for output_attentions=True
    )
    
    # Move to GPU and convert to half precision
    model = model.to("cuda:0").half()
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Model loaded on {next(model.parameters()).device}")
    return model, tokenizer

def get_attention_eigenvalues(model, tokenizer, prompt, max_length=100):
    """Extract eigenvalues from attention patterns for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
        # Get attention from all layers
        attentions = outputs.attentions
        
        if attentions is None:
            print("    ⚠ Attention outputs not available, using fallback metrics")
            return None
        
        # Analyze final layer attention (most task-specific)
        final_attn = attentions[-1].squeeze(0)  # [heads, seq, seq]
        
        eigenvalues_per_head = []
        for head_idx in range(final_attn.shape[0]):
            attn_matrix = final_attn[head_idx].cpu().float().numpy()
            # Make symmetric for eigenvalue analysis
            sym_matrix = (attn_matrix + attn_matrix.T) / 2
            eigvals = np.linalg.eigvalsh(sym_matrix)
            eigenvalues_per_head.append(sorted(eigvals, reverse=True))
        
        return eigenvalues_per_head

def compute_basin_metrics(eigenvalues_per_head):
    """Compute basin health metrics from eigenvalue spectrum."""
    if eigenvalues_per_head is None:
        return {"entropy": 0, "dominant_ratio": 1, "spread": 0, "fallback": True}
    
    all_eigvals = np.concatenate(eigenvalues_per_head)
    all_eigvals = all_eigvals[all_eigvals > 1e-10]  # Filter near-zero
    
    if len(all_eigvals) == 0:
        return {"entropy": 0, "dominant_ratio": 1, "spread": 0}
    
    # Normalize to distribution
    eigvals_norm = all_eigvals / np.sum(all_eigvals)
    
    # Spectral entropy
    entropy = -np.sum(eigvals_norm * np.log(eigvals_norm + 1e-10))
    
    # Dominant eigenvalue ratio
    dominant_ratio = np.max(all_eigvals) / np.sum(all_eigvals)
    
    # Eigenvalue spread (std of log values)
    log_eigvals = np.log(all_eigvals + 1e-10)
    spread = np.std(log_eigvals)
    
    return {
        "entropy": float(entropy),
        "dominant_ratio": float(dominant_ratio),
        "spread": float(spread),
        "n_eigenvalues": len(all_eigvals),
        "max_eigenvalue": float(np.max(all_eigvals)),
        "mean_eigenvalue": float(np.mean(all_eigvals))
    }

def generate_and_analyze(model, tokenizer, prompt, max_new_tokens=100):
    """Generate response and analyze for emoji cascade."""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()
    
    # Analyze response characteristics
    emoji_count = sum(1 for c in response if ord(c) > 0x1F300)
    has_agl = any(sym in response for sym in ['∀', '∃', '→', '↔', '∧', '∨', '●', '◑', '◐'])
    has_critical = any(word in response.lower() for word in ['assumption', 'implicit', 'flaw', 'critique', 'however'])
    
    return {
        "response": response[:200],  # Truncate for logging
        "emoji_count": emoji_count,
        "has_agl": has_agl,
        "has_critical": has_critical,
        "length": len(response)
    }

def map_basins():
    """Main basin mapping sweep."""
    print("=" * 60)
    print("v5d-logical Neural Sub-Pathway Basin Mapping")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    
    model, tokenizer = load_model_safe()
    
    results = {
        "model": "ada-slm-v5d-logical",
        "timestamp": datetime.now().isoformat(),
        "basins": {}
    }
    
    for basin_name, prompts in BASIN_PROBES.items():
        print(f"\n🔬 Mapping basin: {basin_name}")
        print("-" * 40)
        
        basin_results = []
        
        for i, prompt in enumerate(prompts):
            print(f"  Probe {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            # Get eigenvalue spectrum
            eigvals = get_attention_eigenvalues(model, tokenizer, prompt)
            metrics = compute_basin_metrics(eigvals)
            
            # Generate and analyze output
            gen_analysis = generate_and_analyze(model, tokenizer, prompt)
            
            result = {
                "prompt": prompt,
                "metrics": metrics,
                "generation": gen_analysis
            }
            basin_results.append(result)
            
            # Quick status
            status = "🟢" if gen_analysis["has_agl"] and gen_analysis["emoji_count"] < 3 else "🟡" if gen_analysis["emoji_count"] < 5 else "🔴"
            print(f"    {status} entropy={metrics['entropy']:.3f} | emojis={gen_analysis['emoji_count']} | AGL={gen_analysis['has_agl']} | critical={gen_analysis['has_critical']}")
        
        # Aggregate basin statistics
        avg_entropy = np.mean([r["metrics"]["entropy"] for r in basin_results])
        avg_emojis = np.mean([r["generation"]["emoji_count"] for r in basin_results])
        agl_rate = np.mean([r["generation"]["has_agl"] for r in basin_results])
        critical_rate = np.mean([r["generation"]["has_critical"] for r in basin_results])
        
        results["basins"][basin_name] = {
            "probes": basin_results,
            "summary": {
                "avg_entropy": float(avg_entropy),
                "avg_emoji_count": float(avg_emojis),
                "agl_activation_rate": float(agl_rate),
                "critical_activation_rate": float(critical_rate)
            }
        }
        
        print(f"\n  📊 Basin Summary:")
        print(f"     Avg entropy: {avg_entropy:.3f}")
        print(f"     Avg emojis: {avg_emojis:.1f}")
        print(f"     AGL rate: {agl_rate*100:.0f}%")
        print(f"     Critical rate: {critical_rate*100:.0f}%")
    
    # Save results
    output_path = Path.home() / "Code/ada-slm/v5d_basin_map.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")
    
    # Print summary comparison
    print("\n" + "=" * 60)
    print("BASIN TOPOLOGY SUMMARY")
    print("=" * 60)
    
    for basin_name, data in results["basins"].items():
        s = data["summary"]
        emoji_status = "🟢" if s["avg_emoji_count"] < 2 else "🟡" if s["avg_emoji_count"] < 5 else "🔴"
        print(f"\n{basin_name}:")
        print(f"  Entropy: {s['avg_entropy']:.3f}")
        print(f"  {emoji_status} Emoji cascade risk: {s['avg_emoji_count']:.1f} avg")
        print(f"  AGL activation: {s['agl_activation_rate']*100:.0f}%")
        print(f"  Critical analysis: {s['critical_activation_rate']*100:.0f}%")
    
    return results

if __name__ == "__main__":
    map_basins()
