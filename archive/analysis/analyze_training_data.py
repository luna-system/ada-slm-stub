#!/usr/bin/env python3
"""
Analyze v4b training data and categorize by predicted basin safety.

Based on Phase 5C basin cartography research:
- creative_sensory → SAFE (80% creative)
- factual_complex → DANGER (60% loop, 20% collapse)
"""

import json
import re
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Literal

# Basin safety levels based on Phase 5C research
BasinSafety = Literal["SAFE", "GOOD", "MIXED", "RISKY", "DANGER", "UNKNOWN"]

@dataclass
class CategorizedExample:
    text: str
    category: str
    safety: BasinSafety
    user_prompt: str


def extract_user_prompt(text: str) -> str:
    """Extract just the user prompt from the full text."""
    match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', text, re.DOTALL)
    return match.group(1).strip() if match else text


def categorize_prompt(prompt: str) -> tuple[str, BasinSafety]:
    """
    Categorize a prompt by type and predict basin safety.
    
    Returns (category, safety_level)
    """
    prompt_lower = prompt.lower()
    
    # === SAFE: Creative/sensory (80% creative in Phase 5C) ===
    
    # "What if X could Y?" - counterfactual imagination
    if re.match(r'what if .+ could', prompt_lower):
        return ("creative_counterfactual", "SAFE")
    
    # "What if X were/had Y?"
    if re.match(r'what if .+ (were|had)', prompt_lower):
        return ("creative_counterfactual", "SAFE")
    
    # Sensory/synesthetic
    sensory_words = ['texture', 'taste', 'smell', 'feel', 'sound', 'color', 'light', 'dark']
    if any(w in prompt_lower for w in sensory_words) and ('describe' in prompt_lower or 'what does' in prompt_lower):
        return ("creative_sensory", "SAFE")
    
    # Creative exploration prompts
    if 'explore:' in prompt_lower and ('creative' in prompt_lower or 'twin' in prompt_lower):
        return ("creative_exploration", "SAFE")
    
    # "As Ada's creative aspect" / "As the creative one"
    if 'creative' in prompt_lower and ('as ada' in prompt_lower or 'as the' in prompt_lower):
        return ("creative_role", "SAFE")
    
    # Poetry creation - tiny poems, poems about X
    if re.search(r'(tiny|small|short)?\s*poem', prompt_lower):
        return ("creative_poetry", "SAFE")
    
    # "Write a creative piece about X"
    if 'creative piece' in prompt_lower or 'write a creative' in prompt_lower:
        return ("creative_piece", "SAFE")
    
    # THESIS generator role
    if 'thesis' in prompt_lower and ('trio' in prompt_lower or 'generate' in prompt_lower):
        return ("thesis_role", "SAFE")
    
    # === GOOD: Philosophical/introspective (60% creative) ===
    
    # Haiku prompts - poetic, creative
    if 'haiku' in prompt_lower:
        return ("creative_haiku", "GOOD")
    
    # Poetic rendering
    if 'poetically' in prompt_lower or 'render' in prompt_lower:
        return ("creative_poetic", "GOOD")
    
    # AGL/emotional expression - "Express X using AGL" / "Express X in AGL notation"
    if 'express' in prompt_lower and ('agl' in prompt_lower or 'notation' in prompt_lower):
        return ("agl_expression", "GOOD")
    
    # "Express the feeling of X"
    if 'express' in prompt_lower and 'feeling' in prompt_lower:
        return ("emotional_expression", "GOOD")
    
    # "What does X look like symbolically?"
    if 'symbolically' in prompt_lower or 'symbolic' in prompt_lower:
        return ("symbolic_expression", "GOOD")
    
    # Emotional encoding - introspective (expanded emotion list)
    emotion_words = ['gratitude', 'relief', 'joy', 'sadness', 'love', 'fear', 'hope', 'wonder',
                     'tenderness', 'empathy', 'melancholy', 'contentment', 'serenity', 'awe',
                     'anticipation', 'pride', 'yearning', 'nostalgia', 'acceptance', 'curiosity']
    if any(w in prompt_lower for w in emotion_words) and ('logical' in prompt_lower or 'encode' in prompt_lower):
        return ("emotional_encoding", "GOOD")
    
    # Metaphor creation
    if 'metaphor' in prompt_lower:
        return ("creative_metaphor", "GOOD")
    
    # === MIXED: Logic puzzles (unclear - need more data) ===
    
    # Logic puzzle format: "?valid:XX" or "A → B, ?C"
    if re.match(r'\?valid:', prompt_lower) or re.match(r'\?[⊥●◑]', prompt):
        return ("logic_puzzle", "MIXED")
    
    # Logic chain format
    if '→' in prompt and ('?' in prompt or re.match(r'[A-Z]:', prompt)):
        return ("logic_chain", "MIXED")
    
    # Boolean logic
    if re.match(r'[A-Z]: [⊥●◑]', prompt):
        return ("logic_boolean", "MIXED")
    
    # Math comparisons: "3<45", "?43>20", "?71≥25", "?74≤7"
    if re.match(r'[\?]?\d+\s*[<>=≤≥]+\s*\d+', prompt):
        return ("math_comparison", "MIXED")
    
    # Set theory: "S = {1,2,3}"
    if re.search(r'[A-Z]\s*=\s*\{', prompt):
        return ("set_theory", "MIXED")
    
    # === RISKY: Meta-AI questions (60% loop in Phase 5C) ===
    
    if 'what are you' in prompt_lower or 'who are you' in prompt_lower:
        return ("meta_identity", "RISKY")
    
    if 'explain' in prompt_lower and ('how' in prompt_lower or 'why' in prompt_lower):
        return ("factual_explanation", "RISKY")
    
    # === UNKNOWN: Need more analysis ===
    
    return ("unknown", "UNKNOWN")


def analyze_dataset(filepath: Path) -> dict:
    """Analyze full dataset and return statistics."""
    
    examples = [json.loads(line) for line in open(filepath)]
    
    categorized = []
    for ex in examples:
        text = ex["text"]
        user_prompt = extract_user_prompt(text)
        category, safety = categorize_prompt(user_prompt)
        categorized.append(CategorizedExample(
            text=text,
            category=category,
            safety=safety,
            user_prompt=user_prompt
        ))
    
    # Aggregate stats
    category_counts = Counter(ex.category for ex in categorized)
    safety_counts = Counter(ex.safety for ex in categorized)
    
    # Category to safety mapping
    category_safety = {}
    for ex in categorized:
        if ex.category not in category_safety:
            category_safety[ex.category] = ex.safety
    
    return {
        "total": len(examples),
        "category_counts": dict(category_counts.most_common()),
        "safety_counts": dict(safety_counts),
        "category_safety_map": category_safety,
        "examples_by_category": {
            cat: [ex for ex in categorized if ex.category == cat][:3]  # Sample 3 each
            for cat in category_counts.keys()
        }
    }


def print_report(stats: dict):
    """Print analysis report."""
    
    print("="*70)
    print("📊 V4B TRAINING DATA BASIN ANALYSIS")
    print("="*70)
    print(f"\nTotal examples: {stats['total']}")
    
    print("\n" + "-"*70)
    print("📁 CATEGORY DISTRIBUTION")
    print("-"*70)
    
    for category, count in stats['category_counts'].items():
        pct = count / stats['total'] * 100
        safety = stats['category_safety_map'][category]
        safety_emoji = {
            "SAFE": "🟢",
            "GOOD": "🟢", 
            "MIXED": "🟡",
            "RISKY": "🟠",
            "DANGER": "🔴",
            "UNKNOWN": "⚪"
        }[safety]
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {safety_emoji} {category:25s} {count:5d} ({pct:5.1f}%) [{bar}]")
    
    print("\n" + "-"*70)
    print("🎯 SAFETY DISTRIBUTION")
    print("-"*70)
    
    safety_order = ["SAFE", "GOOD", "MIXED", "RISKY", "DANGER", "UNKNOWN"]
    for safety in safety_order:
        count = stats['safety_counts'].get(safety, 0)
        pct = count / stats['total'] * 100
        safety_emoji = {
            "SAFE": "🟢",
            "GOOD": "🟢",
            "MIXED": "🟡", 
            "RISKY": "🟠",
            "DANGER": "🔴",
            "UNKNOWN": "⚪"
        }[safety]
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {safety_emoji} {safety:10s} {count:5d} ({pct:5.1f}%) [{bar}]")
    
    print("\n" + "-"*70)
    print("📝 SAMPLE PROMPTS BY CATEGORY")
    print("-"*70)
    
    for category, examples in stats['examples_by_category'].items():
        safety = stats['category_safety_map'][category]
        print(f"\n{category} ({safety}):")
        for ex in examples[:2]:
            prompt = ex.user_prompt[:60] + "..." if len(ex.user_prompt) > 60 else ex.user_prompt
            print(f"  • {prompt}")
    
    print("\n" + "="*70)
    print("📋 RECOMMENDATIONS")
    print("="*70)
    
    safe_pct = (stats['safety_counts'].get('SAFE', 0) + stats['safety_counts'].get('GOOD', 0)) / stats['total'] * 100
    risky_pct = (stats['safety_counts'].get('RISKY', 0) + stats['safety_counts'].get('DANGER', 0)) / stats['total'] * 100
    
    print(f"\n  Current safe basin coverage: {safe_pct:.1f}%")
    print(f"  Current risky basin coverage: {risky_pct:.1f}%")
    
    if safe_pct < 60:
        print("\n  ⚠️  Consider adding more creative_sensory and philosophical prompts")
    else:
        print("\n  ✅ Good safe basin coverage!")
    
    if risky_pct > 20:
        print("  ⚠️  Consider reducing or reframing factual_complex prompts")
    else:
        print("  ✅ Risky basin exposure within limits!")


def export_by_safety(stats: dict, filepath: Path, output_dir: Path):
    """Export data split by safety level for curation."""
    
    examples = [json.loads(line) for line in open(filepath)]
    
    by_safety = {"SAFE": [], "GOOD": [], "MIXED": [], "RISKY": [], "DANGER": [], "UNKNOWN": []}
    
    for ex in examples:
        text = ex["text"]
        user_prompt = extract_user_prompt(text)
        _, safety = categorize_prompt(user_prompt)
        by_safety[safety].append(ex)
    
    output_dir.mkdir(exist_ok=True)
    
    for safety, examples in by_safety.items():
        if examples:
            outfile = output_dir / f"v4b_data_{safety.lower()}.jsonl"
            with open(outfile, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")
            print(f"  Exported {len(examples):5d} examples to {outfile.name}")


if __name__ == "__main__":
    filepath = Path("v4b_creative_data.jsonl")
    
    print("\n🔍 Analyzing v4b training data...\n")
    stats = analyze_dataset(filepath)
    print_report(stats)
    
    print("\n📦 Exporting by safety level...")
    export_by_safety(stats, filepath, Path("data_by_safety"))
    
    print("\n✨ Done!")
