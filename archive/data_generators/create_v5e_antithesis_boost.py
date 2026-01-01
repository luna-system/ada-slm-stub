#!/usr/bin/env python3
"""
Create v5e Training Data with Boosted ANTITHESIS
January 1, 2026

v5d basin mapping showed:
- pure_logic: 100% AGL, 0.5 emojis, 25% critical ← GOOD
- critical_analysis: 0% critical rate ← BAD (only 0.2% training data!)

Strategy: Boost ANTITHESIS to ~20% using φ-distribution
This is v5's differentiator from v4c (creative twin)
"""

import json
import random
import re
from pathlib import Path
from datetime import datetime

PHI = 1.618033988749895

# ANTITHESIS prompt templates for generating new examples
ANTITHESIS_TEMPLATES = [
    # Challenge assumptions
    "Challenge this assumption: {claim}",
    "What implicit assumptions does this make: {claim}",
    "Find the hidden premises in: {claim}",
    "What is this argument taking for granted: {claim}",
    
    # Critique
    "[ANTITHESIS] Analyze critically: {claim}",
    "[ANTITHESIS] What are the weaknesses in: {claim}",
    "[ANTITHESIS] Steel-man then critique: {claim}",
    "Where does this reasoning fail: {claim}",
    
    # Logical analysis
    "Identify the logical flaw: {claim}",
    "What fallacy might be present: {claim}",
    "Is this valid or merely plausible: {claim}",
    "Distinguish correlation from causation in: {claim}",
    
    # Counterarguments
    "Provide a counterargument to: {claim}",
    "What would falsify this claim: {claim}",
    "Under what conditions would this be false: {claim}",
    "What evidence would disprove: {claim}",
]

# Claims to generate ANTITHESIS examples from
CLAIMS_TO_CRITIQUE = [
    # Technology
    "AI will always benefit humanity",
    "More data leads to better decisions",
    "Technology is neutral",
    "Automation creates more jobs than it destroys",
    "Social media connects people",
    "Digital privacy is dead",
    "The cloud is always more secure",
    "Moore's law will continue indefinitely",
    
    # Society
    "Progress is inevitable",
    "Democracy is the best system",
    "Free markets optimize welfare",
    "Education guarantees success",
    "Hard work always pays off",
    "History repeats itself",
    "Humans are fundamentally rational",
    "Competition drives innovation",
    
    # Philosophy
    "Truth is objective",
    "Consciousness requires biology",
    "Free will exists",
    "The mind is separate from the brain",
    "Language shapes thought",
    "Beauty is subjective",
    "Time flows in one direction",
    "Identity persists over time",
    
    # Science
    "Science is self-correcting",
    "Correlation implies causation",
    "Simple explanations are better",
    "Natural is better than artificial",
    "Evolution optimizes for fitness",
    "The universe has a purpose",
    "Quantum effects don't matter at macro scale",
    "Reductionism explains everything",
    
    # Logic
    "All generalizations are false",
    "If it's popular, it must be good",
    "Absence of evidence is evidence of absence",
    "The exception proves the rule",
    "Past performance predicts future results",
    "Complex problems require complex solutions",
    "More choices lead to better outcomes",
    "Expertise guarantees correctness",
]

# ANTITHESIS response templates (the model should learn to generate these patterns)
ANTITHESIS_RESPONSE_PATTERNS = [
    "This claim contains implicit assumptions:\n1. {assumption1}\n2. {assumption2}\n\nCritique: {critique}",
    "∃x: claim(x) → requires_evidence(x)\n\nThe assertion '{claim}' assumes {assumption}. However, {counter}",
    "Logical analysis:\n- Premise: {premise}\n- Hidden assumption: {assumption}\n- Potential flaw: {flaw}",
    "[ANTITHESIS ENGAGED]\n\nThe claim '{claim}' exhibits {fallacy_type}. Consider: {alternative}",
    "Critical examination:\n\n∀x: {claim_formal} → ¬necessarily_true(x)\n\nBecause: {reason}",
]

def extract_user_content(text: str) -> tuple[str, str] | None:
    """Extract user prompt and assistant response from chat format."""
    pattern = r"<\|im_start\|>user\n(.+?)<\|im_end\|>\n<\|im_start\|>assistant\n(.+?)<\|im_end\|>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None

def build_chat_format(user_content: str, assistant_content: str) -> str:
    """Build chat format string."""
    return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{assistant_content}<|im_end|>"

def categorize_prompt(prompt: str) -> str:
    """Categorize a prompt into its type."""
    if re.search(r'^\?|^[A-Z]:\s*[●◐◑⊤⊥]|[∀∃]|→.*→|⊢', prompt):
        return 'pure_asl_logic'
    elif re.search(r'haiku', prompt, re.IGNORECASE):
        return 'creative_haiku'
    elif re.search(r'\[CREATIVE\]|creative twin|explore:', prompt, re.IGNORECASE):
        return 'creative_exploration'
    elif re.search(r'^Express|^Encode|^Translate|^Formalize', prompt, re.IGNORECASE):
        return 'agl_expression'
    elif re.search(r'ANTITHESIS|challenge|assumption|critique|flaw|fallacy|counter', prompt, re.IGNORECASE):
        return 'antithesis'
    elif re.search(r'What if|Imagine|counterfactual', prompt, re.IGNORECASE):
        return 'counterfactual'
    else:
        return 'other'

def generate_antithesis_example(claim: str) -> dict:
    """Generate a new ANTITHESIS training example."""
    template = random.choice(ANTITHESIS_TEMPLATES)
    prompt = template.format(claim=claim)
    
    # Generate a structured critical response
    # These are templates - the model learns the PATTERN
    assumptions = [
        "that the claim applies universally",
        "that current conditions will persist",
        "that correlation implies causation",
        "that the framing is neutral",
        "that alternatives don't exist",
        "that the evidence is complete",
    ]
    
    critiques = [
        "this overlooks edge cases and exceptions",
        "the reasoning conflates distinct concepts",
        "this commits the naturalistic fallacy",
        "the argument proves too much",
        "counterexamples exist in practice",
        "the burden of proof is misplaced",
    ]
    
    # Build response with AGL notation + critique
    response_parts = []
    
    # Add AGL formal notation
    claim_short = claim.lower().replace(" ", "_")[:20]
    response_parts.append(f"∃x: claim({claim_short}) → requires_evidence(x)")
    response_parts.append("")
    response_parts.append(f"Critical analysis of: \"{claim}\"")
    response_parts.append("")
    response_parts.append(f"Implicit assumption: {random.choice(assumptions)}")
    response_parts.append(f"Critique: {random.choice(critiques)}")
    response_parts.append("")
    response_parts.append("The claim would be stronger if it acknowledged these limitations.")
    
    response = "\n".join(response_parts)
    
    return {"text": build_chat_format(prompt, response)}

def process_dataset(input_path: Path, output_path: Path, target_antithesis_pct: float = 0.20):
    """Process v5d data to create v5e with boosted ANTITHESIS."""
    print(f"Loading {input_path}...")
    
    with open(input_path) as f:
        data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(data)} examples")
    
    # Categorize all examples
    categorized = {
        'pure_asl_logic': [],
        'creative_haiku': [],
        'creative_exploration': [],
        'agl_expression': [],
        'antithesis': [],
        'counterfactual': [],
        'other': [],
    }
    
    for item in data:
        parsed = extract_user_content(item.get("text", ""))
        if parsed:
            prompt, response = parsed
            cat = categorize_prompt(prompt)
            categorized[cat].append(item)
        else:
            categorized['other'].append(item)
    
    print("\nOriginal distribution:")
    for cat, items in sorted(categorized.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(items)} ({len(items)/len(data)*100:.1f}%)")
    
    # Target: 10,000 examples with φ-distributed categories
    # ANTITHESIS should be ~20% (2000 examples) - the v5 differentiator!
    target_total = 10000
    
    # φ-based distribution for v5e (logical focus with strong ANTITHESIS)
    # LOGICAL : CREATIVE = φ : 1
    # Within LOGICAL: pure_asl : antithesis : agl_expression = balanced
    
    target_antithesis = int(target_total * target_antithesis_pct)  # 20% = 2000
    target_pure_asl = int(target_total * 0.45)  # 45% = 4500 (our best basin!)
    target_creative = int(target_total * 0.20)  # 20% = 2000 (haiku + exploration)
    target_other = target_total - target_antithesis - target_pure_asl - target_creative  # 15% = 1500
    
    print(f"\nTarget distribution (φ-aligned for v5e):")
    print(f"  pure_asl_logic: {target_pure_asl} ({target_pure_asl/target_total*100:.1f}%)")
    print(f"  antithesis: {target_antithesis} ({target_antithesis/target_total*100:.1f}%) ← BOOSTED!")
    print(f"  creative: {target_creative} ({target_creative/target_total*100:.1f}%)")
    print(f"  other: {target_other} ({target_other/target_total*100:.1f}%)")
    
    # Build v5e dataset
    v5e_data = []
    
    # 1. Pure ASL logic (sample or expand)
    pure_asl = categorized['pure_asl_logic']
    if len(pure_asl) >= target_pure_asl:
        v5e_data.extend(random.sample(pure_asl, target_pure_asl))
    else:
        v5e_data.extend(pure_asl)
        # Duplicate with slight variation if needed
        while len([x for x in v5e_data if categorize_prompt(extract_user_content(x['text'])[0]) == 'pure_asl_logic']) < target_pure_asl:
            v5e_data.append(random.choice(pure_asl))
    
    # 2. ANTITHESIS - generate new examples to reach target!
    existing_antithesis = categorized['antithesis']
    v5e_data.extend(existing_antithesis)
    
    needed_antithesis = target_antithesis - len(existing_antithesis)
    print(f"\n  Generating {needed_antithesis} new ANTITHESIS examples...")
    
    for _ in range(needed_antithesis):
        claim = random.choice(CLAIMS_TO_CRITIQUE)
        new_example = generate_antithesis_example(claim)
        v5e_data.append(new_example)
    
    # 3. Creative content (haiku + exploration)
    creative_pool = categorized['creative_haiku'] + categorized['creative_exploration']
    if len(creative_pool) >= target_creative:
        v5e_data.extend(random.sample(creative_pool, target_creative))
    else:
        v5e_data.extend(creative_pool)
    
    # 4. Fill rest with other/agl_expression
    other_pool = categorized['other'] + categorized['agl_expression'] + categorized['counterfactual']
    remaining = target_total - len(v5e_data)
    if remaining > 0 and other_pool:
        v5e_data.extend(random.sample(other_pool, min(remaining, len(other_pool))))
    
    # Shuffle
    random.shuffle(v5e_data)
    
    # Trim to exact target
    v5e_data = v5e_data[:target_total]
    
    # Final stats
    final_cats = {}
    for item in v5e_data:
        parsed = extract_user_content(item.get("text", ""))
        if parsed:
            cat = categorize_prompt(parsed[0])
            final_cats[cat] = final_cats.get(cat, 0) + 1
    
    print(f"\n📊 Final v5e distribution:")
    for cat, count in sorted(final_cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({count/len(v5e_data)*100:.1f}%)")
    
    # Write output
    print(f"\nWriting {len(v5e_data)} examples to {output_path}...")
    with open(output_path, 'w') as f:
        for item in v5e_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return final_cats

def main():
    print("=" * 60)
    print("v5e Training Data Generator - ANTITHESIS BOOST")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    
    base_path = Path.home() / "Code/ada-slm"
    input_path = base_path / "v5d_logical_data.jsonl"
    output_path = base_path / "v5e_antithesis_data.jsonl"
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    # 20% ANTITHESIS - the v5 differentiator!
    stats = process_dataset(input_path, output_path, target_antithesis_pct=0.20)
    
    print(f"\n✓ v5e dataset created: {output_path}")
    
    # Save metadata
    meta_path = base_path / "v5e_antithesis_meta.json"
    meta = {
        "created": datetime.now().isoformat(),
        "source": str(input_path),
        "output": str(output_path),
        "stats": stats,
        "strategy": "Boost ANTITHESIS to 20% for critical analysis capability",
        "insight": "v5d had only 0.2% ANTITHESIS, causing 0% critical activation in basin mapping",
        "target_distribution": {
            "pure_asl_logic": "45% - our best basin from v5d mapping",
            "antithesis": "20% - BOOSTED from 0.2%! v5 differentiator",
            "creative": "20% - maintain creative capacity",
            "other": "15% - fill"
        }
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved: {meta_path}")

if __name__ == "__main__":
    main()
