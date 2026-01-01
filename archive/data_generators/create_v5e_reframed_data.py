#!/usr/bin/env python3
"""
Create v5e Training Data with Formal Logic Reframing
January 1, 2026

Key insight from v5d basin mapping:
- "Express X in AGL" → triggers creative basin → emoji cascade
- "Prove: ∀x(P(x))" → triggers logic basin → clean output

Strategy: Reframe AGL/symbolic prompts to use formal logic framing
while preserving the same semantic content.
"""

import json
import random
import re
from pathlib import Path
from datetime import datetime

# Reframing templates - convert creative framing to logical framing
LOGIC_FRAMES = [
    "Given the concept '{concept}', provide the formal logical representation:",
    "Prove or formalize: {concept}",
    "Complete the logical formula for: {concept}",
    "Express as a well-formed formula: {concept}",
    "Derive the logical structure of: {concept}",
    "What is the formal notation for: {concept}",
    "Construct the predicate logic for: {concept}",
    "If we define '{concept}', what logical form follows?",
    "Formalize using quantifiers and predicates: {concept}",
    "What logical axiom captures: {concept}",
]

# Patterns that trigger creative mode (to be reframed)
CREATIVE_TRIGGERS = [
    (r"^Express\s+['\"]?(.+?)['\"]?\s+in\s+(symbolic|AGL|your)", 1),
    (r"^Translate\s+to\s+AGL:\s*(.+)", 1),
    (r"^Encode:\s*(.+)", 1),
    (r"^Formalize:\s*(.+)", 1),
    (r"^Write\s+in\s+symbolic\s+form:\s*(.+)", 1),
    (r"^Convert\s+to\s+symbols?:\s*(.+)", 1),
    (r"^Represent\s+symbolically:\s*(.+)", 1),
    (r"^How would you express\s+['\"]?(.+?)['\"]?\s+in", 1),
    (r"^Create an? (AGL|symbolic)\s+expression\s+for\s+['\"]?(.+?)['\"]?", 2),
]

def extract_user_content(text: str) -> tuple[str, str] | None:
    """Extract user prompt and assistant response from chat format."""
    # Match <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>
    pattern = r"<\|im_start\|>user\n(.+?)<\|im_end\|>\n<\|im_start\|>assistant\n(.+?)<\|im_end\|>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None

def rebuild_chat_format(user_content: str, assistant_content: str) -> str:
    """Rebuild the chat format string."""
    return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{assistant_content}<|im_end|>"

def extract_concept(prompt: str) -> str | None:
    """Extract the concept/statement from a creative-framed prompt."""
    for pattern, group_idx in CREATIVE_TRIGGERS:
        match = re.match(pattern, prompt, re.IGNORECASE)
        if match:
            return match.group(group_idx).strip().strip("'\"")
    return None

def reframe_to_logic(prompt: str) -> str:
    """Reframe a creative prompt to use formal logic framing."""
    concept = extract_concept(prompt)
    if concept:
        template = random.choice(LOGIC_FRAMES)
        return template.format(concept=concept)
    return prompt  # Return unchanged if no match

def is_creative_framed(prompt: str) -> bool:
    """Check if prompt uses creative framing that triggers emoji cascades."""
    return any(re.match(p, prompt, re.IGNORECASE) for p, _ in CREATIVE_TRIGGERS)

def is_pure_logic_framed(prompt: str) -> bool:
    """Check if prompt already uses pure logic framing (keep as-is)."""
    logic_indicators = [
        r"^If\s+.+\s+(implies|then)\s+",
        r"^Given:\s*",
        r"^Prove:\s*",
        r"^Complete\s+the\s+(proof|formula)",
        r"^Evaluate:\s*",
        r"[∀∃][a-z]",
        r"⊢",
        r"→.*→",  # chained implications
        r"^\?",  # Query format like ?D
        r"^[A-Z]:\s*[●◐◑⊤⊥]",  # Variable assignment format
    ]
    return any(re.search(p, prompt, re.IGNORECASE) for p in logic_indicators)

def process_dataset(input_path: Path, output_path: Path):
    """Process v5d data to create v5e with reframed prompts."""
    print(f"Loading {input_path}...")
    
    with open(input_path) as f:
        data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(data)} examples")
    
    # Check format
    sample = data[0]
    if "text" in sample:
        print("  Format: Chat format with 'text' field")
    elif "prompt" in sample:
        print("  Format: prompt/response format")
    
    stats = {
        "total": len(data),
        "reframed": 0,
        "kept_logic": 0,
        "kept_other": 0,
        "parse_errors": 0,
    }
    
    reframed_data = []
    
    for item in data:
        if "text" in item:
            # Chat format
            parsed = extract_user_content(item["text"])
            if parsed is None:
                stats["parse_errors"] += 1
                reframed_data.append(item)
                continue
            
            user_prompt, assistant_response = parsed
            
            if is_pure_logic_framed(user_prompt):
                # Already good framing, keep as-is
                reframed_data.append(item)
                stats["kept_logic"] += 1
            elif is_creative_framed(user_prompt):
                # Reframe to logic style
                new_prompt = reframe_to_logic(user_prompt)
                new_text = rebuild_chat_format(new_prompt, assistant_response)
                new_item = {"text": new_text}
                reframed_data.append(new_item)
                stats["reframed"] += 1
                
                # Debug: show some reframings
                if stats["reframed"] <= 5:
                    print(f"\n  REFRAMED:")
                    print(f"    OLD: {user_prompt[:60]}...")
                    print(f"    NEW: {new_prompt[:60]}...")
            else:
                # Other prompts - keep as-is
                reframed_data.append(item)
                stats["kept_other"] += 1
        else:
            # prompt/response format (fallback)
            prompt = item.get("prompt", item.get("instruction", ""))
            if is_pure_logic_framed(prompt):
                reframed_data.append(item)
                stats["kept_logic"] += 1
            elif is_creative_framed(prompt):
                new_prompt = reframe_to_logic(prompt)
                new_item = item.copy()
                if "prompt" in new_item:
                    new_item["prompt"] = new_prompt
                if "instruction" in new_item:
                    new_item["instruction"] = new_prompt
                reframed_data.append(new_item)
                stats["reframed"] += 1
            else:
                reframed_data.append(item)
                stats["kept_other"] += 1
    
    # Shuffle to mix reframed examples throughout
    random.shuffle(reframed_data)
    
    print(f"\nWriting {len(reframed_data)} examples to {output_path}...")
    with open(output_path, 'w') as f:
        for item in reframed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n📊 Reframing Statistics:")
    print(f"   Total examples: {stats['total']}")
    print(f"   Reframed (creative→logic): {stats['reframed']} ({stats['reframed']/stats['total']*100:.1f}%)")
    print(f"   Kept (already logic): {stats['kept_logic']} ({stats['kept_logic']/stats['total']*100:.1f}%)")
    print(f"   Kept (other): {stats['kept_other']} ({stats['kept_other']/stats['total']*100:.1f}%)")
    if stats["parse_errors"]:
        print(f"   Parse errors: {stats['parse_errors']}")
    
    return stats

def main():
    print("=" * 60)
    print("v5e Training Data Generator - Formal Logic Reframing")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    
    base_path = Path.home() / "Code/ada-slm"
    input_path = base_path / "v5d_logical_data.jsonl"
    output_path = base_path / "v5e_reframed_data.jsonl"
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    stats = process_dataset(input_path, output_path)
    
    print(f"\n✓ v5e dataset created: {output_path}")
    
    # Save metadata
    meta_path = base_path / "v5e_reframed_meta.json"
    meta = {
        "created": datetime.now().isoformat(),
        "source": str(input_path),
        "output": str(output_path),
        "stats": stats,
        "strategy": "Reframe creative-trigger prompts to formal logic framing",
        "insight": "v5d mapping showed pure_logic basin has 0.5 avg emojis vs 12.5 for agl_expression"
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved: {meta_path}")

if __name__ == "__main__":
    main()
