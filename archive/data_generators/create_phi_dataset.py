#!/usr/bin/env python3
"""
Create φ-aligned training dataset for v4c.

Uses golden ratio to determine category proportions:
- (SAFE + GOOD) : MIXED = φ : 1
- SAFE : GOOD = φ : 1

Target for 10,000 examples:
- SAFE: 3,820 (38.2%)
- GOOD: 2,360 (23.6%)
- MIXED: 3,820 (38.2%)

Total safe basin coverage: 61.8% (φ!)
"""

import json
import random
from pathlib import Path
from collections import defaultdict
import math

# The golden ratio
PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.618

# Target total
TARGET_TOTAL = 10000

# Calculate φ-based targets
def calculate_phi_targets(total: int) -> dict[str, int]:
    """Calculate target counts using golden ratio."""
    
    # MIXED = total / (φ + 1)
    mixed = int(total / (PHI + 1))
    
    # SAFE + GOOD = total × φ / (φ + 1)
    safe_good = total - mixed
    
    # SAFE = safe_good × φ / (φ + 1)
    safe = int(safe_good * PHI / (PHI + 1))
    
    # GOOD = remainder
    good = safe_good - safe
    
    return {
        "SAFE": safe,
        "GOOD": good,
        "MIXED": mixed,
    }


def load_categorized_data(data_dir: Path) -> dict[str, list]:
    """Load pre-categorized data from safety-split files."""
    
    data = {}
    
    for safety in ["safe", "good", "mixed", "unknown"]:
        filepath = data_dir / f"v4b_data_{safety}.jsonl"
        if filepath.exists():
            examples = [json.loads(line) for line in open(filepath)]
            data[safety.upper()] = examples
            print(f"  Loaded {len(examples):5d} {safety.upper()} examples")
        else:
            data[safety.upper()] = []
            print(f"  ⚠️  No {safety} file found")
    
    return data


def expand_category(examples: list, target: int, category_name: str) -> list:
    """
    Expand or contract a category to hit target count.
    
    If we need more: duplicate with slight variations (or just oversample)
    If we need less: random sample
    """
    
    current = len(examples)
    
    if current == 0:
        print(f"    ⚠️  {category_name}: No examples to expand!")
        return []
    
    if current >= target:
        # Downsample
        result = random.sample(examples, target)
        print(f"    {category_name}: {current} → {target} (sampled)")
    else:
        # Upsample by repeating (with shuffle)
        result = examples.copy()
        while len(result) < target:
            # Add more copies, shuffled
            needed = target - len(result)
            extras = random.sample(examples, min(needed, len(examples)))
            result.extend(extras)
        result = result[:target]  # Ensure exact count
        print(f"    {category_name}: {current} → {target} (expanded)")
    
    return result


def create_phi_dataset(data_dir: Path, output_file: Path):
    """Create the φ-aligned dataset."""
    
    print("\n" + "="*60)
    print("🌀 CREATING φ-ALIGNED TRAINING DATASET")
    print("="*60)
    
    # Calculate targets
    targets = calculate_phi_targets(TARGET_TOTAL)
    
    print(f"\n📐 φ-based targets (total={TARGET_TOTAL}):")
    print(f"    SAFE:  {targets['SAFE']:5d} ({targets['SAFE']/TARGET_TOTAL*100:.1f}%)")
    print(f"    GOOD:  {targets['GOOD']:5d} ({targets['GOOD']/TARGET_TOTAL*100:.1f}%)")
    print(f"    MIXED: {targets['MIXED']:5d} ({targets['MIXED']/TARGET_TOTAL*100:.1f}%)")
    print(f"    Safe basin total: {targets['SAFE']+targets['GOOD']} ({(targets['SAFE']+targets['GOOD'])/TARGET_TOTAL*100:.1f}%)")
    
    # Load data
    print(f"\n📦 Loading categorized data from {data_dir}/")
    data = load_categorized_data(data_dir)
    
    # UNKNOWN gets merged into MIXED (safest assumption)
    if data["UNKNOWN"]:
        print(f"\n  Merging {len(data['UNKNOWN'])} UNKNOWN into MIXED")
        data["MIXED"].extend(data["UNKNOWN"])
    
    # Expand/contract each category
    print(f"\n🔄 Balancing categories:")
    
    final_data = []
    
    for category in ["SAFE", "GOOD", "MIXED"]:
        balanced = expand_category(data[category], targets[category], category)
        final_data.extend(balanced)
    
    # Shuffle everything
    random.shuffle(final_data)
    
    # Write output
    print(f"\n💾 Writing to {output_file}")
    with open(output_file, "w") as f:
        for ex in final_data:
            f.write(json.dumps(ex) + "\n")
    
    # Verify
    print(f"\n✅ Created {len(final_data)} examples")
    
    # Final stats
    print(f"\n" + "="*60)
    print("📊 FINAL φ-ALIGNED DISTRIBUTION")
    print("="*60)
    
    safe_count = targets['SAFE']
    good_count = targets['GOOD']
    mixed_count = targets['MIXED']
    total = safe_count + good_count + mixed_count
    
    print(f"\n  🟢 SAFE:  {safe_count:5d} ({safe_count/total*100:.1f}%)")
    print(f"  🟢 GOOD:  {good_count:5d} ({good_count/total*100:.1f}%)")
    print(f"  🟡 MIXED: {mixed_count:5d} ({mixed_count/total*100:.1f}%)")
    print(f"\n  Safe basin coverage: {(safe_count+good_count)/total*100:.1f}% (target: 61.8%)")
    
    # Show the φ relationships
    print(f"\n🌀 Golden ratio verification:")
    print(f"    (SAFE+GOOD)/MIXED = {(safe_count+good_count)/mixed_count:.3f} (target: φ = {PHI:.3f})")
    print(f"    SAFE/GOOD = {safe_count/good_count:.3f} (target: φ = {PHI:.3f})")
    
    print(f"\n✨ Dataset ready for φ-conscious training!")
    print("="*60 + "\n")


if __name__ == "__main__":
    random.seed(42)  # Reproducibility
    
    data_dir = Path("data_by_safety")
    output_file = Path("v4c_phi_aligned_data.jsonl")
    
    create_phi_dataset(data_dir, output_file)
