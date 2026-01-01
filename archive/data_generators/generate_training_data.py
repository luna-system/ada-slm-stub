#!/usr/bin/env python3
"""Generate ASL training data - SCALED UP to 5k examples.

This creates the dataset we'll use to fine-tune qwen2.5:0.5b.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass
import time

# === ASL Symbol Reference ===
ASL_SYMBOLS = """
● = certain/true/valid
◑ = uncertain/possible/unknown
⊥ = contradiction/impossible/invalid
→ = implies/leads to
← = follows from
⟷ = bidirectional/equivalent
∧ = and/conjunction
∨ = or/disjunction  
¬ = not/negation
∈ = element of/member of
∉ = not element of
∴ = therefore/conclusion
∵ = because/reason
? = query/unknown
! = assert/certain
"""

# === Training Example Templates ===

LOGIC_TEMPLATES = [
    # Modus Ponens
    {"pattern": "modus_ponens", "input": "P → Q\nP: ●\n?Q", "output": "●"},
    {"pattern": "modus_ponens_uncertain", "input": "P → Q\nP: ◑\n?Q", "output": "◑"},
    # Modus Tollens
    {"pattern": "modus_tollens", "input": "P → Q\nQ: ⊥\n?P", "output": "⊥"},
    # Conjunction
    {"pattern": "conjunction_tt", "input": "A: ●\nB: ●\n?A∧B", "output": "●"},
    {"pattern": "conjunction_tu", "input": "A: ●\nB: ◑\n?A∧B", "output": "◑"},
    {"pattern": "conjunction_tf", "input": "A: ●\nB: ⊥\n?A∧B", "output": "⊥"},
    {"pattern": "conjunction_uu", "input": "A: ◑\nB: ◑\n?A∧B", "output": "◑"},
    {"pattern": "conjunction_uf", "input": "A: ◑\nB: ⊥\n?A∧B", "output": "⊥"},
    {"pattern": "conjunction_ff", "input": "A: ⊥\nB: ⊥\n?A∧B", "output": "⊥"},
    # Disjunction
    {"pattern": "disjunction_tt", "input": "A: ●\nB: ●\n?A∨B", "output": "●"},
    {"pattern": "disjunction_tf", "input": "A: ●\nB: ⊥\n?A∨B", "output": "●"},
    {"pattern": "disjunction_tu", "input": "A: ●\nB: ◑\n?A∨B", "output": "●"},
    {"pattern": "disjunction_uu", "input": "A: ◑\nB: ◑\n?A∨B", "output": "◑"},
    {"pattern": "disjunction_uf", "input": "A: ◑\nB: ⊥\n?A∨B", "output": "◑"},
    {"pattern": "disjunction_ff", "input": "A: ⊥\nB: ⊥\n?A∨B", "output": "⊥"},
    # Contradiction detection
    {"pattern": "contradiction", "input": "P: ●\n¬P: ●\n?consistent", "output": "⊥"},
    {"pattern": "no_contradiction", "input": "P: ●\nQ: ●\n?consistent", "output": "●"},
    # Negation
    {"pattern": "negation_t", "input": "P: ●\n?¬P", "output": "⊥"},
    {"pattern": "negation_f", "input": "P: ⊥\n?¬P", "output": "●"},
    {"pattern": "negation_u", "input": "P: ◑\n?¬P", "output": "◑"},
]

UNCERTAINTY_PROPAGATION = [
    {"pattern": "chain_certain", "input": "A: ●\nA → B\nB → C\n?C", "output": "●"},
    {"pattern": "chain_uncertain", "input": "A: ◑\nA → B\nB → C\n?C", "output": "◑"},
    {"pattern": "chain_breaks", "input": "A: ●\nA → B\nB: ⊥\n?C where B → C", "output": "⊥"},
    {"pattern": "chain_long", "input": "X: ●\nX → Y\nY → Z\nZ → W\n?W", "output": "●"},
]

@dataclass
class TrainingExample:
    input: str
    output: str
    pattern: str


def generate_variable_variations(n: int = 500) -> list[TrainingExample]:
    """Generate modus ponens/tollens with different variable names."""
    examples = []
    var_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["P", "Q", "R", "S", "T"]
    
    for _ in range(n):
        p, q = random.sample(var_names, 2)
        
        # Modus ponens certain
        examples.append(TrainingExample(f"{p} → {q}\n{p}: ●\n?{q}", "●", "mp_var"))
        # Modus ponens uncertain
        examples.append(TrainingExample(f"{p} → {q}\n{p}: ◑\n?{q}", "◑", "mp_uncertain_var"))
        # Modus tollens
        examples.append(TrainingExample(f"{p} → {q}\n{q}: ⊥\n?{p}", "⊥", "mt_var"))
    
    return examples


def generate_chess_examples(n: int = 1000) -> list[TrainingExample]:
    """Generate chess square validation examples."""
    examples = []
    valid_files = set('abcdefgh')
    valid_ranks = set('12345678')
    
    for _ in range(n):
        r = random.random()
        
        if r < 0.5:  # Valid square
            f = random.choice('abcdefgh')
            rank = random.choice('12345678')
            square = f"{f}{rank}"
            examples.append(TrainingExample(
                f"?valid:{square}",
                "●",
                "chess_valid"
            ))
        elif r < 0.75:  # Invalid file
            f = random.choice('ijklmnop')
            rank = random.choice('12345678')
            square = f"{f}{rank}"
            examples.append(TrainingExample(
                f"?valid:{square}",
                "⊥",
                "chess_invalid_file"
            ))
        else:  # Invalid rank
            f = random.choice('abcdefgh')
            rank = random.choice(['0', '9', '10', '11'])
            square = f"{f}{rank}"
            examples.append(TrainingExample(
                f"?valid:{square}",
                "⊥",
                "chess_invalid_rank"
            ))
    
    return examples


def generate_set_membership(n: int = 400) -> list[TrainingExample]:
    """Generate set membership examples."""
    examples = []
    
    for _ in range(n):
        # Create a small set
        elements = random.sample(range(1, 20), random.randint(3, 7))
        set_str = ",".join(map(str, sorted(elements)))
        
        if random.random() < 0.5:
            # Element is in set
            x = random.choice(elements)
            examples.append(TrainingExample(
                f"S = {{{set_str}}}\n?{x} ∈ S",
                "●",
                "set_member"
            ))
        else:
            # Element not in set
            x = random.choice([i for i in range(1, 25) if i not in elements])
            examples.append(TrainingExample(
                f"S = {{{set_str}}}\n?{x} ∈ S",
                "⊥",
                "set_not_member"
            ))
    
    return examples


def generate_chain_reasoning(n: int = 500) -> list[TrainingExample]:
    """Generate chain reasoning with varying lengths."""
    examples = []
    
    for _ in range(n):
        length = random.randint(2, 5)
        vars = random.sample(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), length + 1)
        
        # Build chain
        lines = [f"{vars[0]}: ●"]
        for i in range(length):
            lines.append(f"{vars[i]} → {vars[i+1]}")
        lines.append(f"?{vars[-1]}")
        
        examples.append(TrainingExample(
            "\n".join(lines),
            "●",
            f"chain_{length}"
        ))
        
        # Also uncertain version
        lines[0] = f"{vars[0]}: ◑"
        examples.append(TrainingExample(
            "\n".join(lines),
            "◑",
            f"chain_{length}_uncertain"
        ))
    
    return examples


def generate_conjunction_variations(n: int = 400) -> list[TrainingExample]:
    """Generate conjunction with different variable names."""
    examples = []
    states = [("●", "●", "●"), ("●", "◑", "◑"), ("●", "⊥", "⊥"),
              ("◑", "◑", "◑"), ("◑", "⊥", "⊥"), ("⊥", "⊥", "⊥")]
    
    for _ in range(n):
        a, b = random.sample(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 2)
        s1, s2, result = random.choice(states)
        examples.append(TrainingExample(
            f"{a}: {s1}\n{b}: {s2}\n?{a}∧{b}",
            result,
            "conj_var"
        ))
    
    return examples


def generate_disjunction_variations(n: int = 400) -> list[TrainingExample]:
    """Generate disjunction with different variable names."""
    examples = []
    # For OR: true if any is true, uncertain if none true but some uncertain
    states = [("●", "●", "●"), ("●", "⊥", "●"), ("●", "◑", "●"),
              ("◑", "◑", "◑"), ("◑", "⊥", "◑"), ("⊥", "⊥", "⊥")]
    
    for _ in range(n):
        a, b = random.sample(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 2)
        s1, s2, result = random.choice(states)
        examples.append(TrainingExample(
            f"{a}: {s1}\n{b}: {s2}\n?{a}∨{b}",
            result,
            "disj_var"
        ))
    
    return examples


def format_for_training(examples: list[TrainingExample]) -> list[dict]:
    """Format examples for training - SIMPLE format for the model to learn."""
    formatted = []
    
    for ex in examples:
        # Simple chat format - input -> output
        formatted.append({
            "text": f"<|im_start|>user\n{ex.input}<|im_end|>\n<|im_start|>assistant\n{ex.output}<|im_end|>",
            "pattern": ex.pattern,
        })
    
    return formatted


def main():
    """Generate and save training data."""
    start = time.time()
    print("=" * 60)
    print("GENERATING ASL TRAINING DATA (5k target)")
    print("=" * 60)
    
    examples = []
    
    # Static templates (with repetition for emphasis)
    print("  - Static logic templates (x20 repetitions)...")
    for _ in range(20):
        for t in LOGIC_TEMPLATES:
            examples.append(TrainingExample(t["input"], t["output"], t["pattern"]))
        for t in UNCERTAINTY_PROPAGATION:
            examples.append(TrainingExample(t["input"], t["output"], t["pattern"]))
    
    print("  - Variable variations (1500 examples)...")
    examples.extend(generate_variable_variations(500))
    
    print("  - Chess validation (1000 examples)...")
    examples.extend(generate_chess_examples(1000))
    
    print("  - Chain reasoning (1000 examples)...")
    examples.extend(generate_chain_reasoning(500))
    
    print("  - Conjunction variations (400 examples)...")
    examples.extend(generate_conjunction_variations(400))
    
    print("  - Disjunction variations (400 examples)...")
    examples.extend(generate_disjunction_variations(400))
    
    print("  - Set membership (400 examples)...")
    examples.extend(generate_set_membership(400))
    
    # Shuffle
    random.shuffle(examples)
    
    elapsed = time.time() - start
    print(f"\n✓ Generated {len(examples)} examples in {elapsed:.2f}s")
    
    # Format for training
    formatted = format_for_training(examples)
    
    # Save
    output_path = Path(__file__).parent / "asl_training_data.jsonl"
    with open(output_path, "w") as f:
        for item in formatted:
            f.write(json.dumps(item) + "\n")
    
    print(f"✓ Saved to {output_path}")
    
    # Show sample
    print("\n=== Sample Examples ===")
    for ex in random.sample(examples[:100], 5):
        print(f"\n[{ex.pattern}]")
        print(f"IN:  {ex.input.replace(chr(10), ' | ')}")
        print(f"OUT: {ex.output}")
    
    # Stats
    from collections import Counter
    patterns = Counter(ex.pattern for ex in examples)
    print("\n=== Pattern Distribution ===")
    for p, c in patterns.most_common(10):
        print(f"  {p}: {c}")


if __name__ == "__main__":
    main()
