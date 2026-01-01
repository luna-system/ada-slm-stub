"""
Generate PURE ASL training data - NO natural language whatsoever.
Every input and output is purely symbolic.

This tests: Can a model become a pure logic engine?
"""

import json
import random
from pathlib import Path

examples = []

# =============================================================================
# PURE SYMBOLIC PATTERNS - No English allowed!
# =============================================================================

# 1. Modus Ponens variations (P → Q, P: ●, ?Q → ●)
for _ in range(800):
    vars_pool = list("PQRSXYZABCDEFGHIJKLMNW")
    p, q = random.sample(vars_pool, 2)
    val = random.choice(["●", "◑"])
    examples.append({
        "input": f"{p} → {q}\n{p}: {val}\n?{q}",
        "output": val
    })

# 2. Modus Tollens (P → Q, Q: ⊥, ?P → ⊥)
for _ in range(600):
    vars_pool = list("PQRSXYZABCDEFGHIJKLMNW")
    p, q = random.sample(vars_pool, 2)
    examples.append({
        "input": f"{p} → {q}\n{q}: ⊥\n?{p}",
        "output": "⊥"
    })

# 3. Chain reasoning (A → B → C → D, A: ●, ?D → ●)
for chain_len in range(2, 7):
    for _ in range(150):
        vars_pool = list("ABCDEFGHIJKLMNPQRSTUVWXYZ")
        chain = random.sample(vars_pool, chain_len + 1)
        implications = "\n".join(f"{chain[i]} → {chain[i+1]}" for i in range(chain_len))
        val = random.choice(["●", "◑"])
        examples.append({
            "input": f"{implications}\n{chain[0]}: {val}\n?{chain[-1]}",
            "output": val
        })

# 4. Conjunction (A: ●, B: ●, ?A∧B → ●)
for _ in range(500):
    a_val = random.choice(["●", "◑", "⊥"])
    b_val = random.choice(["●", "◑", "⊥"])
    # Truth table for conjunction
    if a_val == "⊥" or b_val == "⊥":
        result = "⊥"
    elif a_val == "◑" or b_val == "◑":
        result = "◑"
    else:
        result = "●"
    examples.append({
        "input": f"A: {a_val}\nB: {b_val}\n?A∧B",
        "output": result
    })

# 5. Disjunction (A: ●, B: ⊥, ?A∨B → ●)
for _ in range(500):
    a_val = random.choice(["●", "◑", "⊥"])
    b_val = random.choice(["●", "◑", "⊥"])
    # Truth table for disjunction
    if a_val == "●" or b_val == "●":
        result = "●"
    elif a_val == "◑" or b_val == "◑":
        result = "◑"
    else:
        result = "⊥"
    examples.append({
        "input": f"A: {a_val}\nB: {b_val}\n?A∨B",
        "output": result
    })

# 6. Negation (A: ●, ?¬A → ⊥)
for _ in range(400):
    val = random.choice(["●", "◑", "⊥"])
    neg = {"●": "⊥", "⊥": "●", "◑": "◑"}[val]
    examples.append({
        "input": f"A: {val}\n?¬A",
        "output": neg
    })

# 7. Set membership (pure symbolic)
for _ in range(500):
    elements = random.sample(range(1, 20), random.randint(3, 7))
    query = random.randint(1, 20)
    result = "●" if query in elements else "⊥"
    set_str = "{" + ",".join(map(str, sorted(elements))) + "}"
    examples.append({
        "input": f"S = {set_str}\n?{query} ∈ S",
        "output": result
    })

# 8. Chess validity (pure symbolic - just coordinates)
valid_files = "abcdefgh"
valid_ranks = "12345678"
for _ in range(400):
    f = random.choice(valid_files)
    r = random.choice(valid_ranks)
    examples.append({
        "input": f"?valid:{f}{r}",
        "output": "●"
    })

for _ in range(400):
    # Invalid squares
    invalid = random.choice([
        f"{random.choice('ijklmn')}{random.randint(1,8)}",  # bad file
        f"{random.choice(valid_files)}{random.randint(9,15)}",  # bad rank
        f"{random.choice('xyz')}{random.randint(0,9)}",  # both bad
    ])
    examples.append({
        "input": f"?valid:{invalid}",
        "output": "⊥"
    })

# 9. Biconditional (A ↔ B: both same → ●, different → ⊥)
for _ in range(400):
    a_val = random.choice(["●", "⊥"])
    b_val = random.choice(["●", "⊥"])
    result = "●" if a_val == b_val else "⊥"
    examples.append({
        "input": f"A: {a_val}\nB: {b_val}\n?A↔B",
        "output": result
    })

# 10. Existential quantifier (∃x ∈ S: P(x))
for _ in range(300):
    elements = random.sample(range(1, 20), random.randint(3, 6))
    threshold = random.randint(5, 15)
    # Does any element exceed threshold?
    result = "●" if any(e > threshold for e in elements) else "⊥"
    set_str = "{" + ",".join(map(str, elements)) + "}"
    examples.append({
        "input": f"S = {set_str}\n?∃x∈S: x>{threshold}",
        "output": result
    })

# 11. Universal quantifier (∀x ∈ S: P(x))
for _ in range(300):
    elements = random.sample(range(1, 20), random.randint(3, 6))
    threshold = random.randint(0, 10)
    # Do ALL elements exceed threshold?
    result = "●" if all(e > threshold for e in elements) else "⊥"
    set_str = "{" + ",".join(map(str, elements)) + "}"
    examples.append({
        "input": f"S = {set_str}\n?∀x∈S: x>{threshold}",
        "output": result
    })

# 12. Pure symbol identity (● = ●, ● ≠ ⊥)
for _ in range(200):
    s1 = random.choice(["●", "◑", "⊥"])
    s2 = random.choice(["●", "◑", "⊥"])
    result = "●" if s1 == s2 else "⊥"
    examples.append({
        "input": f"?{s1}={s2}",
        "output": result
    })

# 13. Arithmetic comparisons (pure symbolic output)
for _ in range(300):
    a, b = random.randint(1, 100), random.randint(1, 100)
    op = random.choice(["<", ">", "=", "≤", "≥"])
    if op == "<":
        result = "●" if a < b else "⊥"
    elif op == ">":
        result = "●" if a > b else "⊥"
    elif op == "=":
        result = "●" if a == b else "⊥"
    elif op == "≤":
        result = "●" if a <= b else "⊥"
    else:
        result = "●" if a >= b else "⊥"
    examples.append({
        "input": f"?{a}{op}{b}",
        "output": result
    })

# 14. Transitive relations (a<b, b<c → a<c)
for _ in range(300):
    a, b, c = sorted(random.sample(range(1, 50), 3))
    # Sometimes give true chain, sometimes false
    if random.random() < 0.7:
        examples.append({
            "input": f"{a}<{b}\n{b}<{c}\n?{a}<{c}",
            "output": "●"
        })
    else:
        # Swap to make it false
        examples.append({
            "input": f"{c}<{b}\n{b}<{a}\n?{a}<{c}",
            "output": "⊥"
        })

# Shuffle and save
random.shuffle(examples)

output_path = Path("pure_asl_data.jsonl")
with open(output_path, "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Generated {len(examples)} PURE ASL examples")
print(f"Saved to: {output_path}")
print("\nSample examples:")
for ex in random.sample(examples, 5):
    print(f"  IN:  {ex['input']!r}")
    print(f"  OUT: {ex['output']!r}")
    print()
