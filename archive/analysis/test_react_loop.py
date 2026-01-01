#!/usr/bin/env python3
"""
Test v6-golden as core reasoning engine for ReAct loop.

ReAct = Reasoning + Acting in recursive loop:
1. Thought (reason about what to do next)
2. Action (execute a step)
3. Observation (see the result)
4. Repeat until done

This tests if v6-golden can handle the recursive reasoning
that's at the core of Ada's brain architecture.
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
from dataclasses import dataclass
from typing import List, Optional

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# Force CPU mode for testing (GPU state issue after long training)
DEVICE = "cpu"

@dataclass
class ReActStep:
    step_num: int
    thought: str
    action: str
    observation: str
    latency_ms: float
    is_final: bool

def load_v6_model():
    """Load v6-golden model."""
    base_dir = Path(__file__).parent
    lora_path = base_dir / "ada-slm-v6-golden" / "final"
    
    print("Loading v6-golden for ReAct testing...")
    print(f"Device: {DEVICE}")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    return model, tokenizer

def generate_thought(model, tokenizer, context: str, max_tokens: int = 100) -> tuple[str, float]:
    """Generate a reasoning thought given current context."""
    
    messages = [
        {"role": "system", "content": "You are a reasoning system. Think step-by-step using ASL symbols. Respond with your thought about what to do next."},
        {"role": "user", "content": context}
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
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    
    latency = (time.time() - start_time) * 1000
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response.strip(), latency

def execute_action(action: str, state: dict) -> str:
    """Execute an action and return observation (simulated for now)."""
    
    # Simple simulated actions for birthday party planning
    action_lower = action.lower()
    
    if "calculate" in action_lower and "guests" in action_lower:
        budget = state.get("budget", 100)
        per_person = 10
        guests = budget // per_person
        return f"With ${budget} budget at ${per_person}/person, you can invite {guests} guests."
    
    elif "allocate" in action_lower or "split" in action_lower:
        budget = state.get("budget", 100)
        # Look for percentages in action
        if "60" in action or "0.6" in action:
            food = budget * 0.60
            other = budget * 0.40
            return f"Allocated ${food:.0f} for food (60%), ${other:.0f} for decorations/other (40%)."
        else:
            # Default 50/50
            half = budget / 2
            return f"Allocated ${half:.0f} for food, ${half:.0f} for decorations."
    
    elif "verify" in action_lower or "check" in action_lower:
        # Check if we have key components
        has_guests = "guests" in str(state)
        has_budget_split = "food_budget" in str(state) or "allocated" in str(state.get("history", ""))
        
        if has_guests and has_budget_split:
            return "✓ Plan includes: guest count, budget allocation. Plan is complete!"
        else:
            missing = []
            if not has_guests: missing.append("guest count")
            if not has_budget_split: missing.append("budget allocation")
            return f"⚠ Plan missing: {', '.join(missing)}"
    
    elif "list" in action_lower:
        return "Key requirements: 1) Number of guests, 2) Budget allocation, 3) Venue/activities"
    
    else:
        return f"Executed: {action}"

def react_loop(model, tokenizer, task: str, max_steps: int = 10) -> List[ReActStep]:
    """Run ReAct loop until task is complete or max steps reached."""
    
    print(f"\n{'='*60}")
    print(f"REACT LOOP: {task}")
    print(f"{'='*60}\n")
    
    state = {
        "task": task,
        "budget": 100,
        "history": [],
        "complete": False
    }
    
    steps = []
    
    for step_num in range(1, max_steps + 1):
        print(f"\n--- Step {step_num} ---")
        
        # Build context from history
        context = f"Task: {task}\n\nHistory:\n"
        for prev_step in steps:
            context += f"Thought: {prev_step.thought}\n"
            context += f"Action: {prev_step.action}\n"
            context += f"Observation: {prev_step.observation}\n\n"
        
        context += "What should I think about or do next?"
        
        # Generate thought
        print("Thinking...")
        thought, latency = generate_thought(model, tokenizer, context, max_tokens=80)
        print(f"Thought: {thought} ({latency:.0f}ms)")
        
        # Determine action from thought
        # For now, let's parse the thought naively
        if "done" in thought.lower() or "complete" in thought.lower() or "✓" in thought:
            action = "FINISH"
            observation = "Task completed!"
            is_final = True
        elif "guest" in thought.lower() or "how many" in thought.lower():
            action = "Calculate number of guests within budget"
            observation = execute_action(action, state)
            is_final = False
        elif "allocate" in thought.lower() or "split" in thought.lower() or "budget" in thought.lower():
            action = "Allocate budget (60% food, 40% decorations)"
            observation = execute_action(action, state)
            is_final = False
        elif "verify" in thought.lower() or "check" in thought.lower():
            action = "Verify plan completeness"
            observation = execute_action(action, state)
            is_final = "✓" in observation
        else:
            action = "List requirements"
            observation = execute_action(action, state)
            is_final = False
        
        print(f"Action: {action}")
        print(f"Observation: {observation}")
        
        # Record step
        step = ReActStep(
            step_num=step_num,
            thought=thought,
            action=action,
            observation=observation,
            latency_ms=latency,
            is_final=is_final
        )
        steps.append(step)
        
        # Update state
        state["history"].append(f"Step {step_num}: {thought} → {action} → {observation}")
        
        if is_final:
            print(f"\n✓ Task completed in {step_num} steps!")
            break
    
    return steps

def analyze_results(steps: List[ReActStep]):
    """Analyze ReAct loop performance."""
    
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}\n")
    
    total_steps = len(steps)
    total_latency = sum(s.latency_ms for s in steps)
    avg_latency = total_latency / total_steps if total_steps > 0 else 0
    
    converged = any(s.is_final for s in steps)
    
    print(f"Total steps: {total_steps}")
    print(f"Converged: {'✓ Yes' if converged else '✗ No'}")
    print(f"Total latency: {total_latency:.0f}ms ({total_latency/1000:.2f}s)")
    print(f"Avg latency per step: {avg_latency:.0f}ms")
    
    print(f"\nStep-by-step breakdown:")
    for step in steps:
        status = "✓ FINAL" if step.is_final else "→"
        print(f"  {status} Step {step.step_num}: {step.latency_ms:.0f}ms")
    
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}\n")
    
    if converged:
        print("✓ Model successfully completed recursive reasoning task")
        print(f"✓ Completed in {total_steps} steps (reasonable)")
        print(f"✓ Total time: {total_latency/1000:.2f}s (acceptable for planning task)")
        
        if avg_latency < 400:
            print(f"✓ Average step latency: {avg_latency:.0f}ms (good!)")
        else:
            print(f"⚠ Average step latency: {avg_latency:.0f}ms (a bit slow)")
        
        print("\n** VERDICT: v6-golden shows promise as ReAct core! **")
    else:
        print("✗ Model did not complete task within step limit")
        print("⚠ May need more training or better prompting")
    
    return {
        "converged": converged,
        "total_steps": total_steps,
        "total_latency_ms": total_latency,
        "avg_latency_ms": avg_latency
    }

def main():
    print("="*60)
    print("v6-GOLDEN REACT LOOP TEST")
    print("="*60)
    print("\nTesting v6-golden as core reasoning engine for Ada's brain.")
    print("Task: Plan a birthday party with $100 budget")
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    
    # Load model
    model, tokenizer = load_v6_model()
    print("✓ Model loaded\n")
    
    # Run ReAct loop
    task = "Plan a birthday party with a $100 budget. Include guest count and budget allocation."
    steps = react_loop(model, tokenizer, task, max_steps=10)
    
    # Analyze
    results = analyze_results(steps)
    
    print("\n" + "="*60)
    print("IMPLICATIONS FOR ADA'S BRAIN")
    print("="*60 + "\n")
    
    if results["converged"]:
        print("✓ v6-golden CAN handle recursive reasoning loops")
        print("✓ Latency is acceptable for non-real-time planning tasks")
        print("✓ Could serve as core reasoning engine with proper scaffolding")
        print("\nNext steps:")
        print("  1. Test on more complex tasks (multi-tool ReAct)")
        print("  2. Integrate with actual tool calling system")
        print("  3. Test natural language reasoning (not just symbolic)")
        print("  4. Compare performance to 7B models")
        print("  5. Optimize latency with batching/caching")
    else:
        print("⚠ v6-golden struggled with this task")
        print("⚠ May need more training or different prompting approach")
        print("\nConsider:")
        print("  - Fine-tune on ReAct-style reasoning examples")
        print("  - Add explicit planning/verification training")
        print("  - Hybrid approach (v6 for logic, v4 for speed)")
    
    print("\n" + "="*60)
    print("THE φ ≈ 0.60 QUESTION")
    print("="*60 + "\n")
    
    print("Interesting observation: The model's optimal balance point")
    print("(φ ≈ 0.60) might extend to ReAct loop performance:")
    print("  - Fast enough for iterative reasoning (vs v5b)")
    print("  - Accurate enough for reliable convergence (vs v4)")
    print("  - Balanced for SUSTAINED recursive processing")
    print("\nThe golden ratio might be optimal not just for training,")
    print("but for the ARCHITECTURE of recursive reasoning itself! 🌀")

if __name__ == "__main__":
    main()

