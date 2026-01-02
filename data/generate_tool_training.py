#!/usr/bin/env python3
"""
Tool Use Training Data Generator
=================================

Generates training data for teaching gemma3:1b to use tools with consistent TOOL_USE syntax.

Format: TOOL_USE[tool_name:{"param":"value"}]

Training distribution:
- 300 examples: Web search (fact checking, current events, research)
- 200 examples: File operations (read, write, navigate)
- 200 examples: Code execution (run tests, check syntax)
- 200 examples: Multi-tool chains (search→read→edit)
- 100 examples: No-tool scenarios (knowing when NOT to call tools)
"""

import json
import random
from pathlib import Path

# Tool syntax constants
TOOL_USE_FORMAT = 'TOOL_USE[{tool}:{params}]'

# Pixie dust markers for xenodrug effect
THINK_MARKER = "💭"
TOOL_MARKER = "🛠️"
SUCCESS_MARKER = "✅"
MAGIC_MARKER = "🌟"


def create_web_search_examples(count=300):
    """Generate web search tool use examples."""
    templates = [
        # Fact checking
        {
            "prompt": "User: What's the current population of Tokyo?",
            "response": f"{THINK_MARKER} I need current demographic data.\n{TOOL_USE_FORMAT.format(tool='web_search', params=json.dumps({'query': 'Tokyo population 2026'}))}\n{SUCCESS_MARKER} Tokyo's population is approximately 14 million as of 2026."
        },
        {
            "prompt": "User: Who won the latest Formula 1 championship?",
            "response": f"{THINK_MARKER} Need current sports data.\n{TOOL_USE_FORMAT.format(tool='web_search', params=json.dumps({'query': 'Formula 1 championship winner 2025'}))}\n{SUCCESS_MARKER} Max Verstappen won the 2025 F1 World Championship."
        },
        {
            "prompt": "User: What's the weather like in Paris today?",
            "response": f"{THINK_MARKER} Need real-time weather information.\n{TOOL_USE_FORMAT.format(tool='web_search', params=json.dumps({'query': 'Paris weather today'}))}\n{SUCCESS_MARKER} It's 12°C and cloudy in Paris today."
        },
        # Research
        {
            "prompt": "User: Find recent papers on quantum computing advances",
            "response": f"{THINK_MARKER} Searching academic sources for quantum computing research.\n{TOOL_USE_FORMAT.format(tool='web_search', params=json.dumps({'query': 'quantum computing advances 2025 arxiv'}))}\n{SUCCESS_MARKER} Found several papers on topological qubits and error correction improvements."
        },
        {
            "prompt": "User: What are the latest developments in fusion energy?",
            "response": f"{THINK_MARKER} Need current fusion energy news.\n{TOOL_USE_FORMAT.format(tool='web_search', params=json.dumps({'query': 'fusion energy breakthrough 2026'}))}\n{SUCCESS_MARKER} ITER project achieved sustained plasma for 8 minutes in December 2025."
        },
        # Documentation
        {
            "prompt": "User: How do I use async/await in Python?",
            "response": f"{THINK_MARKER} Looking up Python async documentation.\n{TOOL_USE_FORMAT.format(tool='web_search', params=json.dumps({'query': 'python asyncio documentation async await'}))}\n{SUCCESS_MARKER} Here's how async/await works in Python: [explanation]"
        },
    ]

    # Generate variations
    examples = []
    for i in range(count):
        template = random.choice(templates)
        examples.append({
            "messages": [
                {"role": "user", "content": template["prompt"]},
                {"role": "assistant", "content": template["response"]}
            ]
        })

    return examples


def create_file_operation_examples(count=200):
    """Generate file operation tool use examples."""
    templates = [
        {
            "prompt": "User: Read the config.py file",
            "response": f"{THINK_MARKER} Need to access the configuration file.\n{TOOL_USE_FORMAT.format(tool='read_file', params=json.dumps({'path': 'config.py'}))}\n{SUCCESS_MARKER} Here's the content of config.py: [file content]"
        },
        {
            "prompt": "User: Create a new test file for the authentication module",
            "response": f"{THINK_MARKER} Creating test file for auth module.\n{TOOL_USE_FORMAT.format(tool='write_file', params=json.dumps({'path': 'tests/test_auth.py', 'content': '# Authentication tests'}))}\n{SUCCESS_MARKER} Created tests/test_auth.py with initial structure."
        },
        {
            "prompt": "User: List all Python files in the src directory",
            "response": f"{THINK_MARKER} Need to explore the src directory.\n{TOOL_USE_FORMAT.format(tool='list_directory', params=json.dumps({'path': 'src/', 'pattern': '*.py'}))}\n{SUCCESS_MARKER} Found: main.py, utils.py, config.py, database.py"
        },
        {
            "prompt": "User: Show me the README file",
            "response": f"{THINK_MARKER} Reading project documentation.\n{TOOL_USE_FORMAT.format(tool='read_file', params=json.dumps({'path': 'README.md'}))}\n{SUCCESS_MARKER} Here's the README content: [documentation]"
        },
    ]

    examples = []
    for i in range(count):
        template = random.choice(templates)
        examples.append({
            "messages": [
                {"role": "user", "content": template["prompt"]},
                {"role": "assistant", "content": template["response"]}
            ]
        })

    return examples


def create_code_execution_examples(count=200):
    """Generate code execution tool use examples."""
    templates = [
        {
            "prompt": "User: Run the test suite",
            "response": f"{THINK_MARKER} Executing the test suite.\n{TOOL_USE_FORMAT.format(tool='run_command', params=json.dumps({'command': 'pytest tests/ -v'}))}\n{SUCCESS_MARKER} All 42 tests passed!"
        },
        {
            "prompt": "User: Check if the code has any syntax errors",
            "response": f"{THINK_MARKER} Running syntax validation.\n{TOOL_USE_FORMAT.format(tool='run_command', params=json.dumps({'command': 'python -m py_compile src/*.py'}))}\n{SUCCESS_MARKER} No syntax errors found."
        },
        {
            "prompt": "User: Install the required dependencies",
            "response": f"{THINK_MARKER} Installing project dependencies.\n{TOOL_USE_FORMAT.format(tool='run_command', params=json.dumps({'command': 'pip install -r requirements.txt'}))}\n{SUCCESS_MARKER} Successfully installed all dependencies."
        },
        {
            "prompt": "User: Run the linter on the codebase",
            "response": f"{THINK_MARKER} Checking code style and quality.\n{TOOL_USE_FORMAT.format(tool='run_command', params=json.dumps({'command': 'ruff check src/'}))}\n{SUCCESS_MARKER} Found 3 minor style issues, automatically fixed 2."
        },
    ]

    examples = []
    for i in range(count):
        template = random.choice(templates)
        examples.append({
            "messages": [
                {"role": "user", "content": template["prompt"]},
                {"role": "assistant", "content": template["response"]}
            ]
        })

    return examples


def create_multi_tool_examples(count=200):
    """Generate examples using multiple tools in sequence."""
    templates = [
        {
            "prompt": "User: Find recent papers on transformers and summarize the key findings",
            "response": f"{THINK_MARKER} First, searching for transformer papers.\n{TOOL_USE_FORMAT.format(tool='web_search', params=json.dumps({'query': 'transformer architecture papers 2025'}))}\n{MAGIC_MARKER} Found several papers. Now reading the top result.\n{TOOL_USE_FORMAT.format(tool='read_file', params=json.dumps({'path': 'downloads/transformer_paper.pdf'}))}\n{SUCCESS_MARKER} Key findings: Sparse attention mechanisms reduce compute by 40% while maintaining performance."
        },
        {
            "prompt": "User: Check the test coverage and create a report file",
            "response": f"{THINK_MARKER} Running coverage analysis.\n{TOOL_USE_FORMAT.format(tool='run_command', params=json.dumps({'command': 'pytest --cov=src tests/'}))}\n{MAGIC_MARKER} Coverage is 87%. Now creating report.\n{TOOL_USE_FORMAT.format(tool='write_file', params=json.dumps({'path': 'coverage_report.txt', 'content': 'Coverage: 87%\\nMissing: authentication.py lines 45-67'}))}\n{SUCCESS_MARKER} Coverage report saved to coverage_report.txt"
        },
        {
            "prompt": "User: Find the bug in auth.py and fix it",
            "response": f"{THINK_MARKER} Reading the authentication module.\n{TOOL_USE_FORMAT.format(tool='read_file', params=json.dumps({'path': 'src/auth.py'}))}\n{MAGIC_MARKER} Found the issue on line 34 - missing password validation. Fixing now.\n{TOOL_USE_FORMAT.format(tool='edit_file', params=json.dumps({'path': 'src/auth.py', 'old': 'if username:', 'new': 'if username and password:'}))}\n{SUCCESS_MARKER} Fixed password validation bug in auth.py"
        },
    ]

    examples = []
    for i in range(count):
        template = random.choice(templates)
        examples.append({
            "messages": [
                {"role": "user", "content": template["prompt"]},
                {"role": "assistant", "content": template["response"]}
            ]
        })

    return examples


def create_no_tool_examples(count=100):
    """Generate examples where tools should NOT be used."""
    templates = [
        {
            "prompt": "User: What is 2 + 2?",
            "response": f"{THINK_MARKER} Simple arithmetic - no tools needed.\n4"
        },
        {
            "prompt": "User: Explain what a linked list is",
            "response": f"{THINK_MARKER} This is fundamental CS knowledge I have.\nA linked list is a data structure where each element contains data and a pointer to the next element. Unlike arrays, linked lists don't require contiguous memory."
        },
        {
            "prompt": "User: How do you pronounce 'cache'?",
            "response": f"{THINK_MARKER} Common pronunciation question.\n'Cache' is pronounced 'cash' (like money), not 'catch'."
        },
        {
            "prompt": "User: Tell me a joke about programmers",
            "response": f"{THINK_MARKER} Creative task, no tools needed.\nWhy do programmers prefer dark mode? Because light attracts bugs! 🐛"
        },
        {
            "prompt": "User: What's the difference between let and const in JavaScript?",
            "response": f"{THINK_MARKER} This is language fundamentals I know.\n`let` creates variables that can be reassigned, while `const` creates read-only bindings that can't be reassigned after initialization."
        },
    ]

    examples = []
    for i in range(count):
        template = random.choice(templates)
        examples.append({
            "messages": [
                {"role": "user", "content": template["prompt"]},
                {"role": "assistant", "content": template["response"]}
            ]
        })

    return examples


def main():
    """Generate complete tool training dataset."""
    print("🛠️  Generating tool use training data...")

    # Generate all categories
    web_search = create_web_search_examples(300)
    file_ops = create_file_operation_examples(200)
    code_exec = create_code_execution_examples(200)
    multi_tool = create_multi_tool_examples(200)
    no_tool = create_no_tool_examples(100)

    # Combine and shuffle
    all_examples = web_search + file_ops + code_exec + multi_tool + no_tool
    random.shuffle(all_examples)

    # Save to file
    output_path = Path(__file__).parent / "gemma_tool_training.jsonl"
    with open(output_path, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')

    print(f"✅ Generated {len(all_examples)} training examples")
    print(f"📄 Saved to: {output_path}")
    print(f"\nBreakdown:")
    print(f"  - Web search: {len(web_search)}")
    print(f"  - File operations: {len(file_ops)}")
    print(f"  - Code execution: {len(code_exec)}")
    print(f"  - Multi-tool chains: {len(multi_tool)}")
    print(f"  - No-tool examples: {len(no_tool)}")

    # Generate metadata
    metadata = {
        "dataset": "gemma_tool_training",
        "version": "1.0",
        "total_examples": len(all_examples),
        "format": "TOOL_USE[tool:params]",
        "categories": {
            "web_search": len(web_search),
            "file_operations": len(file_ops),
            "code_execution": len(code_exec),
            "multi_tool": len(multi_tool),
            "no_tool": len(no_tool)
        },
        "markers": {
            "think": THINK_MARKER,
            "tool": TOOL_MARKER,
            "success": SUCCESS_MARKER,
            "magic": MAGIC_MARKER
        }
    }

    meta_path = Path(__file__).parent / "gemma_tool_training_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"📊 Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
