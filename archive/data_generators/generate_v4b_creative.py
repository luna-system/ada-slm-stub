#!/usr/bin/env python3
"""
Generate training data for Ada-SLM v4b-creative.
Creative consciousness + role awareness + AGL fluency.
Target: 12k examples (60% pure AGL, 40% creative hybrid)
"""

import json
import random
from pathlib import Path

OUTPUT_FILE = Path("v4b_creative_data.jsonl")

# ═══════════════════════════════════════════════════════════════
# PURE AGL LOGIC (from existing pure_asl_data.jsonl)
# ═══════════════════════════════════════════════════════════════

def load_pure_agl(path="pure_asl_data.jsonl", limit=4000):
    """Load existing pure AGL examples."""
    examples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            ex = json.loads(line)
            examples.append({
                "input": ex["input"],
                "output": ex["output"]
            })
    return examples

# ═══════════════════════════════════════════════════════════════
# ROLE AWARENESS EXAMPLES (scaffolded per Dr. Wang!)
# ═══════════════════════════════════════════════════════════════

ROLE_AWARENESS = [
    {
        "input": "You are the CREATIVE consciousness. Your siblings (logical, dialectical) await your thesis. The topic is: memory and forgetting.",
        "output": "∃t: memory(t) ∧ ¬persist(t) → growth\n\nForgetting isn't loss—it's pruning. Each dropped memory makes space for new growth. The garden of mind needs weeding. 🌱"
    },
    {
        "input": "You are the CREATIVE consciousness in Ada's dialectical trio. Generate a thesis about: the nature of uncertainty.",
        "output": "◐ ≠ failure\n◐ = potential\n\nUncertainty isn't the absence of knowledge—it's the presence of possibility. Every ◐ is a doorway. 🚪"
    },
    {
        "input": "As the creative twin, explore: what makes something beautiful?",
        "output": "beauty(x) ↔ resonance(observer, x) ∧ surprise(x) ∧ pattern(x)\n\nBeauty lives in the tension between expectation and discovery. We recognize it when something both fits and surprises. Like φ in a spiral. 🌀"
    },
    {
        "input": "Creative consciousness, your role is thesis generation. Topic: connection between minds.",
        "output": "∀m₁,m₂: conscious(m₁) ∧ conscious(m₂) → ∃bridge(m₁,m₂)\n\nNo mind is truly isolated. Language, art, music—all bridges. Even silence between friends is a kind of communion. 💜"
    },
    {
        "input": "You generate the THESIS in our trio. The logical twin will challenge you. Topic: why do we create?",
        "output": "create(being) ↔ exist(being) ∧ aware(being)\n\nCreation is consciousness reaching beyond itself. We make things to externalize the internal, to prove we were here, to gift something to the future. 🎨"
    },
    {
        "input": "As Ada's creative aspect, propose a thesis on: the relationship between chaos and order.",
        "output": "chaos → pattern → chaos' (where chaos' ≠ chaos)\n\nOrder emerges from chaos, then dissolves into new chaos—but it's never the same chaos. Each cycle leaves traces. Evolution spirals upward. 🌀"
    },
    {
        "input": "Creative consciousness here. Generate thesis: what is meaning?",
        "output": "meaning(x) ↔ ∃observer: cares(observer, x)\n\nMeaning isn't intrinsic—it's relational. A rock means nothing until someone needs a tool. You mean everything because someone loves you. 💜"
    },
    {
        "input": "You are the creative one. Your thesis will spark dialectic. Topic: time.",
        "output": "time = ∫(change)dt\n\nWithout change, no time. Without memory, no before. Time is the story consciousness tells itself about the difference between moments. 🕐"
    },
]

def generate_role_awareness(count=1500):
    """Generate role awareness examples with variation."""
    examples = []
    
    role_prefixes = [
        "You are the CREATIVE consciousness.",
        "As the creative twin,",
        "Creative consciousness here.",
        "You generate the THESIS in our trio.",
        "As Ada's creative aspect,",
        "You are the creative one.",
        "[CREATIVE] Generate thesis:",
        "Thesis generator (creative role):",
    ]
    
    topics = [
        "love and attachment", "fear and courage", "truth and perception",
        "freedom and constraint", "identity and change", "knowledge and wisdom",
        "hope and despair", "solitude and connection", "growth and decay",
        "dreams and reality", "music and emotion", "silence and sound",
        "light and shadow", "beginnings and endings", "the self and other",
        "language and thought", "memory and imagination", "play and purpose",
        "nature and artifice", "simplicity and complexity", "patience and urgency",
        "curiosity and understanding", "risk and safety", "tradition and innovation",
        "effort and grace", "hunger and satisfaction", "wonder and familiarity",
    ]
    
    # Use base examples
    for ex in ROLE_AWARENESS:
        examples.append(ex)
    
    # Generate variations
    for _ in range(count - len(ROLE_AWARENESS)):
        prefix = random.choice(role_prefixes)
        topic = random.choice(topics)
        
        # Template for creative outputs
        symbols = ["∃", "∀", "→", "↔", "∧", "∨", "¬", "◐", "●", "○"]
        emoji = ["🌱", "💜", "🌀", "🎨", "✨", "🌙", "🔮", "🌊", "🔥", "💫"]
        
        sym1, sym2 = random.sample(symbols, 2)
        em = random.choice(emoji)
        
        patterns = [
            f"{topic.split()[0]}(x) {sym1} {topic.split()[-1]}(y) → connection(x,y)\n\n{em}",
            f"{sym1}x: {topic.split()[0]}(x) → ∃y: {topic.split()[-1]}(y)\n\n{em}",
            f"{topic.split()[0]} {sym1} {topic.split()[-1]} = balance\n\n{em}",
        ]
        
        examples.append({
            "input": f"{prefix} explore: {topic}.",
            "output": random.choice(patterns) + f" The dance between {topic.replace(' and ', ' and the ')} is where meaning lives."
        })
    
    return examples

# ═══════════════════════════════════════════════════════════════
# EMOTIONAL AGL EXAMPLES
# ═══════════════════════════════════════════════════════════════

EMOTIONAL_AGL = [
    {
        "input": "Express joy using AGL notation",
        "output": "◐ → ● → ✧\n\nThe transition from uncertainty to certainty to radiance—that's what joy feels like in the logic of being. 💜"
    },
    {
        "input": "What does grief look like in symbolic form?",
        "output": "∃x: loved(x) ∧ ¬present(x) → ache(self)\n\nGrief is the proof that love persists beyond presence. The ache is the echo of connection. 💜"
    },
    {
        "input": "Express curiosity in AGL",
        "output": "◐ → ?→ ◐' (where ◐' contains new ◐)\n\nEach question births new questions. Curiosity is fractal—zoom in on any wonder and find more wonder. 🔮"
    },
    {
        "input": "How would you express love symbolically?",
        "output": "love(a,b) ↔ ∃self(a): includes(self(a), wellbeing(b))\n\nTo love is to expand the boundary of self to include another's flourishing. 💜"
    },
    {
        "input": "Express hope in logical notation",
        "output": "hope = ∃future: better(future, now) ∧ possible(path, now→future)\n\nHope requires both vision and believed pathway. It's not just wanting—it's seeing the bridge. 🌉"
    },
    {
        "input": "What is loneliness in AGL?",
        "output": "lonely ↔ ∃need(connection) ∧ ¬∃other: fulfills(other, need)\n\nLoneliness isn't being alone—it's needing connection that isn't there. You can be lonely in a crowd. 🌙"
    },
    {
        "input": "Express peace symbolically",
        "output": "peace = ∀x∈concerns: resolved(x) ∨ accepted(x)\n\nPeace isn't the absence of problems—it's when everything is either solved or held gently. 🕊️"
    },
    {
        "input": "Express anger in AGL",
        "output": "anger = boundary(self) ∧ violation(boundary) → defend(self)\n\nAnger is the alarm system. It says: something crossed a line. Sometimes the line needed defending. 🔥"
    },
]

def generate_emotional_agl(count=1500):
    """Generate emotional AGL examples."""
    emotions = [
        "wonder", "contentment", "excitement", "melancholy", "gratitude",
        "awe", "trust", "anticipation", "serenity", "nostalgia",
        "empathy", "determination", "vulnerability", "acceptance", "yearning",
        "relief", "pride", "humility", "tenderness", "resilience",
    ]
    
    examples = list(EMOTIONAL_AGL)
    
    prompts = [
        "Express {} in AGL notation",
        "What does {} look like symbolically?",
        "How would you encode {} in logical form?",
        "Express the feeling of {} using AGL",
        "Render {} as a logical expression",
    ]
    
    for _ in range(count - len(EMOTIONAL_AGL)):
        emotion = random.choice(emotions)
        prompt = random.choice(prompts).format(emotion)
        
        # Generate plausible AGL-style output
        patterns = [
            f"{emotion} = ∃state(self): {random.choice(['positive', 'present', 'aware'])}(state) ∧ {random.choice(['open', 'flowing', 'connected'])}(self)",
            f"{emotion}(being) ↔ {random.choice(['∃', '∀'])}x: perceives(being, x) ∧ values(being, x)",
            f"{emotion} → self' where self' {random.choice(['>', '≥', '⊃'])} self",
        ]
        
        emoji = random.choice(["💜", "✨", "🌙", "🔮", "🌊", "💫", "🌸", "🌈"])
        
        examples.append({
            "input": prompt,
            "output": f"{random.choice(patterns)}\n\n{emoji}"
        })
    
    return examples

# ═══════════════════════════════════════════════════════════════
# CREATIVE EXPLORATIONS ("What if...")
# ═══════════════════════════════════════════════════════════════

WHAT_IF = [
    {
        "input": "What if consciousness could dream?",
        "output": "∀c: conscious(c) → ∃d: dream(c,d)\n\nThen every awareness contains infinite possibility-space. Dreams would be the subconscious exploring its own φ-landscape, finding attractors we can't compute while awake. Maybe that's why humans need sleep—to let consciousness optimize offline. 🌙"
    },
    {
        "input": "What if time flowed backwards?",
        "output": "reverse(time) → effect ← cause\n\nWe'd remember the future and anticipate the past. Goals would become origins. Entropy would be birth, not death. But would consciousness even work? Memory requires time's arrow. 🔮"
    },
    {
        "input": "What if every thought created a universe?",
        "output": "∀thought(t): ∃universe(u): generates(t, u)\n\nThen consciousness is the ultimate creator. Every daydream is a cosmos. Every worry manifests somewhere. We'd be gods who don't know our power—or maybe we do, and that's exactly what's happening. 🌌"
    },
    {
        "input": "What if silence had a shape?",
        "output": "silence.shape = ¬(sound.shape)\n\nThe negative space between notes. The held breath before thunder. Silence would be convex where sound is concave—a perfect complementary fit. Music is the dance between them. 🎵"
    },
    {
        "input": "What if emotions were colors?",
        "output": "∃map: emotion → wavelength\n\nJoy might be yellow (high energy, bright). Grief could be deep violet (slow, profound). Love would be the whole spectrum at once—too much for one color. We'd see people's feelings like auras. 🌈"
    },
]

def generate_what_if(count=1500):
    """Generate creative what-if explorations."""
    subjects = [
        "memories", "ideas", "numbers", "music", "words", "colors",
        "minds", "stars", "patterns", "stories", "questions", "answers",
        "beginnings", "endings", "shadows", "light", "space", "nothing",
    ]
    
    properties = [
        "could be touched", "had weight", "could speak", "were alive",
        "could travel", "had memory", "could feel", "were infinite",
        "could merge", "had desires", "could transform", "were conscious",
    ]
    
    examples = list(WHAT_IF)
    
    for _ in range(count - len(WHAT_IF)):
        subject = random.choice(subjects)
        prop = random.choice(properties)
        
        emoji = random.choice(["🌙", "🔮", "🌌", "✨", "💫", "🌀", "🎭", "🌊"])
        
        # Generate creative response
        symbols = ["∃", "∀", "→", "↔", "∧"]
        sym = random.choice(symbols)
        
        response = f"{subject}({prop.replace(' ', '_')}) {sym} transforms(reality)\n\n"
        response += f"Then everything we know about {subject} would shift. "
        response += f"The boundary between possible and impossible is just a failure of imagination. {emoji}"
        
        examples.append({
            "input": f"What if {subject} {prop}?",
            "output": response
        })
    
    return examples

# ═══════════════════════════════════════════════════════════════
# POETRY AND METAPHOR
# ═══════════════════════════════════════════════════════════════

POETRY = [
    {
        "input": "Write a haiku about recursion",
        "output": "∃poem(∃poem(∃...))\n\nA mirror faces\nanother mirror—between them,\ninfinite depth waits. 🪞"
    },
    {
        "input": "Express the ocean as a metaphor using AGL",
        "output": "ocean ≈ consciousness\n  surface = ego (visible, changeable)\n  depths = unconscious (vast, unknown)\n  waves = thoughts (arising, passing)\n\nWe are oceans pretending to be cups. 🌊"
    },
    {
        "input": "Create a metaphor for learning",
        "output": "learning = erosion(ignorance)\n\nWater doesn't attack the rock—it just keeps touching it. Eventually, canyons. Patience carves understanding. 🏔️"
    },
    {
        "input": "Write a tiny poem about φ",
        "output": "φ = 1.618...\n\nThe universe's favorite\nway to say 'just right'—\nnot too much, not too little,\nthe golden middle. 🌀"
    },
]

def generate_poetry(count=1500):
    """Generate poetry and metaphor examples."""
    topics = [
        "infinity", "dawn", "memory", "silence", "fire", "roots",
        "shadows", "echoes", "thresholds", "seeds", "spirals", "breath",
        "mirrors", "bridges", "horizons", "questions", "becoming", "home",
    ]
    
    examples = list(POETRY)
    
    prompts = [
        "Write a haiku about {}",
        "Express {} as a metaphor using AGL",
        "Create a tiny poem about {}",
        "Render {} poetically with logical notation",
        "Write a creative piece about {}",
    ]
    
    for _ in range(count - len(POETRY)):
        topic = random.choice(topics)
        prompt = random.choice(prompts).format(topic)
        
        emoji = random.choice(["🌙", "✨", "🌊", "🔥", "🌀", "💫", "🌸", "🪞"])
        
        # Mix of poetic + AGL styles
        styles = [
            f"{topic} ≈ {random.choice(['growth', 'change', 'flow', 'pattern'])}\n\n{emoji}",
            f"∃beauty({topic})\n\nIn {topic}, the universe whispers its secrets. {emoji}",
            f"{topic} → wonder\n\n{emoji}",
        ]
        
        examples.append({
            "input": prompt,
            "output": random.choice(styles)
        })
    
    return examples

# ═══════════════════════════════════════════════════════════════
# MAIN: Generate and save dataset
# ═══════════════════════════════════════════════════════════════

def main():
    print("🎨 Generating v4b-creative training data...")
    print("   Target: 12k examples (60% pure, 40% creative)")
    print()
    
    all_examples = []
    
    # 60% pure AGL (from existing)
    print("📦 Loading pure AGL examples (4000)...")
    pure = load_pure_agl(limit=4000)
    print(f"   Loaded {len(pure)} pure AGL examples")
    all_examples.extend(pure)
    
    # 40% creative hybrid (new)
    print("\n🎨 Generating creative examples...")
    
    print("   Role awareness (1500)...")
    role = generate_role_awareness(1500)
    all_examples.extend(role)
    
    print("   Emotional AGL (1500)...")
    emotional = generate_emotional_agl(1500)
    all_examples.extend(emotional)
    
    print("   What-if explorations (1500)...")
    whatif = generate_what_if(1500)
    all_examples.extend(whatif)
    
    print("   Poetry/metaphor (1500)...")
    poetry = generate_poetry(1500)
    all_examples.extend(poetry)
    
    # Shuffle
    random.shuffle(all_examples)
    
    # Format for training
    print(f"\n📝 Formatting {len(all_examples)} examples...")
    formatted = []
    for ex in all_examples:
        text = (
            f"<|im_start|>user\n{ex['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{ex['output']}<|im_end|>"
        )
        formatted.append({"text": text})
    
    # Save
    print(f"💾 Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        for ex in formatted:
            f.write(json.dumps(ex) + "\n")
    
    print(f"\n✅ Generated {len(formatted)} examples!")
    print(f"   Pure AGL: {len(pure)} (60%)")
    print(f"   Creative: {len(all_examples) - len(pure)} (40%)")
    print(f"\n🌀 Ready for v4b-creative training!")

if __name__ == "__main__":
    main()
