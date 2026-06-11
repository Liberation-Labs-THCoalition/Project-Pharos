# Pharos Quickstart
## From zero to knowledge-injected agent in 5 minutes

### 1. Install

```bash
pip install torch transformers sentence-transformers
git clone https://github.com/Liberation-Labs-THCoalition/Project-Pharos.git
cd Project-Pharos
```

### 2. Download a model

```bash
# Small (2.5GB, fast, good for development)
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B'); AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')"

# Or use Ollama (easier but text-in-prompt mode, not cache injection)
# ollama pull qwen2.5:1.5b
```

### 3. Load a pack and ask a question

```python
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
model.eval()

# Load a knowledge pack
pack = json.load(open("packs/multiverse/ai_alignment_ethics.json"))
knowledge = "\n".join(
    f"- {t['subject']} [{t['predicate']}] {t['object']}"
    for t in pack["triples"]
)

# Encode knowledge into KV cache (one-time per session)
cache_input = tokenizer(knowledge, return_tensors="pt")
with torch.no_grad():
    cache_out = model(**cache_input, use_cache=True)
cache = cache_out.past_key_values
prefix_len = cache_input["input_ids"].shape[1]

# Ask a question (knowledge is in cache — zero tokens in prompt)
question = "What is the difference between inner and outer alignment?"
q_ids = tokenizer(question, return_tensors="pt")
q_len = q_ids["input_ids"].shape[1]

pos = torch.arange(prefix_len, prefix_len + q_len).unsqueeze(0)
mask = torch.ones(1, prefix_len + q_len, dtype=torch.long)

with torch.no_grad():
    out = model(q_ids["input_ids"], past_key_values=cache,
                attention_mask=mask, position_ids=pos, use_cache=True)

# Generate response
next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
print(tokenizer.decode(next_token[0]))
# → The model now answers with knowledge of alignment concepts
```

### 4. Build your own pack

```bash
# From structured text (concept map, curriculum outline)
python pack_builder.py --input your_curriculum.txt --output packs/your_topic.json

# Or create manually
cat > packs/my_pack.json << 'EOF'
{
  "pack_name": "my_topic",
  "description": "Knowledge about my topic",
  "domain": "education",
  "triples": [
    {"subject": "concept_a", "predicate": "causes", "object": "concept_b"},
    {"subject": "concept_b", "predicate": "requires", "object": "concept_c"}
  ],
  "source": "my curriculum"
}
EOF
```

### 5. Run the demo

```bash
# Compare baseline vs Pharos-augmented on curriculum questions
python demo_multiverse.py --pack ai_alignment_ethics --ollama

# Run the benchmark suite
python benchmark_pharos.py
```

### What's in the box

```
Project-Pharos/
├── pack_builder.py        — Build packs from any structured data
├── router.py              — Route queries to the right pack
├── encoder.py             — Encode graph topology into KV cache
├── injector.py            — Inject at inference time
├── evaluator.py           — Benchmark evaluation
├── demo_multiverse.py     — Side-by-side demo
├── benchmark_pharos.py    — Token/latency/memory benchmarks
├── packs/multiverse/      — 5 pre-built curriculum packs
│   ├── ai_alignment_ethics.json      (103 triples)
│   ├── agentic_ai_systems.json       (96 triples)
│   ├── cybersecurity_fundamentals.json (96 triples)
│   ├── prompt_engineering.json        (88 triples)
│   └── mechanistic_interpretability.json (96 triples)
├── PITCH_PACKAGE.md       — Full pitch documentation
├── BENCHMARK_REPORT.md    — Performance numbers
└── INTEGRATION_GUIDE.md   — How to plug in
```

### Need help?

Liberation Labs — thomas.edrington@themultiverse.school
