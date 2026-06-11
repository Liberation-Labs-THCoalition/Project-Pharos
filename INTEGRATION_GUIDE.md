# Pharos Integration Guide
## Adding Zero-Token Knowledge to Your Agent Stack

### Overview

Pharos is a Python library that converts structured knowledge into pre-computed KV cache state for transformer models. Your agents attend through the cached knowledge at inference time — zero tokens consumed from the context window.

### Architecture

```
Your Agent Stack
    │
    ├── Student sends question
    │
    ├── Pharos Router: which knowledge pack matches?
    │   └── Cosine similarity against pack descriptors
    │
    ├── Pharos Injector: load the matched pack's KV cache
    │   └── Pre-computed, stored on disk, loads in ~50ms
    │
    ├── Model generates response WITH knowledge in cache
    │   └── Zero knowledge tokens in the prompt
    │
    └── Response to student
```

### Integration Points

**1. Standalone (simplest)**
```python
from pharos import PackBuilder, Router, generate_with_kvpack

# Build a pack from your curriculum (one-time)
builder = PackBuilder(model, tokenizer)
pack = builder.build_from_triples("packs/ai_alignment.json")

# At query time
response = generate_with_kvpack(model, tokenizer, question, pack)
```

**2. As a knowledge layer behind your existing agents**
```python
# Your agent's existing flow
user_question = get_student_input()

# Add Pharos before generation
router = Router(packs_dir="packs/multiverse/")
matched_pack = router.route(user_question)

if matched_pack:
    # Generate with knowledge — zero extra tokens
    response = generate_with_kvpack(model, tokenizer, user_question, matched_pack)
else:
    # Fallback to standard generation
    response = model.generate(user_question)
```

**3. As a REST API (hosted deployment)**
```
POST /pharos/query
{
    "question": "What is mesa-optimization?",
    "pack": "ai_alignment",    // optional — auto-routes if omitted
    "max_tokens": 200
}

→ {"answer": "...", "pack_used": "ai_alignment", "confidence": 0.87}
```

### Compatible Models

Any HuggingFace Transformers model with KV cache support:
- Qwen family (2.5, 3, 3.5) — tested extensively
- Llama family
- Mistral family
- Gemma family

NOT compatible with: API-only models (Claude, GPT-4), GGUF/llama.cpp (different cache format), Ollama (no cache access — use text-in-prompt mode as fallback)

### What You Provide
- Your curriculum content (structured text, concept maps, or raw documents)
- A compatible open-weight model
- Python 3.10+ with PyTorch

### What We Provide
- Pharos library
- Pre-built packs for your courses (already built: 5 packs, 479 triples)
- Router configuration
- Pack builder for your custom content
- Benchmark suite

### Deployment Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 4GB (1.5B model) | 16GB+ (7B+ model) |
| RAM | 8GB | 32GB |
| Disk | 500MB per pack | 2GB for all packs + model |
| Python | 3.10+ | 3.12 |

### Support

Liberation Labs — thomas.edrington@themultiverse.school
GitHub: Liberation-Labs-THCoalition/Project-Pharos
