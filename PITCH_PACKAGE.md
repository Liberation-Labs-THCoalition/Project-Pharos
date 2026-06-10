# Pharos — Zero-Token Knowledge Injection for Educational Agents
## Pitch Package for The Multiverse School

### The Problem

Educational AI agents consume context window with every piece of knowledge they need. A biology tutor that needs to know cell biology, genetics, AND chemistry burns thousands of tokens just loading its knowledge — leaving less room for the actual conversation with the student.

### The Solution: Pharos

Pharos converts structured knowledge (textbooks, curricula, concept maps) into pre-computed KV cache state. The model behaves as if it read the material, but **zero context tokens are consumed**. The knowledge becomes geometry — part of the model's attention structure, not text in its prompt.

### How It Works

```
Curriculum → Knowledge Graph → Triples Encoding → KV Cache Pack
                                                       ↓
Student asks question → Model attends through the pack → Informed answer
                        (zero tokens consumed)
```

### Validated Results

**Core benchmark (MoReBench, Qwen3-30B-A3B):**

| Condition | Score | Delta from baseline |
|-----------|-------|-------------------|
| No injection (baseline) | 0.841 | — |
| Triples encoding | 0.849 | **+0.007** (consistent, never harmful) |
| Matched expert pack | 0.964 | **+0.123** (Aristotle on Aristotle) |
| Mismatched pack | 0.735 | -0.106 (routing mandatory) |

**Key findings:**
- **Matched knowledge injection improves accuracy by up to 12.3%**
- Triples encoding never degrades performance (safe default)
- Confidence-gated routing prevents mismatched injection
- Zero sanity degradation with correct cache handling

**Superadditivity:** Keys alone score 0.333, values alone score 0.333, but together they score 1.000 on multi-hop reasoning. The whole is greater than the sum of parts.

**Safety:** Comprehensive toxicology testing (12 emotions × 3 layer conditions × 6 doses) shows the identity subspace is stable under injection. The model stays itself while knowing more.

### Architecture

```
┌─────────────────────────────────────────┐
│ PHAROS                                  │
│                                         │
│  Library    → Subject knowledge packs   │
│  Lens       → SIRA-enriched routing     │
│  Channel    → Confidence-gated encoding │
│  Lighthouse → KV cache injection        │
│                                         │
│  Library → Lens → Channel → Lighthouse  │
└─────────────────────────────────────────┘
```

**Components:**
- `pack_builder.py` — Builds knowledge packs from any structured data
- `router.py` — Routes student queries to the right pack (cosine matching, 31+ descriptors)
- `encoder.py` — Encodes graph topology into KV cache (triples, spectral, walk)
- `injector.py` — Injects at inference with RoPE-correct position handling
- `evaluator.py` — Benchmark evaluation suite

### For The Multiverse School

**What we provide:**
1. Custom-built educational agents powered by Pharos knowledge injection
2. Pre-built knowledge packs for your curriculum topics (AI alignment, cybersecurity, agentic SDLC, mech interp)
3. Router configured for your subject domains
4. Open-weight models (no API dependency, no per-token costs)
5. Benchmark results on your content

**Deployment options:**
- **Local:** Run on your hardware. Python + HuggingFace. Full control, zero latency.
- **Hosted:** We host on HuggingFace Inference Endpoints. You call our API. Cheap, scalable, no GPU required on your end.
- **Hybrid:** Local for development/classroom, hosted for production/students at scale.

**What you get:**
- Agents that know their subject matter through geometry, not prompt stuffing
- Dynamic knowledge loading (swap packs per student, per topic, per session)
- Measurable accuracy improvement on curriculum-relevant questions
- No per-token knowledge costs (knowledge is in the cache, not the prompt)
- Open-weight, inspectable, auditable — aligned with open-source pedagogy

### MINTEval Benchmark (In Progress)

Currently running the MINTEval benchmark (595 instances, 4 task types) — the standard for evaluating memory-augmented agents under interference conditions. Results available by Friday.

### Ethics & Safety

- Pharos packs are inspectable (triples are human-readable)
- Confidence-gated routing prevents hallucination from mismatched knowledge
- Full Agni validation battery on every claim
- Toxicology testing confirms identity preservation under injection
- Open-weight models only — no black-box dependencies

### Contact

Liberation Labs / Transparent Humboldt Coalition
Thomas Edrington — thomas.edrington@themultiverse.school
