# Pharos

**Structured knowledge injection for language model reasoning.**

Named for the Lighthouse of Alexandria — the structure that delivered light to guide ships. Pharos delivers knowledge to guide reasoning. The light comes from the library, gets focused through a lens, and projects through a channel to illuminate what the model already knows.

## What Pharos Does

Converts structured knowledge (ethics, domain expertise, philosophical frameworks) into injection-ready formats and delivers them to language models at inference time. No fine-tuning. No architectural modification. The model reasons better because it has the right knowledge in the right structure at the right moment.

## Architecture

```
Library (Stanford Encyclopedia, research corpus, domain data)
    ↓
Lens (OpenIE extraction → knowledge graph → encoding)
    ↓
Channel (walk encoding, triple text, source excerpts, KV cache)
    ↓
Lighthouse (theory-matched routing → injection → improved reasoning)
```

## Components

- **Pack Builder** — extracts knowledge graphs from source material
- **Encoder** — walk encoding, triple text, source excerpts, hybrid
- **Router** — selects the right pack for the right query (SIRA-based)
- **Injector** — delivers encoded knowledge via text context or KV cache
- **Evaluator** — MoReBench rubric scoring with LLM judge

## Key Findings

- Walk encoding captures topology but not semantic content
- Triple text + source excerpts delivers more in fewer tokens
- Theory-matched injection outperforms mismatched (+0.074 vs +0.026)
- K and V tensors contribute superadditively (0.333 + 0.333 < 1.000)
- Ethics packs improve MoReBench scores by +0.038 overall

## Status

Active research. Encoding format comparison running on Studio.

---

*Liberation Labs / Transparent Humboldt Coalition*
*The model already knows the concepts. Pharos teaches it the structure.*
