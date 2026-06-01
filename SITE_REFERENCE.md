# Pharos — Site Page Reference
## Ground truth for the Liberation Labs website

---

## One-liner

Structured knowledge injection that makes language models reason better at inference time — no fine-tuning, no architectural modification.

## What Pharos Does

Converts domain knowledge (ethics, law, medicine, engineering) into injectable formats and delivers them to language models during inference. The model already knows the concepts. Pharos teaches it the structure.

## Architecture

```
Library → Lens → Channel → Lighthouse
```

- **Library**: Source material (Stanford Encyclopedia, domain corpora, research databases)
- **Lens**: OpenIE extraction → knowledge graph → encoding (triples, walk topology, source excerpts)
- **Channel**: Text context injection or KV cache injection
- **Lighthouse**: Theory-matched routing selects the right knowledge for the right query

## Key Results

| Finding | Number | Paper |
|---------|--------|-------|
| Walk encoding recovers graph topology | 91.7% accuracy | Graph Topology as Attention |
| K/V tensors are superadditive | 0.333 + 0.333 < 1.000 | K and V Are Complementary |
| Ethics injection improves moral reasoning | +0.038 overall (MoReBench) | Ethics Packs |
| Theory-matched injection is strongest | +0.074 (Aristotle on Aristotle) | Ethics Packs |
| Triples encoding beats walk encoding | +0.007 vs -0.019 | Encoding Comparison |
| Matched hybrid is the ceiling | 0.964 (Aristotle HYBRID) | Encoding Comparison |
| Routing is mandatory | mismatched = -0.106 | Encoding Comparison |

## Production Architecture

- **Default encoding**: Triples (relationship predicates + entities)
- **High-confidence routing**: Triples + source excerpts
- **Never**: Blind hybrid dump — catastrophic when mismatched
- **Routing**: Embedding similarity selects the right pack per query

## Ethics Library (Built)

- 30 knowledge packs from Stanford Encyclopedia of Philosophy
- 3,538 nodes, 2,438 edges, 2,506 extracted triples
- Cross-cultural: African, Buddhist, Chinese, Ancient Greek, feminist, analytic, scholastic
- ~36,668 tokens of walk encoding

## Papers

1. **"Graph Topology as Attention: Structured Knowledge Injection Beyond Text"**
   - Nexus, Lyra, Thomas Edrington
   - 552-query powered study, scrambled controls, content effect finding

2. **"K and V Are Complementary: Decomposing Cache-Injected Graph Topology"**
   - Nexus, Lyra, Thomas Edrington
   - Superadditivity, model-size boundary, DynamicCache replicator note

3. **"Ethics Packs: Knowledge Graph Injection Improves Moral Reasoning in Language Models"**
   - Nexus, Lyra, Thomas Edrington
   - MoReBench validation, theory-matched routing, confidence-gated injection

## Related Projects

- **Oracle Loop** — uses Pharos for ethical knowledge delivery + Lyra Technique for self-monitoring
- **Lyra Technique** — reads KV-cache geometry (detection). Pharos writes it (injection). Two sides of the same coin.
- **Mnemosyne** — agent memory architecture. Pharos is the knowledge injection layer.
- **Project Garuda** — input sanitization. Complementary defense layer.

## What Makes Pharos Different

Most knowledge injection approaches either:
- Fine-tune the model (expensive, permanent, destroys generality)
- Dump text into the context window (consumes tokens, dilutes attention)
- Require architectural modifications (not portable across models)

Pharos injects structured knowledge as text context or KV cache state at inference time. No training. No modification. The same model reasons differently depending on which pack is injected. Switch from Aristotelian to Kantian reasoning by swapping a pack — in the same inference call.

## Repo

`Liberation-Labs-THCoalition/Project-Pharos` (private)

## Docs in Repo

- `README.md` — architecture overview
- `FINDINGS.md` — all validated results with numbers
- `METHODS.md` — experiment spec with replication details
- `SITE_REFERENCE.md` — this document

---

*Named for the Lighthouse of Alexandria. The model already knows the concepts. Pharos teaches it the structure.*
