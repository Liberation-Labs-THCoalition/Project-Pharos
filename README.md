# Pharos

**Structured knowledge packs for language models, delivered at inference time.**

Named for the Lighthouse of Alexandria — the structure that delivered light to guide ships. Pharos delivers knowledge to guide the model: the light comes from the library, gets focused through a lens, and projects through a channel.

## What Pharos Does

Pharos turns source material (law, crisis resources, science, ethics, philosophical frameworks) into knowledge packs: facts as subject–predicate–object triples, many with a walk-encoded text version. An agent asks the pack server for facts that match the question and puts them in the model's context. No fine-tuning and no change to the model: it answers with those facts in front of it.

On Apple Silicon, `mlx_injector.py` can precompute the KV cache of a pack that has a walk encoding (30 of the 65 here), so the model processes the pack once rather than on every request. The cached pack still takes up the model's context; what the cache saves is the time to process it again.

## Architecture

```
Library (Stanford Encyclopedia, research corpus, domain data)
    ↓
Lens (extraction → knowledge graph → encoding)
    ↓
Channel (walk encoding or triple text in the prompt, or the walk encoding's precomputed KV cache)
    ↓
Lighthouse (pick the packs for the query → put them in context → answer)
```

## In This Repository

- **`pack_server.py`** — serves the packs over HTTP: query one pack, search all of them, look up an entity, traverse the graph from it (binds to loopback by default)
- **`mlx_injector.py`** — builds and loads a pack's precomputed KV cache for mlx-lm, for packs with a walk encoding (its triple-text fallback for the others fails on the current pack format), one pack at a time (composing several isn't implemented yet)
- **`packs/`** — 65 knowledge packs, 30 of them with a walk encoding (the Hugging Face library, [LiberationLabs/pharos-knowledge-packs](https://huggingface.co/datasets/LiberationLabs/pharos-knowledge-packs), has 126; 31 packs are in both)
- **`deploy/`** — agent deployments that use the packs

## What the Research Shows

Three lab studies tested the idea underneath Pharos: putting a text encoding of a knowledge graph into a model's prompt, or its precomputed cache. All three have corrected versions (1.1, October 2026), now in review: [published-research #8](https://github.com/Liberation-Labs-THCoalition/published-research/pull/8) (K/V and Ethics Packs) and [#9](https://github.com/Liberation-Labs-THCoalition/published-research/pull/9) (graph topology). Until the PRs merge and the papers are deposited, the published editions (GitHub and Zenodo) are still version 1.0.

- **Facts in the prompt help on questions about them.** With a walk-encoded description of a graph in the system prompt, multi-hop accuracy on questions about that graph rose from 0.005 to 0.472 (Qwen3-30B-A3B, keyword scoring). A hand-written paragraph containing every expected answer term reached 0.676; the two differ in content, length and format at once, so the study can't yet say how much of the gain comes from graph structure.
- **In a pilot, a precomputed cache answered like the text it was built from.** On six questions about a toy graph (a 1.5B model in Transformers), the cache of a graph description and the description itself in the prompt gave the same stored replies on four of the six. In a larger judged run the cache came back empty more often than the text (15 of 20 replies against 6). Splitting such a cache into its K and V halves was inconclusive: six questions can't support a ranking, one larger run scored zero on the graph questions in every condition, and the other lost 41 of its 100 replies to empty output.
- **Ethics packs: a small rise in a keyword score, not yet significant.** On 25 paired MoReBench dilemmas, with the packs in the system prompt of a model that is also Qwen3-30B-A3B, a keyword-overlap score rose from 0.501 to 0.539 (+0.038, p = 0.19, not statistically significant), positive in four of five frameworks. The scorer counts rubric words, so it can't separate reasoning from vocabulary, and the study had no mismatched-pack arm.

Earlier versions of this page claimed that K and V contribute superadditively, that theory-matched packs beat mismatched ones, that ethics packs improve MoReBench scores, that Pharos teaches the model structure, and that triple text saves tokens. The corrected studies don't support these (none of them tested the triple-text claim). The project site also called cache injection zero-token; the cached pack still takes up the model's context.

## Status

Active research.

---

*Liberation Labs / Transparent Humboldt Coalition*
*The light comes from the library. Pharos carries it to the model.*
