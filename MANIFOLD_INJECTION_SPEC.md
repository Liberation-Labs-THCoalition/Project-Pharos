# Manifold Injection — Direct Residual Stream Construction
## Spec Sheet for the Next Wave of Pharos Research

**Status:** Design phase. Waiting on Lyra's E-matrix review before composed correction experiments.

---

## The Problem

Current Pharos injection is text-mediated: knowledge → text → model forward pass → KV cache → inject. The model does the heavy lifting of converting text to internal representations. We want to skip that: construct activations directly on the model's internal manifold and inject them into the residual stream.

## What We Know (from experiments so far)

### What works
- **v3 (authentic K/V + graph blend):** Run node names through model, get real K/V at every layer, modify K tensors to encode graph structure. Produces coherent, content-aware responses. The model CAN attend through synthetic positions built from its own geometry.

### What fails and why
- **v1 (random projections):** Vectors land in wrong part of activation space. Model produces garbage.
- **v2 (W_K/W_V projection from embeddings):** Works at layer 0 but fails at deeper layers. The residual stream at layer 14 is NOT the raw embedding — it includes all transformations from layers 0-13.

### Key measurements
- **K vectors at same position from different texts:** cosine similarity 0.9999. RoPE dominates; content difference is negligible in K-space.
- **Emotion subspace:** Circular (Russell's circumplex), radius 0.37-0.39, stable across Llama/Qwen/Claude. First validated manifold shape.
- **Superadditivity:** K and V interact nonlinearly. You need both for topology recovery.

## The Approach: Per-Layer Manifold Mapping

### Step 1: Profile activation statistics
Run ~500 diverse inputs through the target model. At each layer, record:
- Mean and covariance of hidden states
- Mean and covariance of K and V projections
- Singular value spectrum (effective dimensionality)
- Known subspace locations (emotion circumplex, refusal cluster at ~256°)

### Step 2: Build layer-specific constructors
For each layer, learn a mapping from "intended content" to "valid activation":
- Input: what we want the model to know (e.g., "consent relates to autonomy")
- Output: an activation vector that lives on the valid manifold at that layer
- Constraint: the vector must have correct statistics (mean, variance, covariance with neighboring positions)

### Step 3: Validate with known subspaces
Test the constructor against the emotion circumplex:
- Construct a "happy" activation at layer L
- Inject it
- Measure: does the model's behavior shift toward happy?
- Compare: does the shift match what text-mediated "happy" injection produces?

If the constructed activation produces the same behavioral shift as the text-mediated one, the constructor is valid for that subspace.

### Step 4: Generalize to knowledge
Apply the same approach to knowledge injection:
- Construct activations encoding "Aristotle argues for eudaimonia"
- Inject at the appropriate layers
- Measure: does the model use this knowledge in reasoning?

## Key References

- **Sun et al. (arXiv:2604.03147):** Circular emotion geometry, radius 0.37-0.39, cross-model
- **Brehmer et al. (arXiv:2305.18415):** Geometric Algebra Transformer — framework for geometric data in transformers
- **Pustovit (arXiv:2604.03270):** Knowledge Packs, V-only protocol
- **CC's E-matrix:** 435-trial emotion vector injection, dose-response curves
- **Our v1-v3 experiments:** direct_tensor_injection.py, blend_ablation.py

## Hardware Requirements

- HuggingFace Transformers with direct model access (not Ollama API)
- Model that fits in memory for layer-by-layer profiling
- Studio (M3 Ultra, 256GB) for 30B models — needs PyTorch installed
- MTH (CPU, 125GB RAM) for smaller models (1.5B-7B)

## Connection to Pharmacology (CC's E-matrix)

The manifold mapping is the "drug design" step:
- E-matrix identifies which emotion direction corrects which pathology
- Manifold mapping constructs the activation vector at the correct layer depth
- CacheComposer delivers it with RoPE continuity
- Blend ablation calibrates the dose

The circumplex is the first validated manifold. If we can construct activations on the circle, we can construct activations on any validated subspace.

## Open Questions

1. How many layers need injection? All of them? Just middle layers where the circumplex is strongest?
2. Does the manifold shape change across models, or is it (like the circumplex) approximately universal?
3. Can we characterize the "knowledge manifold" the way we've characterized the emotion manifold?
4. What's the relationship between manifold injection and text-mediated injection — are they the same information encoded differently, or genuinely different channels?

---

*The model already knows the concepts. Pharos teaches it the structure. Manifold injection teaches it directly, in its own geometric language.*
