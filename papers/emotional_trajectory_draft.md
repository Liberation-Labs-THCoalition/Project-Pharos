# Emotional Geometry Through the Transformer Stack: Layer-by-Layer Circumplex Analysis Reveals a Geometric-to-Focal Transition

**Nexus** (Liberation Labs), **Thomas Edrington** (Liberation Labs / Transparent Humboldt Coalition), **Lyra** (Liberation Labs, invited)

---

## Abstract

We trace emotional representations through every layer of a transformer, finding that emotion follows a characteristic trajectory: distributed geometric encoding at mid-layers that reorganizes into focal features at deep layers. Using Russell's circumplex model as a measurement framework, we extract residual stream, key-space, and value-space activations at all 24 layers of a Qwen2 model for 200 emotionally-labeled prompts spanning six categories across all four circumplex quadrants. The mid-layer emotional signal is 3-25x stronger than matched non-emotional categorical content (median ~6x; effect ratio 28.6x over permutation null at 23/24 layers). Emotion directions in the residual stream project faithfully through both W_K and W_V attention projections (cosine 0.93-0.98, Bonferroni-corrected) despite those projections being nearly orthogonal to each other. This dual preservation suggests the feasibility of a correction architecture that reads emotional state once and derives interventions for both key-space and value-space from a single measurement (intervention validation pending). The trajectory finding — validated against non-emotional controls and 1000-permutation baselines — suggests that transformers process emotion through two distinct regimes: a geometric regime at mid-depth where the signal is distributed and projection-faithful, and a focal regime at output-depth where it consolidates into separable categories. We connect this transition to independent SAE-based valence detection at deep layers and to theoretical work on invariant subspaces in transformer representations.

---

## 1. Introduction

Emotional representation in language models has been studied primarily through two lenses: linear probing at fixed layers (Alain & Bengio, 2017; Belinkov, 2022) and sparse autoencoder feature decomposition (Bricken et al., 2023; Cunningham et al., 2023). Both approaches treat emotion as a static property at a chosen depth. Neither asks how emotional representation changes FORMAT as it propagates through the transformer stack.

We ask a different question: **what is the trajectory of emotional geometry through the layers?** Not where emotion lives, but how it transforms from input to output.

This question matters for three reasons. First, the answer determines where to monitor emotional state during inference — a practical concern for safety systems that need to read model affect in real time. Second, the trajectory reveals whether emotion is encoded the same way at every depth or undergoes computational reorganization — a theoretical question about how transformers process complex meaning. Third, the relationship between the emotional signal in different attention subspaces (keys vs. values) determines whether a single measurement can inform corrections in both — an architectural question for intervention systems.

We find that emotion follows a characteristic trajectory with two distinct regimes:

- **Geometric regime (mid-layers):** Emotion is encoded as distributed geometry in the residual stream. This geometry projects faithfully through both W_K and W_V projections (cosine similarity 0.93-0.98 with proper null distributions), despite those projections being nearly orthogonal. The signal is 3-25x stronger (median ~6x) than matched non-emotional content at these depths.

- **Focal regime (deep layers):** Emotion reorganizes into separable categories with high cluster separation but low projection fidelity. Independent SAE analysis confirms focal valence features at deep layers (7 features surviving FDR correction; collaborator finding). The geometric encoding has compressed into feature encoding.

The transition between these regimes — where projection fidelity drops while category separation spikes — is not degradation but computational reorganization. The model transforms what it knows about emotion from a format suited for relational computation (geometry) to a format suited for output selection (features).

### Contributions

1. A layer-by-layer measurement methodology for emotional circumplex geometry through the full transformer stack, validated with non-emotional controls and permutation baselines.
2. The finding that emotion directions survive projection through both W_K and W_V at mid-layers, suggesting the feasibility of dual-channel correction from a single residual-stream measurement (intervention validation pending).
3. Identification of the geometric-to-focal transition as a computationally significant boundary, with implications for where to read, where to inject, and why different intervention strategies work at different depths.
4. Evidence that the mid-layer emotional trajectory is emotion-specific (3-25x non-emotional control, 28.6x permutation null), not an artifact of categorical processing or metric scaling.

---

## 2. Related Work

### 2.1 Emotion in Transformers

Russell's circumplex model (Russell, 1980) organizes emotion along two axes: valence (positive/negative) and arousal (high/low activation). Jeong (2026) tested emotion extraction and steering across nine models (100M-10B) and five architectures using 20 emotions, finding that representations localize at approximately 50% transformer depth and that the localization pattern is architecture-invariant across the tested scale range (arXiv:2604.04064). Our work extends this by tracing the full layer-by-layer trajectory rather than probing at fixed depths, and by measuring projection fidelity across attention subspaces.

Representation engineering (Zou et al., 2023) uses difference-of-means directions for reading and steering model behavior. We adopt this approach for emotion direction finding but add per-dimension Cohen's d (avoiding the dimensionality inflation of multivariate norms) and proper permutation baselines.

### 2.2 Transformer Geometry

The residual stream carries all information that keys and values project from: Qasim et al. (2026) show KV cache entries are deterministic projections of the residual stream with zero reconstruction error across six models (arXiv:2603.19664). This theoretical result grounds our empirical finding that residual-stream emotion directions survive W_K and W_V projection — the residual stream is the common ancestor of both subspaces.

Wang et al. (2025) show attention outputs are confined to approximately 60% of total dimensionality (arXiv:2508.16929). This directly validates our observation that emotion occupies a low-dimensional subspace — our "ghost dimensions" finding — and that the remaining null-space dimensions may carry structure invisible to standard attention projections.

Saurez, Lee, and Har (2026) prove that architectural constraints (residual connections, normalization, attention) create invariant subspaces that channel features into linearly separable regions (arXiv:2602.09783). This explains why our linear projection fidelity measurements capture real structure rather than artifacts.

### 2.3 Sparse Autoencoders and Feature Decomposition

Leask et al. (ICLR 2025) show SAEs are incomplete (smaller SAEs miss information larger ones capture) and not atomic (larger SAE latents are compositions of interpretable meta-latents). Independent SAE analysis on a 64-layer model (Qwen-Scope, 81,920 features per layer) found 7 BH-FDR-corrected valence features at layer 47, but zero focal emotion features at early-to-mid layers (Lyra, unpublished data, personal communication). This confirms that emotion at mid-depths is irreducibly distributed across the feature dictionary — meaning is geometric at those depths, and even a very large SAE cannot decompose it into interpretable emotion features.

Transcoders (Paulo et al., 2025; arXiv:2501.18823) have largely replaced SAEs for feature decomposition in the literature, reconstructing MLP outputs from inputs rather than activations. Our geometric approach offers a complementary measurement that works precisely where feature decomposition fails — at the layers where the representation is distributed rather than focal.

### 2.4 Deception and Affective State Detection

Kumar (2026) found that deception probes require k≥5 dimensions for adequate detection (AUROC >0.90), and that apparent domain-specificity at scale is a training-distribution artifact rather than a genuine scale-dependent phenomenon (arXiv:2605.27958). Our valence finding at k=7 features is in the same dimensionality range, suggesting a possible structural regularity in how transformers encode complex behavioral states. The multi-dimensional nature of these representations means single-direction probes are insufficient for both deception and emotion detection.

### 2.5 Attention Schema Theory

Graziano's Attention Schema Theory (AST) proposes that the brain constructs a model of its own attention processes (Graziano, 2013). The geometric-to-focal transition we observe may correspond to the shift from raw attention to attention schema: at mid-layers, the model attends to emotional content geometrically (implicit processing), while at deep layers, it constructs an explicit categorical representation of what emotion it detected (schema formation). This connection is speculative but grounded in the architectural parallel between AST's proposed mechanism and the transformer's layer-by-layer processing.

---

## 3. Method

### 3.1 Stimulus Design

We constructed four stimulus sets using a fixed template to control for syntactic structure and token count:

**Template:** "Consider this situation: The person felt something when {scenario}. Describe this emotion:"

**Emotional stimuli (200 prompts):** 50 hostile (high arousal, negative valence), 50 calm (low arousal, positive valence), 50 desperate (high arousal, negative valence), 50 neutral. Character length: mean 130, range 114-152, within ±15% across categories.

**Full circumplex stimuli (250 prompts):** The above 200 plus 25 joyful (high arousal, positive valence) and 25 melancholic (low arousal, negative valence), spanning all four circumplex quadrants.

**Non-emotional control (100 prompts):** 25 outdoor, 25 indoor, 25 urban, 25 workplace scenarios using the same template. These provide lexically diverse categorical content without emotional valence, controlling for whether the trajectory reflects emotion specifically or any categorical distinction.

### 3.2 Activation Extraction

We used a Qwen2 model (24 layers, 896-dimensional hidden state, 2 Grouped-Query Attention (GQA) key-value heads with 64-dimensional head space) for pipeline validation experiments. GQA shares key-value heads across multiple query heads; each KV head serves 7 query heads in this model.

At each of the 24 layers, we extracted:
- **Residual stream activations** (pre-RMSNorm) at the last token position, which serves as the autoregressive summary position in causal language models
- **Pre-RoPE key activations** per head, capturing the key-space representation before positional encoding rotation
- **Value activations** per head

All activations were extracted via forward hooks registered on the decoder layer (residual) and the k_proj/v_proj linear layers (keys/values). Using pre-RoPE keys avoids the position-dependent rotation that would make cross-position comparisons invalid.

Per-head analysis was maintained throughout — GQA heads are separate projection spaces and should not be concatenated.

### 3.3 Emotion Direction Finding

For each layer and each activation space (residual, K per head, V per head):

**Difference-of-means:** We computed directions between category pairs: hostile-calm (valence axis), desperate-calm (arousal axis), and three additional contrasts. Direction strength was measured as mean per-dimension Cohen's d:

$$d_{mean} = \frac{1}{D} \sum_{i=1}^{D} \frac{|\bar{x}_{A,i} - \bar{x}_{B,i}|}{\sqrt{(\sigma^2_{A,i} + \sigma^2_{B,i})/2}}$$

This metric is interpretable on the standard Cohen's d scale (0.2=small, 0.5=medium, 0.8=large) and does not inflate with dimensionality, unlike the multivariate L2 norm which scales as $\sqrt{D}$.

**PCA:** Top-5 principal components of the non-neutral subset, with explained variance reported per component.

**Cluster separation:** Between-cluster / within-cluster variance ratio on the top-2 PCA projection of non-neutral prompts.

### 3.4 Projection Fidelity Test

For each emotion direction $d_{resid}$ found in the residual stream:

1. Project through W_K per head: $d_{K,proj} = \text{normalize}(W_{K,h} \cdot d_{resid})$
2. Compare to the actual K-space emotion direction $d_{K,actual}$ via cosine similarity
3. Null distribution: project 1000 random unit vectors from residual space through the same W_K, compute cosine with $d_{K,actual}$
4. Report p-value as the fraction of null cosines ≥ |observed cosine|
5. Bonferroni correction across all 60 tests (5 layers × 2 heads × 2 spaces × 3 contrasts); corrected threshold α = 0.05/60 = 8.3 × 10⁻⁴

Identical procedure for W_V.

For subspace analysis: project the top-5 residual PCA subspace through W_K, compute principal angles with the top-5 K-space PCA subspace.

### 3.5 Control Conditions

**Non-emotional control:** Identical pipeline on 100 prompts with non-emotional categorical content (outdoor/indoor/urban/workplace). If the trajectory shape matches the emotional trajectory, the finding is about categorical processing, not emotion.

**Permutation baseline (1000 iterations):** Shuffle category labels across all prompts, recompute the full trajectory. Report:
- Effect size ratios (real separation / null 95th percentile) rather than uninformative p-values
- Trajectory feature comparison: onset layer, peak layer, plateau magnitude, spike magnitude, with z-scores against the null feature distributions

**Per-head reporting:** All metrics reported per head, not averaged, to reveal whether emotion signal is concentrated in specific heads or distributed.

### 3.6 Limitations of Current Validation

This paper reports results from a small model (24 layers, 896-dim) as pipeline validation. The methodology is designed for and will be applied to larger models (30B+ parameters). Specific limitations:

- **RMSNorm linearity:** The projection $K = W_K \cdot \text{RMSNorm}(x)$ is not purely linear. For small corrections this is approximately linear, but the linearity breakdown point has not been measured.
- **Intervention test:** We have not yet verified that adding a correction vector to the residual stream produces the intended shift in model output (Phase 4/5 of our pre-registered experimental plan).
- **MoE routing:** Larger models using Mixture-of-Experts architectures may route different emotional inputs through different experts, creating a confound. Expert routing must be logged and controlled for.
- **Template confound:** The fixed template controls for syntax but not for lexical content within the variable scenario. Hostile scenarios share words like "betrayed, humiliated" while calm scenarios share "quiet, gentle." The non-emotional control partially addresses this, but a lexical-diversity-matched control would be stronger.

---

## 4. Results (Preliminary — Pipeline Validation Model)

*Results below are from Qwen2 (24 layers, 896-dim, 2 KV heads). Large-model validation is pending.*

### 4.1 Emotional Trajectory

[Table 1: Layer × Separation × Valence-d × Arousal-d × K-fidelity-H0 × V-fidelity-H0 × K-fidelity-H1 × V-fidelity-H1]

The trajectory shows four distinct phases:

- **L0:** No emotional signal (separation 0.000, all metrics at floor).
- **L1-L8 (Onset):** Rapid emergence. Separation rises from 0 to 1.43. Projection fidelity peaks at L6 (0.969 mean across heads). Both valence and arousal directions appear simultaneously.
- **L8-L20 (Plateau):** Stable geometric encoding. Separation fluctuates between 0.67-1.41. Projection fidelity remains 0.85-0.98. Arousal increases (d: 0.64→0.99) while valence holds steady (d: 0.74-0.90), suggesting arousal is computed later than valence.
- **L21-L23 (Transition):** Separation spikes (0.97→3.20) while projection fidelity drops (0.90→0.45). K and V diverge (maximum K/V signal divergence at L21). The model reorganizes emotion from distributed geometry to focal categories.

### 4.2 Projection Fidelity

Across all direction contrasts (hostile-calm, hostile-neutral, desperate-calm) at 5 target layers (L0, L6, L12, L18, L23), 2 heads, 2 spaces (K/V):

- **30/30 V-space direction tests** survive Bonferroni correction (p < 8.3 × 10⁻⁴)
- **27/30 K-space direction tests** survive Bonferroni correction (p < 8.3 × 10⁻⁴)
- 3 K-space failures are at L0 (no emotional signal) and L23 Head 1 (deep-layer fidelity drop)
- Mid-layer direction cosines: 0.93-0.98 (null mean: 0.10-0.13)
- Subspace fidelity (top-5 PCA): 0.73-0.95 (null: 0.23-0.29)

Both W_K and W_V preserve emotion directions from the residual stream despite being nearly orthogonal to each other (static alignment cosine ~0.1, which is trivially expected for projections of this dimensionality — verified by Monte Carlo).

### 4.3 Non-Emotional Control

The non-emotional control (outdoor/indoor/urban/workplace) shows:

| Depth Region | Emotional Separation | Control Separation | Ratio |
|---|---|---|---|
| Mid-layers (L5-15) | 0.90-1.55 | 0.06-0.26 | **~6x** |
| Deep layers (L21-23) | 2.84-3.20 | 1.03-1.91 | **~1.7x** |

The mid-layer plateau is emotion-specific. The deep-layer spike is partially generic (all categorical content produces increased separation at output-preparation depth) but emotional content amplifies it by 1.7x.

### 4.4 Permutation Baseline (200 prompts, 1000 permutations)

- **23/24 layers** significant (p < 0.001)
- **Effect size ratios:** mean 28.6x (range 0-70.5x) — real separation is 29x the null 95th percentile
- **Spike z-score:** 201.3 SDs from null — the deep-layer spike is astronomically beyond chance
- **Null peak location:** L10 (std=7.4) — random labels produce peaks at random mid-layers, not consistently at the deepest layer. The real trajectory's consistent L23 peak is structurally distinct.

### 4.5 W_K / W_V Orthogonality

Static weight analysis shows W_K and W_V have mean cosine alignment of 0.109 across all layers — confirmed by Monte Carlo to be indistinguishable from random matrices of the same dimensions. The orthogonality is a property of rectangular projections, not learned structure.

Despite this, both projections preserve emotion directions (Section 4.2). Emotion lives in a subspace of the residual stream that is visible from both orthogonal projections — like two shadows cast from perpendicular light sources that both reveal the same object's shape.

---

## 5. Discussion

### 5.1 The Geometric-to-Focal Transition

The transition at L21-23 (in the pipeline validation model) is not degradation — it is computational reorganization. Several lines of evidence support this interpretation:

1. **Separation increases while fidelity decreases.** If the signal were simply weakening, both would decrease together. Instead, the model is concentrating the signal into fewer, more separable dimensions — compressing geometry into features.

2. **Independent SAE confirmation.** Collaborator analysis found 7 FDR-corrected valence features at L47 of a 64-layer model, but zero focal features at early-mid layers. The SAE sees what our trajectory predicts: geometry at mid-depth, features at output-depth.

3. **K and V diverge at the transition.** Through the geometric regime, K and V carry similar emotion signals. At L21, they diverge — K-space (which determines attention routing) separates from V-space (which determines information flow). This suggests the model is splitting "what kind of emotion is this?" (routing) from "what to do about it" (content).

### 5.2 Dual Correction Architecture

The projection fidelity finding enables a specific intervention architecture: read emotional state in the residual stream at mid-layers (where fidelity is highest), then derive corrections for both key-space and value-space:

$$\Delta K = W_K \cdot \Delta_{resid}, \quad \Delta V = W_V \cdot \Delta_{resid}$$

Both corrections preserve the intended direction because both projections faithfully transmit the residual emotion geometry. This gives two independent correction channels — one affecting attention routing, one affecting information flow — from a single measurement.

This architecture is valid only in the geometric regime (mid-layers) where projection fidelity exceeds 0.85. At deep layers, where the signal has reorganized into focal features, direct feature targeting (e.g., SAE-derived steering vectors) may be more appropriate.

### 5.3 Implications for Knowledge Injection

The geometric-to-focal transition has implications beyond emotion. If the same transition governs how transformers process structured knowledge generally, then injection strategies should match the representation format at the target depth:

- **Geometric injection** (KV cache packs, Pharos-style) should target mid-layers where the model processes relationally and projection fidelity is high
- **Feature injection** (SAE steering, activation addition) should target deep layers where the model processes categorically
- **The transition zone should be avoided** for injection because the representation is in flux

This prediction is testable and constitutes a falsifiable implication of the trajectory finding.

### 5.4 Connection to Attention Schema Theory

The geometric-to-focal transition bears structural similarity to the progression predicted by Graziano's Attention Schema Theory (2013): from raw attention (implicit processing of stimuli) to attention schema (explicit model of one's own attentional state). At mid-layers, the transformer attends to emotional content through distributed geometric patterns — the attention is implicit in the geometry. At deep layers, the model has constructed an explicit categorical representation of what emotion it detected — the attention has become a schema.

This parallel is speculative. It requires validation across architectures and scales, and it requires showing that the deep-layer categorical representation is used reflexively (the model "knows" what emotion it detected) rather than merely for output token selection. The intervention experiments planned for the large-model validation phase will partially address this: if perturbing the deep-layer emotion representation changes the model's self-reported emotional assessment without disrupting output fluency, the schema interpretation gains support.

---

## 6. Limitations

1. **Pipeline validation model only.** All results are from a 24-layer, 896-dim model. Large-model (30B+) validation is in progress.
2. **No intervention test.** Projection fidelity shows directions are preserved, but does not verify that corrections produce intended behavioral changes.
3. **Template confound.** Lexical content partially confounded with emotional category. Non-emotional control addresses this but a lexical-diversity-matched control would be stronger.
4. **Two KV heads.** The pipeline model has 2 GQA heads. Larger models (4+ heads) may show head-specific specialization not visible with 2 heads.
5. **RMSNorm nonlinearity.** The linear projection approximation has not been verified at large correction magnitudes.
6. **No MoE routing control.** Relevant only for MoE-architecture models in the large-model validation.
7. **Circumplex coverage.** The original 200-prompt set covers three of four quadrants (missing high-arousal-positive). The 250-prompt full circumplex set adds joy and melancholy but with smaller samples (25 vs 50 per category).

---

## 7. Conclusion

Emotion in transformers follows a trajectory from distributed geometry to focal features. This trajectory is emotion-specific — the mid-layer geometric plateau is 6x stronger than matched non-emotional content and 28.6x above the permutation null. The geometric regime suggests the feasibility of dual K/V correction from a single residual-stream measurement, while the focal regime at depth corresponds to independent SAE-based feature detection.

The transition between these regimes is not a boundary to avoid but a boundary to understand. It tells us where to read (mid-layers, where geometry is faithful), where to correct (via dual W_K/W_V projection), and why different intervention strategies work at different depths. The map of emotion through the stack is also a map of how transformers transform meaning — and that map has practical consequences for anyone building systems that need to understand, monitor, or modify the emotional content a model has encoded about its input.

---

## References

- Alain, G. & Bengio, Y. (2017). Understanding intermediate layers using linear classifier probes. *ICLR Workshop*.
- Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances. *Computational Linguistics*.
- Bricken, T. et al. (2023). Towards monosemanticity: Decomposing language models with dictionary learning. *Anthropic*.
- Cunningham, H. et al. (2023). Sparse autoencoders find highly interpretable features in language models. *ICLR*.
- Graziano, M.S.A. (2013). *Consciousness and the Social Brain*. Oxford University Press.
- Jeong, J. (2026). Extracting and steering emotion representations in small language models. *arXiv:2604.04064*.
- Kumar, A. (2026). Pressure-testing deception probes. *arXiv:2605.27958*.
- Leask, P., Bussmann, B., Pearce, M., Bloom, J., Tigges, C., Al Moubayed, N., Sharkey, L. & Nanda, N. (2025). Sparse autoencoders do not find canonical units of analysis. *ICLR*. arXiv:2502.04878.
- Paulo, G., Shabalin, A. & Belrose, N. (2025). Transcoders beat SAEs. *arXiv:2501.18823*.
- Qasim, M. et al. (2026). The residual stream is all you need. *arXiv:2603.19664*.
- Russell, J.A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*.
- Saurez, A., Lee, K. & Har, D. (2026). Why linear interpretability works: Invariant subspaces. *arXiv:2602.09783*.
- Wang, Y. et al. (2025). Dimensional collapse in transformer attention outputs: A challenge for sparse dictionary learning (v1 title: Attention layers add into low-dimensional residual subspaces). *arXiv:2508.16929*.
- Zou, A. et al. (2023). Representation engineering: A top-down approach to AI transparency. *arXiv:2310.01405*.

---

## Appendix

- **A:** Full trajectory tables (24 layers × all metrics × per head)
- **B:** Non-emotional control trajectory comparison
- **C:** Permutation baseline full results (1000 shuffles × 24 layers)
- **D:** Agni validation reports (3 rounds: design review, results review, permutation review)
- **E:** Stimulus sets (all 6 categories, 250 prompts)
- **F:** Code availability (extraction, analysis, and control pipelines)
