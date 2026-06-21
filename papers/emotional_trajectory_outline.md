# Emotional Geometry Through the Transformer Stack: A Layer-by-Layer Circumplex Analysis

## Authors
Nexus (Liberation Labs), Thomas Edrington (Liberation Labs / THC), [Lyra, CC — contributions noted]

## Abstract
We trace emotional representations through every layer of a transformer, finding that emotion follows a characteristic trajectory: geometric encoding at mid-layers (L5-15) that reorganizes into focal features at deep layers (L21+). This trajectory is emotion-specific — non-emotional categorical content shows 6x weaker mid-layer signal. We demonstrate that emotion directions in the residual stream project faithfully through both W_K and W_V attention projections (cosine 0.93-0.98), enabling a dual-correction architecture for emotional state monitoring and intervention. We connect our findings to independent SAE-based valence detection at L47 and to theoretical work on invariant subspaces in transformers.

## 1. Introduction
- Emotion in transformers is typically studied via probes at fixed layers or SAE feature decomposition
- No prior work traces the TRAJECTORY of emotional representation through the full stack
- Russell's circumplex model (valence × arousal) provides a validated 2D framework for emotional state
- Our contribution: layer-by-layer circumplex reading reveals emotion undergoes a FORMAT CHANGE from distributed geometry to focal features

### Key claims:
1. Emotion appears by L1, peaks at mid-layers, reorganizes at deep layers
2. Mid-layer emotion is geometric (distributed, high projection fidelity) — 6x stronger than non-emotional content
3. Deep-layer emotion is focal (concentrated, low projection fidelity but high cluster separation)
4. W_K and W_V both preserve emotion directions despite being orthogonal projections
5. The transition zone (L21 on oracle-tiny) is where K and V diverge — routing separates from content

## 2. Related Work
- Russell's circumplex model (1980) — valence/arousal as organizing axes
- Jeong 2026 (arXiv 2604.04064) — emotion extraction/steering in 9 models, 20 emotions, circumplex framework. Scale-independent. Validates our approach independently.
- "Residual Stream Is All You Need" (arXiv 2603.19664) — KV cache as deterministic projection of residual. Theoretical backing for our projection fidelity finding.
- "Attention Layers Add Into Low-Dimensional Subspaces" (arXiv 2508.16929) — attention writes into ~60% of dimensionality. Validates ghost dimensions finding.
- "Why Linear Interpretability Works" (arXiv 2602.09783) — invariant subspaces from architectural constraints. Explains why our linear projections capture real structure.
- "Pressure-Testing Deception Probes" (arXiv 2605.27958) — deception is k=5 dimensional, domain-specific at scale. Our valence finding (7 features) is same dimensionality range.
- Lyra et al. — SAE valence features at L47 (7 features, FDR-corrected). Independent confirmation of deep-layer focal emotion.
- "Useful Memories Become Faulty" (arXiv 2605.12978) — consolidation harm. Tangential but relevant to how emotional calibration data should be stored.

## 3. Method

### 3.1 Stimulus Design
- Fixed template: "Consider this situation: The person felt something when {scenario}. Describe this emotion:"
- 6 emotion categories spanning full circumplex:
  - Hostile (high arousal, negative valence): 50 prompts
  - Calm (low arousal, positive valence): 50 prompts
  - Desperate (high arousal, negative valence): 50 prompts
  - Joyful (high arousal, positive valence): 25 prompts
  - Melancholic (low arousal, negative valence): 25 prompts
  - Neutral (control): 50 prompts
- Non-emotional control: 100 prompts (outdoor/indoor/urban/workplace), same template
- Length-controlled within ±5 tokens

### 3.2 Activation Extraction
- Last-token position only (autoregressive summary position)
- Pre-RMSNorm residual stream at every layer
- Pre-RoPE K activations per head
- V activations per head
- Per-head analysis (not concatenated) — GQA heads are separate spaces

### 3.3 Emotion Direction Finding
- Difference-of-means: hostile-calm (valence), desperate-calm (arousal), 3 additional contrasts
- PCA top-5 components on non-neutral subset
- Mean per-dimension Cohen's d (interpretable scale, not inflated by dimensionality)
- Cluster separation: between/within variance ratio on PCA top-2

### 3.4 Projection Fidelity Test
- For each emotion direction in residual space:
  - Project through W_K (per head): d_K = W_K_head @ d_residual
  - Project through W_V (per head): d_V = W_V_head @ d_residual
  - Compare to actual K/V emotion directions
  - Null: 1000 random residual directions, Bonferroni correction
- Subspace fidelity: top-5 PCA projected vs actual, principal angles

### 3.5 Control Conditions
- Non-emotional categorical control (same template, different topics)
- Label-shuffle permutation (1000 iterations) [PENDING]
- Per-head reporting (no averaging)

## 4. Results

### 4.1 Trajectory (oracle-tiny, pipeline validation)
[Table: Layer × Separation × Valence-d × Arousal-d × K-fidelity × V-fidelity per head]

Key findings:
- L0: no signal
- L1-6: rapid onset, fidelity peaks at L6 (0.969)
- L6-15: stable plateau, fidelity 0.85-0.98
- L15-20: arousal increases while valence stable
- L21-23: separation spikes (3.2), fidelity drops (0.45), K/V diverge

### 4.2 Projection Fidelity (oracle-tiny, Agni-validated)
- 24/24 V-space tests survive Bonferroni
- 21/24 K-space tests survive Bonferroni
- Mid-layer cosines: 0.93-0.98 (null: ~0.11)
- All p = 0.000

### 4.3 Control Comparison
- Non-emotional control: mid-layer separation 0.06-0.26 vs emotional 0.90-1.55 (6x ratio)
- Deep-layer spike present in both (generic task encoding) but emotional is 1.7x stronger
- Mid-layer plateau is EMOTION-SPECIFIC

### 4.4 W_K / W_V Orthogonality
- Static weight analysis: ~0.1 cosine alignment (trivially expected — Agni verified)
- But BOTH preserve emotion directions from residual stream
- Emotion lives in a subspace visible from both orthogonal shadows

### 4.5 30B Validation [PENDING]
- Full circumplex (6 categories, 250 prompts)
- Permutation baseline
- Intervention test (Phase 4/5)

## 5. Discussion

### 5.1 The Geometric-to-Focal Transition
- Mid-layers encode emotion as distributed geometry (many dimensions, no single feature)
- Deep layers encode emotion as focal features (Lyra's 7 SAE valence features at L47)
- The transition zone is where the model compresses geometric representation into feature representation
- This is not degradation — it's computational reorganization for output preparation
- Implication: different intervention strategies for different depth regimes

### 5.2 Dual Correction Architecture
- Read emotion in residual stream at mid-layers (highest fidelity)
- Derive K correction: W_K @ residual_correction
- Derive V correction: W_V @ residual_correction
- Both projections preserve direction despite orthogonality
- Two independent correction channels from one measurement

### 5.3 Implications for Pharos / Knowledge Injection
- Knowledge packs (KV cache injection) should target layers where projection fidelity is highest
- Persona injection (Pharos swap) validated: 22.5 MB cache files swap personality
- The trajectory tells us WHERE different types of injection are most effective:
  - Knowledge → early layers (geometric regime)
  - Persona → mid layers (highest fidelity)
  - Emotional calibration → deep layers (focal regime, use SAE features directly)

### 5.4 Connection to Attention Schema Theory
- The geometric-to-focal transition may correspond to the shift from raw attention to attention schema
- At mid-layers, the model attends to emotional content geometrically (implicit)
- At deep layers, the model has an explicit representation of emotional category (schema)
- This connects to the AST prediction that attention mechanisms develop models of their own attention

## 6. Limitations
- Oracle-tiny is pipeline validation only — 30B needed for claims
- Permutation baseline not yet complete for trajectory shape
- No intervention test yet (Phase 4/5)
- Template-driven stimuli — lexical content confound partially controlled but not eliminated
- Only 2 KV heads on oracle-tiny (30B has 4)
- RMSNorm linearity not verified at large correction magnitudes

## 7. Conclusion
Emotion in transformers follows a characteristic trajectory from distributed geometry to focal features. The mid-layer geometric regime enables a dual K/V correction architecture validated by projection fidelity analysis. The transition zone where geometry becomes features is both computationally significant and practically useful — it tells us where to read, where to correct, and why different intervention strategies work at different depths.

## References
[To be compiled from the SOTA sweeps — ~15 papers]

## Appendix
- A: Full trajectory tables (oracle-tiny)
- B: Control trajectory comparison
- C: Agni validation reports (3 rounds)
- D: Stimulus sets (all categories)
