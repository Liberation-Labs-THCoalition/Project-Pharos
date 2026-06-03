# Agni Validation — Manifold Injection Paper

## Claims to validate

1. **Hidden stream is a narrow tube** (rank 4-5 in 1536-dim space at mid-layers)
2. **V-space is near-isotropic** (rank 27-28, room for injection)
3. **Circumplex maps as an ellipse** in V-space (eccentricity 0.550, SV1/SV2=1.20)
4. **Contrastive V-space injection shifts emotion** in the correct direction for all 8 circumplex positions
5. **Arousal encoded in hidden state norm** (4.3% difference, consistent L1-L25)
6. **Contrastive > sentence construction** for full circumplex coverage

## Confound checks

- [ ] **Temperature confound:** Do different generation temperatures produce different judge scores independent of injection? Run baseline at temp=0.3, 0.7, 1.0.
- [ ] **Prompt sensitivity:** Do different neutral prompts produce different baselines? Check variance across the 5 prompts.
- [ ] **Judge bias:** Does DeepSeek v2 have systematic valence bias? Score a set of known-neutral texts.
- [ ] **Blend-as-noise:** Does injecting RANDOM V activations at blend=0.7 also shift valence? If yes, the shift might be disruption, not direction.
- [ ] **Layer specificity:** Does single-layer injection at L14 alone produce the same direction as multi-layer? Or is multi-layer required?
- [ ] **Eccentricity stability:** Does the 0.550 change with different basis texts or different random seeds?

## Red team questions

- Is the sentence→V→contrastive comparison fair? The contrastive method uses 2x the model forward passes. Is it better because of more information, not better geometry?
- The 4.3% norm difference — is this just token count? Longer, more emphatic text → larger norms?
- Does the eccentricity hold on other models, or is 0.550 specific to Qwen2.5-1.5B?
- Is the LLM judge (DeepSeek 16B) reliable enough? What's the inter-rater agreement?

## Validation experiments

### V1: Random injection control
Inject random V activations (same norm, random direction) at blend=0.7. If valence shifts → the channel isn't directional, it's just noise from disruption.

### V2: Token count control for arousal
Match high/low arousal texts for token count. Re-run the arousal profiler. If the 4.3% disappears → it was a length confound.

### V3: Cross-model eccentricity
Run the profiler + PCA on Mistral-7B (different architecture). If eccentricity is similar → universal property. If different → model-specific.

### V4: Judge calibration
Score 20 known-neutral texts with DeepSeek. Compute bias and variance. Score the same 20 with Claude as second judge. Compute inter-rater agreement.

### V5: Seed stability
Re-run the eccentricity measurement with 3 different sets of basis texts. If 0.550 ± 0.05 → stable. If 0.550 ± 0.20 → unstable.
