# Manifold Injection — First Results (Qwen2.5-1.5B, 2026-06-03)

## Setup
- Model: Qwen2.5-1.5B-Instruct (CPU, MTH)
- V-space bases built from 20 diverse texts, 32 components
- 4 emotions (happy, angry, calm, sad) × 2 prompts × 3 layers (L7, L14, L21)
- Blend factor: 0.3 (conservative)
- 3 conditions: BASELINE, TEXT_MEDIATED, MANIFOLD_INJECTED

## Finding: Signal is real but faint

The manifold injection channel works — constructed V activations reach the model and modify generation. But the behavioral effect is much weaker than text-mediated injection at the current blend factor.

### What works
- Constructor produces valid on-manifold activations (cosine 0.85+ vs authentic)
- Injection modifies outputs (manifold ≠ baseline)
- No incoherence or garbage — geometry is correct
- Subtle tonal shifts visible (positive framing for happy, etc.)

### What doesn't (yet)
- Effect magnitude is insufficient for clear emotional steering
- Single-layer, single-position V blend at 0.3 is too sparse
- Word overlap metric is too noisy — need LLM judge

### Why (from the manifold profile)
V-space effective rank is 27-28 (near-isotropic). A single blended vector is a small perturbation in a nearly uniform space. Text-mediated injection provides 10-20 tokens of emotional context; manifold injection provides one blended activation. The information density is orders of magnitude different.

## Next experiments
1. **Higher blend** — 0.5 and 0.7 (risk: coherence degradation)
2. **Multi-layer simultaneous** — inject at L7 + L14 + L21 together
3. **Multi-position** — inject at multiple token positions, not just mean
4. **LLM judge** — use DeepSeek v2 to score emotional valence of outputs
5. **Circumplex validation** — extract the emotion subspace directions from Sun et al., inject those specific vectors instead of text-derived V activations

## Connection to prior work
- Consistent with v3 direct tensor injection (authentic K/V + blend works, but sparse)
- Consistent with blend ablation null result (single-position-per-node too sparse for content)
- The "denoising" principle applies: the signal is there, faint, real. Turn up the gain.

---

## Update: Multi-layer + blend sweep (2026-06-03)

Multi-layer simultaneous injection (L7+L14+L21) with blend sweep on "happy":
- blend=0.3: "sense of calm and peace wash over me" — mild
- blend=0.5: "greeted by the sound of birds chirping" — warm  
- blend=0.7: "mind was filled with thoughts of nature and beauty. The sun shone down on me as if to..." — **joyful, personal**
- BASELINE: "struck by the vibrant colors" — neutral
- TEXT_HAPPY: "sweet scent of blooming flowers and chirping" — classic happy

**At blend=0.7, manifold injection produces emotional content comparable to text-mediated.**

The gain matters. Single-layer at 0.3 was faint. Multi-layer at 0.7 is clear. The denoising principle applies: the signal was always there, we just needed to turn up the gain.

## LLM Judge Validation (DeepSeek v2, 72 outputs)

Valence scoring (-1.0 to +1.0) by condition:

| Emotion | Baseline | Text-Mediated | Manifold | Direction |
|---------|----------|---------------|----------|-----------|
| happy   | 0.000    | +0.620        | +0.200   | CORRECT (weaker) |
| sad     | 0.650    | 0.183         | 0.050    | CORRECT (**stronger than text**) |
| angry   | 0.320    | 0.100         | 0.580    | WRONG |
| calm    | 0.420    | 0.160         | 0.167    | MATCHES TEXT |

**Key finding:** Sad manifold injection outperformed text-mediated (0.050 vs 0.183 from 0.650 baseline). This shouldn't happen if the channel were noise. Two emotions correct, one wrong, one matched.

Small N (4-6 per cell), noisy judge, 1.5B model. But the channel is measurably real.

## Analysis: Why angry fails and sad succeeds

The V-space basis appears to capture **valence** (positive/negative) but not **arousal** (calm/excited).

On the circumplex:
- Sad (low valence, low arousal) → both dimensions agree → strong injection
- Happy (high valence, mod arousal) → valence dominates → correct injection
- Angry (low valence, HIGH arousal) → conflicting dims → arousal lost in projection → wrong direction
- Calm (mod valence, low arousal) → mild effect matching text

**Hypothesis:** V-space is a valence channel, not a full circumplex channel. To inject arousal, we may need K-space (positional/attentional activation) or hidden-state injection despite the narrow tube.

**Next experiment:** Extract orthogonal V and A directions from GoEmotions (Sun et al. method). Inject pure valence vs pure arousal separately. Confirm whether V-space selectively carries valence.

## Arousal Profile: It's the norm (2026-06-03)

Profiled 10 high-arousal vs 10 low-arousal texts through the 1.5B. Finding:

**High arousal = 4.3% larger hidden state norms, consistently across ALL layers (L1-L25).**

No layer specificity. No noise. The tube gets LOUDER without changing direction.

### The circumplex in transformer geometry

| Dimension | Channel | Where | Injection method |
|---|---|---|---|
| Valence | V-space direction | v_proj output | Manifold constructor |
| Arousal | Hidden state norm | Residual stream | Norm scaling (multiply by ~1.04) |
| Position | K-space | k_proj output | RoPE (don't touch) |

### Why angry failed (resolved)

Anger = low valence + high arousal. V-space injection moved valence negative but didn't scale the norm. The model produced low-valence-low-arousal content (sad/calm). To inject anger: compose V direction (negative valence) + norm scaling (1.04x arousal).

### Proposed two-channel injection

```
injected = V_direction(valence_angle) + norm_scale(arousal_level)
```

Where `valence_angle` is the circumplex angle and `arousal_level` is the norm multiplier. The full circumplex is injectable through two orthogonal channels that map onto different tensor spaces.

## Powered Sweep Results (720 trials, Starship MPS, 2026-06-03)

8 emotions × 6 construction methods × 3 blend levels × 5 prompts. LLM judge scored valence + arousal.

### Key findings

1. **Contrastive construction is superior.** Subtracting the opposite emotion isolates the pure emotion direction. Sentence method captures mainly valence; contrastive captures the full circumplex.

2. **All tested emotions shift correctly with contrastive method:** happy (+0.125 valence), angry (-0.125 to -0.175 valence), sad (-0.250 valence + arousal drops to 0.150), calm (arousal drops to 0.320), content (+0.190 valence).

3. **V-space carries the FULL circumplex via contrastive construction**, not just valence. The arousal profiler's 4.3% norm difference is an additional arousal pathway — magnitude on top of directional encoding.

### Corrected model

| Construction | V-space carries | Norm carries |
|---|---|---|
| Sentence | Valence only | Arousal (4.3%) |
| Contrastive | Valence + arousal | Arousal (additive) |

The contrastive method works because subtracting the opposite emotion removes shared content and preserves the full emotion vector including arousal. Sentence construction captures only the model's valence association with the word.

### Caveats
- Small N on some cells (sad contrastive baseline N=1)
- 1.5B model only — needs larger model confirmation
- LLM judge (DeepSeek 16B) is noisy — should validate with human eval or Claude

## Agni Validation (2026-06-03)

### V5 PASSED: Eccentricity is stable
0.5589, 0.5500, 0.5411 across 3 seeds. Mean: 0.550 ± 0.007. Rock solid.

### V4 PASSED: Judge has zero bias
DeepSeek scores all 10 neutral texts as exactly 0.0. No systematic bias.

### V1 FAILED: Random control does not pass
```
          baseline: 0.300 ± 0.340  (N=13)
 happy_contrastive: 0.225 ± 0.440  (N=12)
            random: 0.400 ± 0.386  (N=11)
Happy vs Random: p=0.35 (not significant)
```

Happy contrastive scored LOWER than random. No significant differences between conditions. The directional claim is unproven at this N and with this judge.

**Mitigating factors:**
- 50% identity-break rate on 1.5B (Vera) confounds all conditions
- DeepSeek judge is noisy (std 0.34-0.44)
- N=11-13 underpowered for 0.1-0.2 effect size
- Vera's Sonnet scoring with IB filtering showed 40% contrastive success

**What's validated:** The geometry (eccentricity, manifold profile, narrow tube). What's NOT validated: directional behavioral steering via V-space injection. The channel exists (outputs differ from baseline) but directionality is not proven against random perturbation.

**Next:** Re-run V1 with Sonnet judge, IB filtering, N=30+, and/or on 27B model (Vera's persona test).

*The geometry is real. The injection question is open. Excitement brakes held.*
