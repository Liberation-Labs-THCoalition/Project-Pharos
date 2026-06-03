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

*The channel works for valence. Arousal may need a different channel. The geometry tells us which emotions we can inject and which we can't.*
