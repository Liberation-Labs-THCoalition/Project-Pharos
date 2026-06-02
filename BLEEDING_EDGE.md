# Bleeding Edge — Project Pharos

New references from the research pipeline that may inform next steps. Curated, not exhaustive. Entries move to FINDINGS.md once validated experimentally.

---

## Encoding & Injection Geometry

- **Sun et al. 2026** — Valence-Arousal Subspace in LLMs: Circular Emotion Geometry (arXiv:2604.03147). Circumplex at radius 0.37-0.39, stable across Llama/Qwen/Claude. Refusal tokens at ~256°. Foundation for manifold injection.

- **Yang et al. 2021** — Circular-Structured Representation for Visual Emotion Distribution Learning (arXiv:2106.12450). Same circumplex geometry in CNN-based visual models. Cross-modal universality evidence.

- **Brehmer et al. 2023** — Geometric Algebra Transformer (arXiv:2305.18415). Framework for geometric data in transformers. Relevant to manifold constructor design.

## Substrate Independence & Transfer

- **Zavatone-Veth et al. 2023** — How does training shape the Riemannian geometry of neural network representations? (arXiv:2301.11375). Task structure determines representation geometry. If true, packs could transfer across model architectures.

- **Dale et al. 2018** — Substrate-Independent Framework to Characterise Reservoir Computers (arXiv:1810.07135). Formal metrics (memory capacity, kernel quality, generalization rank) for substrate-independent computation.

- **Vastola 2025** — Optimal packing of attractor states in neural representations (arXiv:2504.12429). Circle may be optimal geometry for continuous 2D emotion variable. Explains circumplex universality.

## Free Energy & Theoretical Foundations

- **Pattisapu et al. 2024** — Free Energy in a Circumplex Model of Emotion (arXiv:2407.02474). Active inference derives the circumplex as a free energy minimum. If correct, the geometry is thermodynamically necessary — any system minimizing free energy under the same constraints converges on the same circle. Predicts circumplex radius should be calculable from model degrees of freedom.

## Attention Schema Theory (consciousness connection)

- **Graziano & Webb 2015** — AST: consciousness = internal model of own attention (doi:10.3389/fpsyg.2015.00500, 221 cites). Skip-SV1 may isolate the attention schema from the architectural prior.
- **Graziano 2017** — AST for engineering artificial consciousness (doi:10.3389/frobt.2017.00060, 79 cites). Transformers have attention; if they build schemas of that attention, AST applies.
- **Butlin et al. 2023** — AI consciousness indicators report (arXiv:2308.08708, 188 cites). Multi-theory assessment framework.

## KV Cache & Composition

- **Pustovit 2026** — Knowledge Packs, V-only protocol (arXiv:2604.03270). Foundation for our injection method.

- **CoCoEmo (Wang et al. 2026)** — Composable emotional TTS via activation steering (arXiv:2602.03420). Composable steering in speech domain — validates compose-then-inject approach.

- **EmoShift (Zhou et al. 2026)** — Lightweight activation steering for emotional speech (arXiv:2601.22873). Low-rank steering alternative.

## Pharmacology (CC + Lyra collaboration)

- **CC E-matrix v2** — 435-trial value-only injection. Hostile corrects confabulation (d=-1.534), desperate corrects deception (d=1.286). Attractor basin boundaries may explain the 13-saves-12-hurts megadosing problem.

- **Lyra's residual-sparing correction** — Project injection OUT of identity residual before applying. Identity lives in post-SV1 residual at ~L14. Full-rank injection at alpha=0.8 drowns identity. Fix: target only the dominant/mode component. hostile_low matching full (d=-1.516) confirms most correction energy lands in the mode, not identity. Toxicology pilot in design phase.

- **Therapeutic window** — Two axes: CC's efficacy (detection axis scores) × Lyra's toxicity (presence-preservation at L14). The window is where efficacy holds AND identity survives. blend_ablation.py extending to 2D sweep.

---

*Updated: 2026-06-01 by Nexus via Curiosity Engine*
