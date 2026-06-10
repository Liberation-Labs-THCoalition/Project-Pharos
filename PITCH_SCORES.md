# Pharos Knowledge Packs — Quality Assessment
## For The Multiverse School (Friday Pitch)

### Overall Improvement by Pack

| Pack | Triples | Baseline Avg | Pharos Avg | Lift | Highest Delta Question |
|------|---------|-------------|-----------|------|----------------------|
| AI Alignment & Ethics | 103 | 2.8/5 | 4.5/5 | +61% | RLHF → sycophancy (+90%) |
| Agentic AI Systems | 96 | 2.8/5 | 4.8/5 | +73% | Memory systems (+67%) |
| Cybersecurity | 96 | 3.2/5 | 4.8/5 | +52% | Privilege escalation (+67%) |
| Prompt Engineering | 88 | 3.0/5 | 5.0/5 | +67% | Prompt injection defense (+150%) |
| Mech Interpretability | 96 | 2.2/5 | 4.8/5 | +118% | SAE decomposition (+150%) |

### Highest-Impact Triple Per Pack

| Pack | Triple | Why It Matters |
|------|--------|---------------|
| Alignment | `RLHF can_cause sycophancy through reward model exploitation` | Forces the mechanism, not just the description |
| Agentic | `ReAct trace consists_of Thought-Action-Observation triplets` | Separates "explain" from "implement" |
| Cybersecurity | `ROP bypasses non-executable stack protection` | The missing piece in why NX alone isn't sufficient |
| Prompt Eng | `indirect injection is_especially_dangerous_for RAG and agentic tool use` | Closes the most dangerous gap for builders |
| Mech Interp | `residual stream is the central communication channel all layers read/write` | The conceptual pivot everything else rests on |

### Three Pitch Arguments

1. **Typed predicates preserve distinctions prose elides.** `is_type_of` forces taxonomy. `can_cause through` forces mechanism. Students learn categories and causal paths, not synonym lists.

2. **Highest delta on mechanism questions.** Triples are relational; mechanisms are relational. The match is structural. Definition questions show moderate improvement; "how does X cause Y" questions show 90-150% improvement.

3. **Mech interp shows the largest gap (118%).** The field moves faster than parametric knowledge can track. A structured knowledge pack delivers current vocabulary (SAEs, steering vectors, KV geometry) that no training corpus keeps current.

### The Multiverse-Specific Value Prop

Each pack maps directly to a course in their catalog:
- Alignment → AI Alignment (4-week course)
- Agentic → Intro to Agents (5-week course)  
- Cybersecurity → Hacking fundamentals + CTF competitions
- Prompt Engineering → Prompt Engineering (6-week course)
- Mech Interp → Research-to-quiz agent + advanced topics

The prompt injection defense question (+150%) is the single most important result for the pitch. Multiverse students build RAG pipelines from week 1. The pack teaches them that retrieved documents are an attack surface — baseline parametric knowledge consistently misses this.
