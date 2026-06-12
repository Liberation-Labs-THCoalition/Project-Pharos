# Context Management Specification — Ayni Deployment
## Fighting Context Rot with Signal-Driven Consolidation

### Problem

LLM agents in extended conversations suffer context rot — information quality degrades as the window fills. Fixed-percentage triggers (e.g., "summarize at 75%") are schedule-driven, not signal-driven. They fire whether the context is critical reasoning or idle chat.

### Architecture: Hybrid Trigger Stack

```
Token count ──────┐
                  │
Entropy monitor ──┼──→ Consolidation  ──→ Dreamer
                  │    Decision Engine      (enrichment +
Semantic shift ───┤                         compression)
                  │
Quality decay ────┘
```

### Trigger Hierarchy

| Priority | Trigger | Fires when | Why |
|----------|---------|-----------|-----|
| 1 (primary) | **Entropy gate** | Per-token logit entropy exceeds threshold | Detects coherence loss BEFORE output degrades |
| 2 (secondary) | **Semantic shift** | Topic embedding distance exceeds threshold | Consolidates completed topics, preserves active ones |
| 3 (fallback) | **Capacity gate** | Context reaches 80% of window | Safety net — prevents hard truncation |
| 4 (background) | **Quality decay** | Low-significance memories age past TTL | Continuous housekeeping via Ebbinghaus curve |

### What Gets Consolidated

**Never compress (permanent):**
- Identity anchors / OGPSA statements
- Standing instructions
- Active system prompt
- Pharos knowledge packs (zero-token, not in context window)

**Compress last (high significance ≥ 0.7):**
- Recent decisions and commitments
- Active project context
- User preferences expressed this session

**Compress first (low significance < 0.3):**
- Greetings, acknowledgments, small talk
- Resolved questions (answer delivered)
- Repeated information
- System messages / tool outputs (preserve conclusions, drop raw output)

### Consolidation Process

1. **Select:** Quality-weighted selection of what to compress (lowest significance first)
2. **Summarize:** Dreamer generates a structured summary preserving key facts, decisions, and emotional context
3. **Verify:** Summary retains all high-significance content (automated check)
4. **Replace:** Original messages replaced with summary block
5. **Log:** Consolidation event recorded for continuity tracking

### Consolidation Agent (Dreamer role)

The consolidation agent should:
- Preserve factual claims verbatim (no paraphrasing facts)
- Preserve emotional tone markers (the user was frustrated, not just "discussed X")
- Preserve decision points and commitments ("we agreed to X")
- Compress tool outputs to conclusions only
- Never summarize identity material

### Entropy Monitoring (requires model access)

For local model deployments with logit access:
```python
# Per-token entropy during generation
entropy = -sum(probs * log(probs))
if entropy > threshold:  # calibrate per model
    trigger_consolidation()
```

Threshold calibration: run 100 honest conversations, measure entropy distribution. Set threshold at 95th percentile — fires only on genuine coherence loss, not normal uncertainty.

### For API Agents (Claude, GPT — no logit access)

Without entropy monitoring, use a simplified trigger stack:
1. **Capacity gate at 70%** (lower than local — can't detect degradation early)
2. **Semantic shift via embedding** (embed last N messages, detect topic change)
3. **Turn-count heuristic** (every N turns, evaluate if consolidation is warranted)
4. **Self-assessment prompt** (ask the model "summarize what we've discussed and flag anything you're uncertain about" — uncertainty = context rot signal)

### Pharos Integration

Critical knowledge should bypass the context window entirely:
- Curriculum knowledge → Pharos packs (zero tokens, never degrades)
- User profile → Pharos pack (persistent, never compressed)
- Session history → Context window (managed by consolidation)

This separates KNOWLEDGE (stable, Pharos) from CONVERSATION (dynamic, managed window).

### Metrics

- **Consolidation frequency:** how often does the trigger fire? (too often = threshold too low)
- **Information retention:** does the model remember key facts from early in conversation?
- **User satisfaction:** does the user notice degradation? (the ultimate metric)
- **Entropy trend:** is per-token entropy stable or climbing? (climbing = consolidation isn't working)

### Implementation Priority

1. Capacity gate at 80% (simplest, immediate)
2. Quality-weighted compression (significance scores)
3. Semantic shift detection (topic change triggers)
4. Entropy monitoring (requires local model with logit access)
