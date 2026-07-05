# Project Pharos

**Zero-token knowledge injection for AI agents.**

Pharos encodes domain knowledge as walk-encoded triples — structured facts that ground AI reasoning without consuming context window tokens. 126 packs covering law, science, crisis resources, security, and more.

## What It Does

Give any AI agent instant domain expertise:

```bash
# Query the pack server
curl -s localhost:8300/query -d '{"pack": "statistical-pitfalls", "query": "p-hacking"}'

# Search all 126 packs at once
curl -s localhost:8300/search_all -d '{"query": "informed consent", "max_per_pack": 3}'
```

## The Library

**126 packs • 6,393 triples • Growing weekly**

| Category | Packs | Examples |
|----------|-------|---------|
| Legal (50 states) | 50 | Court rules, filing procedures, eviction defense |
| Crisis & Mutual Aid | 8 | Crisis hotlines, harm reduction, tenant rights, benefits navigation |
| Science & Engineering | 12 | Physics, chemistry, electrical, mechanical, materials science |
| AI/ML Security | 4 | Framework injection, Supabase RLS, security fundamentals |
| Research Methods | 4 | Statistical pitfalls, causal inference, experiment design, logical fallacies |
| Government | 4 | Legislative procedure, administrative law, federal evidence, civil procedure |
| Community Skills | 15+ | Bicycle repair, food preservation, solar off-grid, cooperative law |

## Architecture

```
┌─────────────────────────────────────┐
│         Pack Server (:8300)          │
│   In-memory index • Fuzzy search    │
│   Entity lookup • Graph traversal   │
├─────────────────────────────────────┤
│    126 Knowledge Packs (triples)    │
│    Subject → Predicate → Object     │
└─────────────────────────────────────┘
         ↑ query            ↑ query
    Meridian Agents      Any HTTP client
    Bounty Fleet         LLM pipelines
    Ayni / Multiverse    Your tools
```

## Quick Start

```bash
# Clone
git clone https://github.com/Liberation-Labs-THCoalition/Project-Pharos.git
cd Project-Pharos

# Start the pack server
python3 pack_server.py --port 8300

# Query
curl -s localhost:8300/packs | python3 -m json.tool
```

## Pack Format

Each pack is a `triples.json`:

```json
{
  "description": "Statistical pitfalls in ML research",
  "sources": ["Cohen 1988", "Ioannidis 2005"],
  "triples": [
    {
      "subject": "p-hacking",
      "predicate": "detection_method",
      "object": "pre-register hypotheses and analysis plans before data collection"
    }
  ]
}
```

## Build Your Own Pack

```bash
# From an API
python3 pack_generator/generate.py --source https://api.example.com/data --topic "your domain"

# From a document
python3 pack_generator/generate.py --document paper.pdf --topic "key findings"
```

## Part of the Coalition Stack

Pharos integrates with:
- **[Meridian](../Project-Agni/)** — Research swarm uses packs for domain grounding
- **[Ayni](https://github.com/Liberation-Labs-THCoalition/Project-Ayni)** — Companion agent knowledge layer
- **[Multiverse Agent](https://github.com/Liberation-Labs-THCoalition/multiverse-agent)** — Mutual aid coordination

## License

Apache 2.0 — free to use, modify, and distribute.

---

*Built by [Liberation Labs](https://thcoalition.org) • [GitHub](https://github.com/Liberation-Labs-THCoalition/Project-Pharos) • [Discord](https://discord.gg/vKm7GwXHv)*
