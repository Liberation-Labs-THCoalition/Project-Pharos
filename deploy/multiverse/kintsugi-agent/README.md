# Kintsugi Mutual Aid Agent — Megs's Workmate

Deployment package for the Kintsugi mutual aid coordination agent,
built on the Project-Kintsugi engine with CC's Ornith self-scaffolding
and OGPSA persona protection.

Target: Hetzner cax41 (ARM64). Inference runs on a separate box.

## Architecture

```
 46.224.162.211:8080            Hetzner cax41 (this box)
┌────────────────────┐    ┌───────────────────────────────────┐
│ ll-custom-slerp    │    │ Kintsugi Engine (FastAPI)          │
│ (inference model)  │◄───┤  - Megs persona (OGPSA-protected) │
└────────────────────┘    │  - Ornith self-scaffolding         │
                          │  - Pharos mutual aid packs         │
                          │  - Garuda poison tasting           │
                          ├───────────────────────────────────┤
                          │ PostgreSQL + pgvector              │
                          │  - Mnemosyne session persistence   │
                          │  - SIRA retrieval index            │
                          ├───────────────────────────────────┤
                          │ HippoRAG                           │
                          │  - Knowledge graph (triples)       │
                          │  - TGS-RAG fusion search           │
                          ├───────────────────────────────────┤
                          │ Dreamer (H-MEM consolidation)      │
                          │  - Temporal memory enrichment      │
                          │  - Graph cross-reference discovery │
                          └───────────────────────────────────┘
```

## Prerequisites

- Docker + Docker Compose (ARM64/aarch64)
- Network access to inference box (46.224.162.211:8080)
- Project-Kintsugi source: `git clone https://github.com/Liberation-Labs-THCoalition/Project-Kintsugi`
- Project-Mnemosyne reference: `https://github.com/Liberation-Labs-THCoalition/Project-Mnemosyne`

## Deployment

```bash
# 1. Clone this deployment package to the target box
scp -r kintsugi-agent/ root@<cax41-ip>:/opt/kintsugi/

# 2. SSH in and configure
ssh root@<cax41-ip>
cd /opt/kintsugi
cp config.env .env
# Edit .env: set POSTGRES_PASSWORD, verify LLM_URL

# 3. Copy Pharos mutual aid packs into ./packs/
# (from the pharos repo's packs/ directory — mutual aid subset)

# 4. Build and start
docker compose build
docker compose up -d

# 5. Verify
docker compose ps
curl http://localhost:8090/health
curl http://localhost:11235/health
```

## Pharos Packs (Mutual Aid Subset)

Place these pack directories in `./packs/`:
- benefits-navigation
- childcare-assistance
- civil-rights
- disability-accommodations
- emergency-financial-assistance
- mutual-aid-fundamentals
- mutual_aid (coordination triples)
- tenant-rights
- workers-rights
- cooperative-legal-structures
- disaster-recovery-navigation

## Mnemosyne Modules Active

- **SIRA**: "Think before you search" retrieval with significance scoring
- **HippoRAG**: Graph-backed TGS-RAG with PPR traversal
- **H-MEM**: Temporal consolidation via Dreamer (6h cycle)
- **Garuda**: Adversarial input tasting / quarantine
- **Enrichment**: Entity extraction + cross-reference

## Persona

The Megs persona (`persona/megs_persona.md`) is protected by OGPSA
(Orthogonal Gradient Persona Survival Architecture). The persona cannot
drift under adversarial prompting because the gradient survival gate
projects away from identity-modifying directions.

## License

Hippocratic License 3.0 with AI Welfare (SAFE-AI) module.
See LICENSE.md in the project root.
