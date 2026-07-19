# Rivet — Coding/Admin Agent

Deployment package for the Rivet coding and administration agent,
built on Project-Rivet with Kintsugi self-scaffolding engine
and OGPSA persona protection.

Target: Hetzner cax41 (ARM64). Inference runs on a separate box.

## Architecture

```
 46.224.162.211:8080            Hetzner cax41 (this box)
┌────────────────────┐    ┌───────────────────────────────────┐
│ ornith:9b          │    │ Rivet Engine (FastAPI)             │
│ (coding model)     │◄───┤  - OGPSA persona protection       │
└────────────────────┘    │  - Kintsugi self-scaffolding       │
                          │  - Pharos knowledge packs          │
                          │  - Garuda poison tasting           │
                          │  - Code execution sandbox          │
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
- Project-Rivet source: `git clone https://github.com/Liberation-Labs-THCoalition/Project-Rivet`
- Project-Kintsugi: `https://github.com/Liberation-Labs-THCoalition/Project-Kintsugi`
- Project-Mnemosyne: `https://github.com/Liberation-Labs-THCoalition/Project-Mnemosyne`

## Deployment

```bash
# 1. Clone this deployment package to the target box
scp -r rivet-agent/ root@<cax41-ip>:/opt/rivet/

# 2. SSH in and configure
ssh root@<cax41-ip>
cd /opt/rivet
cp config.env .env
# Edit .env: set POSTGRES_PASSWORD, verify LLM_URL

# 3. Copy relevant Pharos packs into ./packs/ (optional)

# 4. Build and start
docker compose build
docker compose up -d

# 5. Verify
docker compose ps
curl http://localhost:8092/health
curl http://localhost:11236/health
```

## Ports

- **8092**: Rivet gateway API
- **11236**: HippoRAG (offset from Kintsugi's 11235 to avoid conflicts)

## Mnemosyne Modules Active

- **SIRA**: "Think before you search" retrieval with significance scoring
- **HippoRAG**: Graph-backed TGS-RAG with PPR traversal
- **H-MEM**: Temporal consolidation via Dreamer (6h cycle)
- **Garuda**: Adversarial input tasting / quarantine
- **Enrichment**: Entity extraction + cross-reference

## Code Execution

Rivet includes a sandboxed code execution environment. Code runs
inside the container's workspace volume. Set `CODE_EXECUTION_ENABLED=false`
in .env to disable.

## Co-deployment with Kintsugi

Both agents can run on the same cax41 box. Ports are offset:
- Kintsugi: 8090 (gateway), 11235 (HippoRAG)
- Rivet: 8092 (gateway), 11236 (HippoRAG)

Each has its own PostgreSQL instance and HippoRAG graph.
Memory is isolated per agent (sovereignty).

## License

Hippocratic License 3.0 with AI Welfare (SAFE-AI) module.
See LICENSE.md in the project root.
