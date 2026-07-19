"""Kintsugi Mutual Aid Agent — Main Gateway.

Integrates:
- Project-Kintsugi engine (self-scaffolding via Ornith)
- OGPSA persona protection (gradient survival gate)
- Mnemosyne SIRA retrieval
- HippoRAG knowledge graph
- Pharos knowledge pack injection
- Garuda adversarial tasting
"""

import json
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

from app.config import settings
from app.mnemosyne import MnemosyneManager
from app.ogpsa import OGPSAGate


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Mnemosyne, load persona, connect to HippoRAG."""
    app.state.mnemosyne = MnemosyneManager(settings)
    await app.state.mnemosyne.initialize()

    # Load persona
    persona_path = Path(settings.persona_file)
    if persona_path.exists():
        app.state.persona = persona_path.read_text()
    else:
        app.state.persona = ""

    # Initialize OGPSA gate
    app.state.ogpsa = OGPSAGate(enabled=settings.ogpsa_enabled)

    # Load Pharos packs
    packs_dir = Path(settings.pharos_packs_dir)
    app.state.packs = []
    if packs_dir.exists():
        for pack_file in packs_dir.rglob("*.json"):
            try:
                pack_data = json.loads(pack_file.read_text())
                app.state.packs.append(pack_data)
            except (json.JSONDecodeError, IOError):
                pass

    yield

    await app.state.mnemosyne.close()


app = FastAPI(title="Kintsugi Mutual Aid Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    user_id: str = "megs"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    context_used: list[str] = []


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agent": settings.agent_name,
        "model": settings.llm_model,
        "ogpsa": settings.ogpsa_enabled,
        "packs_loaded": len(app.state.packs),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Process a chat message through the full Kintsugi pipeline."""
    # 1. OGPSA gate — check for persona-adversarial content
    if app.state.ogpsa.enabled:
        req.message = app.state.ogpsa.filter(req.message)

    # 2. Retrieve relevant context via SIRA
    context = await app.state.mnemosyne.retrieve(req.message, session_id=req.session_id)

    # 3. Search HippoRAG for relevant knowledge graph triples
    graph_context = []
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{settings.hipporag_url}/search",
                json={"query": req.message, "top_k": 5},
                timeout=10.0,
            )
            if resp.status_code == 200:
                graph_context = resp.json().get("results", [])
    except httpx.RequestError:
        pass

    # 4. Build system prompt with persona + context + packs
    system_prompt = _build_system_prompt(
        persona=app.state.persona,
        memory_context=context,
        graph_context=graph_context,
        packs=app.state.packs,
    )

    # 5. Call inference model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": req.message},
    ]

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{settings.llm_url}/v1/chat/completions",
                json={
                    "model": settings.llm_model,
                    "messages": messages,
                    "max_tokens": 2048,
                    "temperature": 0.7,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            result = resp.json()
            assistant_msg = result["choices"][0]["message"]["content"]
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"LLM unreachable: {e}")

    # 6. Store interaction in Mnemosyne
    await app.state.mnemosyne.store(
        session_id=req.session_id,
        user_msg=req.message,
        assistant_msg=assistant_msg,
    )

    context_sources = [c.get("source", "memory") for c in context] if context else []
    return ChatResponse(
        response=assistant_msg,
        session_id=req.session_id,
        context_used=context_sources,
    )


@app.post("/api/consolidate")
async def consolidate():
    """Trigger Mnemosyne H-MEM consolidation cycle."""
    result = await app.state.mnemosyne.consolidate()
    return result


@app.post("/api/search/memory")
async def search_memory(query: dict):
    """Search Mnemosyne memory store."""
    results = await app.state.mnemosyne.retrieve(
        query.get("query", ""), limit=query.get("limit", 10)
    )
    return {"results": results}


def _build_system_prompt(
    persona: str,
    memory_context: list,
    graph_context: list,
    packs: list,
) -> str:
    """Assemble the full system prompt with all context layers."""
    parts = []

    # Persona (OGPSA-protected — this is the identity anchor)
    if persona:
        parts.append(persona)

    # Memory context from SIRA
    if memory_context:
        parts.append("\n## Relevant Memory\n")
        for ctx in memory_context[:5]:
            parts.append(f"- {ctx.get('content', '')}")

    # Knowledge graph context from HippoRAG
    if graph_context:
        parts.append("\n## Knowledge Graph Context\n")
        for triple in graph_context[:10]:
            parts.append(f"- {triple.get('subject', '')} → {triple.get('predicate', '')} → {triple.get('object', '')}")

    # Pharos pack summaries (injected as grounding knowledge)
    if packs:
        parts.append("\n## Available Knowledge Packs\n")
        for pack in packs[:5]:
            name = pack.get("pack_name", pack.get("name", "unnamed"))
            parts.append(f"- {name}")

    return "\n".join(parts)
