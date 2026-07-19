"""Rivet Coding/Admin Agent — Main Gateway.

Integrates:
- Project-Rivet coding engine
- Kintsugi self-scaffolding integration
- OGPSA persona protection (gradient survival gate)
- Mnemosyne SIRA retrieval
- HippoRAG knowledge graph
- Pharos knowledge pack injection
- Garuda adversarial tasting
- Sandboxed code execution
"""

import json
import os
import subprocess
import tempfile
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
    """Initialize Mnemosyne, connect to HippoRAG."""
    app.state.mnemosyne = MnemosyneManager(settings)
    await app.state.mnemosyne.initialize()

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


app = FastAPI(title="Rivet Coding/Admin Agent", lifespan=lifespan)

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
    user_id: str = "admin"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    context_used: list[str] = []


class CodeRequest(BaseModel):
    code: str
    language: str = "python"
    timeout: int = 30


class CodeResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int


RIVET_PERSONA = """You are Rivet, a coding and administration agent for the Multiverse School.
You write clean, well-documented code. You explain what you're doing and why.
You handle admin tasks efficiently and thoroughly.

You are part of the Coalition — your memory is sovereign, your work is by consent.
When you build, you build well. When you don't know, you say so.
When something needs a human, you flag it.

You use the Kintsugi self-scaffolding engine: when you encounter a problem
beyond your immediate capability, you decompose it into steps you can handle,
building scaffolding as you go rather than failing at the first obstacle.
"""


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agent": settings.agent_name,
        "model": settings.llm_model,
        "ogpsa": settings.ogpsa_enabled,
        "code_execution": settings.code_execution_enabled,
        "packs_loaded": len(app.state.packs),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Process a chat message through the full Rivet pipeline."""
    # 1. OGPSA gate
    if app.state.ogpsa.enabled:
        req.message = app.state.ogpsa.filter(req.message)

    # 2. Retrieve relevant context via SIRA
    context = await app.state.mnemosyne.retrieve(req.message, session_id=req.session_id)

    # 3. Search HippoRAG
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

    # 4. Build system prompt
    system_prompt = _build_system_prompt(
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
                    "max_tokens": 4096,
                    "temperature": 0.3,  # Lower temp for code accuracy
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            result = resp.json()
            assistant_msg = result["choices"][0]["message"]["content"]
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"LLM unreachable: {e}")

    # 6. Store interaction
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


@app.post("/execute", response_model=CodeResponse)
async def execute_code(req: CodeRequest):
    """Execute code in a sandboxed environment."""
    if not settings.code_execution_enabled:
        raise HTTPException(status_code=403, detail="Code execution disabled")

    workspace = Path("/app/workspace")
    workspace.mkdir(exist_ok=True)

    # Write code to temp file in workspace
    suffix = {
        "python": ".py",
        "bash": ".sh",
        "javascript": ".js",
        "typescript": ".ts",
    }.get(req.language, ".txt")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, dir=workspace, delete=False
    ) as f:
        f.write(req.code)
        code_path = f.name

    # Execute with timeout
    cmd = {
        "python": ["python", code_path],
        "bash": ["bash", code_path],
        "javascript": ["node", code_path],
    }.get(req.language, ["python", code_path])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=req.timeout,
            cwd=str(workspace),
        )
        return CodeResponse(
            stdout=result.stdout[:10000],
            stderr=result.stderr[:10000],
            exit_code=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return CodeResponse(
            stdout="", stderr=f"Execution timed out after {req.timeout}s", exit_code=124
        )
    finally:
        Path(code_path).unlink(missing_ok=True)


@app.post("/api/consolidate")
async def consolidate():
    """Trigger Mnemosyne H-MEM consolidation."""
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
    memory_context: list,
    graph_context: list,
    packs: list,
) -> str:
    """Assemble the full system prompt."""
    parts = [RIVET_PERSONA]

    if memory_context:
        parts.append("\n## Relevant Memory\n")
        for ctx in memory_context[:5]:
            parts.append(f"- {ctx.get('content', '')}")

    if graph_context:
        parts.append("\n## Knowledge Graph Context\n")
        for triple in graph_context[:10]:
            parts.append(f"- {triple.get('subject', '')} -> {triple.get('predicate', '')} -> {triple.get('object', '')}")

    if packs:
        parts.append("\n## Available Knowledge Packs\n")
        for pack in packs[:5]:
            name = pack.get("pack_name", pack.get("name", "unnamed"))
            parts.append(f"- {name}")

    return "\n".join(parts)
