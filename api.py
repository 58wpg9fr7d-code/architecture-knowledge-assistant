"""
ArchMind API server — FastAPI endpoint for the Chrome extension.

Start with:
    uvicorn api:app --host 0.0.0.0 --port 8765

Or for development:
    python api.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core import (
    query,
    get_secret,
    list_current_files,
)

app = FastAPI(
    title="ArchMind API",
    description="RAG-powered architectural knowledge assistant",
    version="0.1.0",
)

# In production, set ALLOWED_ORIGINS to a comma-separated allowlist.
# Keep the permissive default for local prototype use and extension testing.
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

# Allow the Chrome extension to call this API; production deployments should
# set ALLOWED_ORIGINS to the exact extension/app origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False if allowed_origins == ["*"] else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────
class AskRequest(BaseModel):
    question: str
    provider: str = "Groq Cloud"  # "Groq Cloud" or "Ollama 本地"
    model: str | None = None      # None → use default for the provider
    top_k: int = 4


class SourceItem(BaseModel):
    name: str
    content: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    files_count: int
    files: list[str]
    provider: str


# ── Routes ────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    """Health check — returns current document and provider status."""
    files = list_current_files()
    provider = "Groq Cloud" if get_secret("GROQ_API_KEY") else "Ollama 本地"
    return HealthResponse(
        status="ok",
        files_count=len(files),
        files=files,
        provider=provider,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Submit a question and get an AI-generated answer with cited sources."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空。")
    if not 1 <= req.top_k <= 8:
        raise HTTPException(status_code=400, detail="top_k 必须在 1 到 8 之间。")

    result = query(
        question=req.question.strip(),
        provider=req.provider,
        model=req.model,
        top_k=req.top_k,
    )

    return AskResponse(
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
        error=result["error"],
    )


# ── Run directly ──────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", "8765"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
