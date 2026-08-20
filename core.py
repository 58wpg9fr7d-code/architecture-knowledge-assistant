"""
ArchMind core module — shared by Streamlit app, API server, and any other frontend.

Provides:
- Document loading & chunking
- Embeddings (hash-based fallback + HuggingFace optional)
- ChromaDB vector store
- LLM calling (Ollama local / Gemini Cloud)
- Prompt construction
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_community.vectorstores import Chroma
except Exception:
    Chroma = None

# ── Paths & constants ─────────────────────
APP_DIR = Path(__file__).parent
DOCUMENTS_DIR = APP_DIR / "documents"
DB_DIR = APP_DIR / ".chroma_db"
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
# Gemini 2.5 Flash is no longer available on the current Gemini API endpoint.
# Ignore that stale deployment setting while keeping newer overrides possible.
_configured_gemini_model = os.getenv("GEMINI_MODEL", "").strip()
DEFAULT_GEMINI_MODEL = (
    "gemini-3.5-flash"
    if not _configured_gemini_model or _configured_gemini_model == "gemini-2.5-flash"
    else _configured_gemini_model
)
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


# ── Secret / env helpers ──────────────────
def get_secret(name: str, default: str | None = None) -> str | None:
    """Read a secret from env, falling back to Streamlit secrets if available."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


# ── Embeddings ────────────────────────────
class HashEmbeddings(Embeddings):
    """Lightweight hash-based embeddings — zero dependencies, decent for CJK text."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = self._tokens(text)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        lowered = text.lower()
        words: list[str] = []
        current: list[str] = []
        for char in lowered:
            if char.isalnum() or "一" <= char <= "鿿":
                current.append(char)
            elif current:
                words.append("".join(current))
                current = []
        if current:
            words.append("".join(current))
        cjk = [c for c in lowered if "一" <= c <= "鿿"]
        return (
            words
            + cjk
            + [a + b for a, b in zip(cjk, cjk[1:])]
            + [a + b + c for a, b, c in zip(cjk, cjk[1:], cjk[2:])]
        )


_embeddings_instance: Embeddings | None = None


def get_embeddings() -> Embeddings:
    """Return an Embeddings instance (singleton).

    The lightweight hash embedding is the default for a reliable cloud demo.
    HuggingFace can be enabled explicitly with ENABLE_HF_EMBEDDINGS=1.
    """
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance

    is_hf_enabled = str(get_secret("ENABLE_HF_EMBEDDINGS", "")).lower() in {"1", "true", "yes"}
    if not is_hf_enabled:
        _embeddings_instance = HashEmbeddings()
        return _embeddings_instance

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        _embeddings_instance = HashEmbeddings()

    return _embeddings_instance


# ── Document loading ──────────────────────
def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def load_documents() -> list[Document]:
    """Load all supported documents from DOCUMENTS_DIR."""
    docs: list[Document] = []
    if not DOCUMENTS_DIR.exists():
        return docs
    for path in sorted(DOCUMENTS_DIR.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = read_pdf(path) if path.suffix.lower() == ".pdf" else read_txt(path)
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": path.name, "path": str(path)},
            ))
    return docs


def split_documents(documents: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "；", ";", ".", " ", ""],
    )
    return splitter.split_documents(list(documents))


# ── File helpers ──────────────────────────
def safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    stem = re.sub(r"[^0-9A-Za-z._\-一-鿿]+", "_", Path(name).stem).strip("._")
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise RuntimeError("只支持 TXT、MD 或 PDF 文件。")
    return f"{stem or 'document'}{suffix}"


def save_uploaded_file(file_bytes: bytes, filename: str) -> str:
    """Save a single uploaded file to DOCUMENTS_DIR. Returns the saved filename."""
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    dest_name = safe_filename(filename)
    dest_path = DOCUMENTS_DIR / dest_name
    dest_path.write_bytes(file_bytes)
    return dest_name


def list_current_files() -> list[str]:
    """Return sorted list of supported filenames in DOCUMENTS_DIR."""
    if not DOCUMENTS_DIR.exists():
        return []
    return sorted(
        f.name for f in DOCUMENTS_DIR.iterdir()
        if f.suffix.lower() in SUPPORTED_SUFFIXES
    )


# ── Vector store ──────────────────────────
def build_vector_store(force_rebuild: bool = False):
    """Build (or load) the ChromaDB vector store from documents."""
    if Chroma is None:
        raise RuntimeError(
            "ChromaDB is not installed. Run: pip install chromadb langchain-community"
        )
    if force_rebuild and DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    embeddings = get_embeddings()
    if DB_DIR.exists() and any(DB_DIR.iterdir()):
        return Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
    docs = load_documents()
    if not docs:
        raise RuntimeError("documents 文件夹里还没有可读取的文件。")
    chunks = split_documents(docs)
    if not chunks:
        raise RuntimeError("文档已读取，但没有切分出可检索的内容。")
    return Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=str(DB_DIR)
    )


def reset_vector_store() -> None:
    """Delete the vector store directory and clear the embeddings cache."""
    global _embeddings_instance
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    _embeddings_instance = None


# ── LLM calling ───────────────────────────
def ask_ollama(prompt: str, model: str | None = None) -> str:
    model = model or DEFAULT_OLLAMA_MODEL
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}},
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def ask_gemini(prompt: str, model: str | None = None) -> str:
    model = model or DEFAULT_GEMINI_MODEL
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 GEMINI_API_KEY。")
    r = requests.post(
        GEMINI_API_URL.format(model=model),
        params={"key": api_key},
        headers={
            "Content-Type": "application/json",
        },
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        },
        timeout=120,
    )
    r.raise_for_status()
    try:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Gemini 返回了无法解析的结果。") from exc


def ask_llm(prompt: str, provider: str = "Gemini Cloud", model: str | None = None) -> str:
    """Call the LLM. `provider` is 'Gemini Cloud' or 'Ollama 本地'."""
    if provider == "Gemini Cloud":
        return ask_gemini(prompt, model)
    return ask_ollama(prompt, model)


# ── Prompt ────────────────────────────────
def make_prompt(question: str, docs: list[Document]) -> str:
    context = "\n\n".join(
        f"[来源 {i}: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )
    return (
        f"你是一个建筑知识库助手。请只根据下面的资料回答问题。\n"
        f"如果资料中没有答案，请明确说\"当前文献中没有足够信息\"。\n"
        f"回答要简洁、专业，并在关键结论后标注来源编号。\n\n"
        f"资料：\n{context}\n\n"
        f"问题：\n{question}\n\n"
        f"回答："
    )


# ── High-level RAG query ──────────────────
def query(question: str, provider: str = "Gemini Cloud", model: str | None = None,
          top_k: int = 4, force_rebuild: bool = False) -> dict:
    """Run a full RAG query: retrieve → generate → return answer + sources.

    Returns:
        {"answer": str, "sources": [{"name": str, "content": str}, ...], "error": str | None}
    """
    result: dict = {"answer": "", "sources": [], "error": None}

    try:
        vector_store = build_vector_store(force_rebuild=force_rebuild)
    except Exception as exc:
        result["error"] = str(exc)
        return result

    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    related_docs = retriever.invoke(question)

    if not related_docs:
        result["error"] = "没有检索到相关文献片段。"
        return result

    prompt = make_prompt(question, related_docs)

    try:
        answer = ask_llm(prompt, provider, model)
    except requests.exceptions.ConnectionError:
        result["error"] = f"无法连接 {provider}。请确认服务已启动。"
        return result
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        result["error"] = f"模型服务返回错误（HTTP {status_code}）。"
        return result
    except Exception as exc:
        result["error"] = f"调用模型失败：{exc}"
        return result

    result["answer"] = answer
    result["sources"] = [
        {"name": doc.metadata.get("source", "unknown"), "content": doc.page_content}
        for doc in related_docs
    ]
    return result
