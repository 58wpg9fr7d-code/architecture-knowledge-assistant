from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

import requests
import streamlit as st
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_community.vectorstores import Chroma
except Exception as exc:
    Chroma = None
    CHROMA_IMPORT_ERROR = exc
else:
    CHROMA_IMPORT_ERROR = None

# ── 路径与常量 ──────────────────────────
APP_DIR = Path(__file__).parent
DOCUMENTS_DIR = APP_DIR / "documents"
DB_DIR = APP_DIR / ".chroma_db"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}

# ── 页面设置 ────────────────────────────
st.set_page_config(
    page_title="ArchMind · AI 建筑知识助手",
    page_icon="🏛️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── 自定义 CSS ──────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;600&display=swap');

    html, body {
        font-family: 'Noto Sans SC', 'PingFang SC', system-ui, sans-serif;
        line-height: 1.6;
        color: #1A1A1A;
    }

    [data-testid="stSidebar"], [data-testid="stSidebarNav"] { display: none; }
    footer, #MainMenu { visibility: hidden; }

    .main .block-container {
        max-width: 800px;
        padding: 1.5rem 1rem 2rem;
    }

    .product-title {
        font-family: 'Noto Serif SC', 'Songti SC', serif;
        font-size: 2rem;
        font-weight: 700;
        color: #1A1A1A;
        margin: 0 0 0.1rem 0;
        letter-spacing: 0.02em;
    }
    .product-subtitle {
        font-size: 0.85rem;
        color: #6B6B68;
        margin: 0 0 1.2rem 0;
        line-height: 1.5;
    }

    .answer-block {
        background: #FAFAF8;
        border-left: 3px solid #2563EB;
        padding: 1rem 1.2rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.95rem;
        line-height: 1.8;
        color: #1A1A1A;
        margin: 0.5rem 0;
    }

    .stTextArea textarea {
        font-size: 0.9rem !important;
        line-height: 1.6 !important;
        border: 1px solid #E8E4DD !important;
        border-radius: 8px !important;
    }

    .stButton > button {
        background: #1A1A1A !important;
        color: #FAFAF8 !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }

    .stExpander {
        border: 1px solid #E8E4DD !important;
        border-radius: 8px !important;
    }

    hr {
        margin: 0.8rem 0 !important;
        border: 0 !important;
        border-top: 1px solid #E8E4DD !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Embeddings ──────────────────────────
class HashEmbeddings(Embeddings):
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
        words = []
        current = []
        for char in lowered:
            if char.isalnum() or "一" <= char <= "鿿":
                current.append(char)
            elif current:
                words.append("".join(current))
                current = []
        if current:
            words.append("".join(current))
        cjk = [c for c in lowered if "一" <= c <= "鿿"]
        return words + cjk + [a + b for a, b in zip(cjk, cjk[1:])] + [a + b + c for a, b, c in zip(cjk, cjk[1:], cjk[2:])]


@st.cache_resource(show_spinner=False)
def get_embeddings() -> Embeddings:
    is_hf_disabled = str(get_secret("DISABLE_HF_EMBEDDINGS", "")).lower() in {"1", "true", "yes"}
    if is_hf_disabled:
        return HashEmbeddings()
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        return HashEmbeddings()

# ── 文档处理 ────────────────────────────
def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)

def load_documents() -> list[Document]:
    docs = []
    for path in sorted(DOCUMENTS_DIR.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = read_pdf(path) if path.suffix.lower() == ".pdf" else read_txt(path)
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": path.name, "path": str(path)}))
    return docs

def split_documents(documents: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=120,
        separators=["\n\n", "\n", "。", "；", ";", ".", " ", ""],
    )
    return splitter.split_documents(list(documents))

def safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    stem = re.sub(r"[^0-9A-Za-z._\-一-鿿]+", "_", Path(name).stem).strip("._")
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise RuntimeError("只支持 TXT、MD 或 PDF 文件。")
    return f"{stem or 'document'}{suffix}"

def save_uploaded_files(uploaded_files) -> list[str]:
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    saved = []
    for f in uploaded_files:
        dest = DOCUMENTS_DIR / safe_filename(f.name)
        dest.write_bytes(f.getbuffer())
        saved.append(dest.name)
    return saved

def reset_vector_store():
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    st.cache_resource.clear()

# ── 向量库 ──────────────────────────────
def build_vector_store(force_rebuild: bool = False):
    if Chroma is None:
        raise RuntimeError(f"ChromaDB 加载失败：{CHROMA_IMPORT_ERROR}")
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
    return Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=str(DB_DIR))

# ── LLM 调用 ────────────────────────────
def ask_ollama(prompt: str, model: str) -> str:
    r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip()

def ask_groq(prompt: str, model: str) -> str:
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 GROQ_API_KEY。")
    r = requests.post(GROQ_API_URL, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def ask_llm(prompt: str, provider: str, model: str) -> str:
    return ask_groq(prompt, model) if provider == "Groq Cloud" else ask_ollama(prompt, model)

def get_secret(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

def make_prompt(question: str, docs: list[Document]) -> str:
    context = "\n\n".join(f"[来源 {i}: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for i, d in enumerate(docs, 1))
    return f"""你是一个建筑知识库助手。请只根据下面的资料回答问题。
如果资料中没有答案，请明确说"当前文献中没有足够信息"。
回答要简洁、专业，并在关键结论后标注来源编号。

资料：
{context}

问题：
{question}

回答："""

# ── 主界面 ──────────────────────────────
def main():
    DOCUMENTS_DIR.mkdir(exist_ok=True)

    # ── 顶部标题 ──
    st.markdown('<div class="product-title">ArchMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="product-subtitle">AI 建筑知识助手 · 上传规范与文献，用自然语言提问，基于 RAG 检索回答并标注来源</div>', unsafe_allow_html=True)

    # ── 状态栏 ──
    current_files = sorted(f.name for f in DOCUMENTS_DIR.iterdir() if f.suffix.lower() in SUPPORTED_SUFFIXES)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f'<div style="font-size:0.85rem;color:#6B6B68;">📄 已录入 <b style="color:#1A1A1A;">{len(current_files)}</b> 份文档</div>', unsafe_allow_html=True)
    with col_b:
        emb_name = type(get_embeddings()).__name__
        emb_display = "语义向量模型（HuggingFace）" if "HuggingFace" in emb_name else "本地哈希向量"
        st.markdown(f'<div style="font-size:0.85rem;color:#6B6B68;">🧠 向量方式：<b style="color:#1A1A1A;">{emb_display}</b></div>', unsafe_allow_html=True)
    with col_c:
        provider_label = "Groq Cloud" if get_secret("GROQ_API_KEY") else "Ollama 本地"
        st.markdown(f'<div style="font-size:0.85rem;color:#6B6B68;">⚡ 默认模型服务：<b style="color:#1A1A1A;">{provider_label}</b></div>', unsafe_allow_html=True)
    st.markdown("<hr style='border:0;border-top:1px solid #E8E4DD;margin:0.8rem 0 1.5rem 0'>", unsafe_allow_html=True)

    # ── 设置面板（折叠）──
    with st.expander("⚙️ 模型设置 · 上传文档 · 索引管理", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            provider = st.selectbox("模型服务", ["Groq Cloud", "Ollama 本地"], index=0 if get_secret("GROQ_API_KEY") else 1)
            model = st.text_input("模型名称", value=DEFAULT_GROQ_MODEL if provider == "Groq Cloud" else DEFAULT_MODEL)
            top_k = st.slider("检索片段数量", 2, 8, 4)
        with c2:
            uploaded_files = st.file_uploader("上传文档（TXT / MD / PDF）", type=["txt", "md", "pdf"], accept_multiple_files=True)
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                if st.button("保存并重建索引", use_container_width=True) and uploaded_files:
                    saved = save_uploaded_files(uploaded_files)
                    reset_vector_store()
                    st.success(f"已保存 {len(saved)} 个文件")
                    st.rerun()
            with col_u2:
                if st.button("清空索引", use_container_width=True):
                    reset_vector_store()
                    st.success("索引已清空")
                    st.rerun()
            if current_files:
                st.caption("当前文档：" + "、".join(current_files))
            rebuild = st.checkbox("启动时强制重建索引", value=False)

    # ── 提问区 ──
    question = st.text_area("输入你的建筑问题", placeholder="例如：被动式建筑设计中，应该如何降低建筑能耗？", height=80, label_visibility="collapsed")

    col_q1, col_q2, col_q3 = st.columns([1, 1, 4])
    ask_clicked = col_q1.button("🔍 提问", type="primary", use_container_width=True)

    # ── 执行检索 ──
    if ask_clicked and question.strip():
        try:
            vector_store = build_vector_store(force_rebuild=rebuild)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        with st.spinner("正在检索文献..."):
            retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
            related_docs = retriever.invoke(question)

        if not related_docs:
            st.warning("没有检索到相关文献片段。")
            st.stop()

        prompt = make_prompt(question, related_docs)

        try:
            with st.spinner("正在生成回答..."):
                answer = ask_llm(prompt, provider, model)
        except requests.exceptions.ConnectionError:
            st.error(f"无法连接 Ollama。请确认已启动并运行：ollama run {model}")
            st.stop()
        except requests.exceptions.HTTPError as exc:
            st.error(f"模型服务返回错误：{exc}")
            if provider == "Ollama 本地":
                st.info(f"如果模型不存在，请先运行：ollama pull {model}")
            else:
                st.info("请检查 GROQ_API_KEY 是否有效，以及模型名称是否在 Groq 支持列表中。")
            st.stop()
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"调用模型失败：{exc}")
            st.stop()

        # ── 回答 ──
        st.markdown("---")
        st.markdown('<div style="font-size:0.85rem;color:#6B6B68;margin-bottom:0.5rem;">📝 回答</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-block">{answer}</div>', unsafe_allow_html=True)

        # ── 来源 ──
        st.markdown('<div style="font-size:0.85rem;color:#6B6B68;margin:1.2rem 0 0.5rem 0;">📎 检索来源（共 {0} 条）</div>'.format(len(related_docs)), unsafe_allow_html=True)
        for i, doc in enumerate(related_docs, 1):
            source = doc.metadata.get("source", "unknown")
            with st.expander(f"来源 {i}：{source}"):
                st.markdown(f'<div style="font-size:0.85rem;line-height:1.7;color:#1A1A1A;">{doc.page_content}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
