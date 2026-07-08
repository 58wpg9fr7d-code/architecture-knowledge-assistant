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
except Exception as exc:  # pragma: no cover - shown in Streamlit UI
    Chroma = None
    CHROMA_IMPORT_ERROR = exc
else:
    CHROMA_IMPORT_ERROR = None


APP_DIR = Path(__file__).parent
DOCUMENTS_DIR = APP_DIR / "documents"
DB_DIR = APP_DIR / ".chroma_db"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


class HashEmbeddings(Embeddings):
    """Small local embedding fallback so the prototype can run before ML deps install."""

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

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        lowered = text.lower()
        words: list[str] = []
        current: list[str] = []

        for char in lowered:
            if char.isalnum() or "\u4e00" <= char <= "\u9fff":
                current.append(char)
            elif current:
                words.append("".join(current))
                current = []
        if current:
            words.append("".join(current))

        # Character bigrams make Chinese keyword retrieval less brittle.
        cjk_chars = [char for char in lowered if "\u4e00" <= char <= "\u9fff"]
        unigrams = cjk_chars
        bigrams = [a + b for a, b in zip(cjk_chars, cjk_chars[1:])]
        trigrams = [a + b + c for a, b, c in zip(cjk_chars, cjk_chars[1:], cjk_chars[2:])]
        return words + unigrams + bigrams + trigrams


@st.cache_resource(show_spinner=False)
def get_embeddings() -> Embeddings:
    use_hf_embeddings = str(get_secret("USE_HF_EMBEDDINGS", "")).lower() in {"1", "true", "yes"}
    if not use_hf_embeddings:
        return HashEmbeddings()

    try:
        import sentence_transformers  # noqa: F401
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        return HashEmbeddings()


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("读取 PDF 需要先安装 pypdf：pip install pypdf") from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def load_documents() -> list[Document]:
    documents: list[Document] = []
    supported_paths = sorted(path for path in DOCUMENTS_DIR.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES)

    for path in supported_paths:
        if path.suffix.lower() == ".pdf":
            text = read_pdf(path)
        else:
            text = read_txt(path)

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "path": str(path)},
                )
            )

    return documents


def safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    stem = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", Path(name).stem).strip("._")
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise RuntimeError("只支持上传 TXT、MD 或 PDF 文件。")
    return f"{stem or 'document'}{suffix}"


def save_uploaded_files(uploaded_files) -> list[str]:
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    saved_files: list[str] = []
    for uploaded_file in uploaded_files:
        filename = safe_filename(uploaded_file.name)
        destination = DOCUMENTS_DIR / filename
        destination.write_bytes(uploaded_file.getbuffer())
        saved_files.append(filename)
    return saved_files


def reset_vector_store() -> None:
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    st.cache_resource.clear()


def split_documents(documents: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "；", ";", ".", " ", ""],
    )
    return splitter.split_documents(list(documents))


def build_vector_store(force_rebuild: bool = False):
    if Chroma is None:
        raise RuntimeError(f"ChromaDB 加载失败：{CHROMA_IMPORT_ERROR}")

    if force_rebuild and DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    embeddings = get_embeddings()
    if DB_DIR.exists() and any(DB_DIR.iterdir()):
        return Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)

    source_documents = load_documents()
    if not source_documents:
        raise RuntimeError("documents 文件夹里还没有可读取的 .txt、.md 或 .pdf 文件。")

    chunks = split_documents(source_documents)
    if not chunks:
        raise RuntimeError("文档已读取，但没有切分出可检索的内容。")

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )


def ask_ollama(prompt: str, model: str) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def ask_groq(prompt: str, model: str) -> str:
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 GROQ_API_KEY 环境变量。")

    response = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def ask_llm(prompt: str, provider: str, model: str) -> str:
    if provider == "Groq Cloud":
        return ask_groq(prompt, model)
    return ask_ollama(prompt, model)


def get_secret(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def make_prompt(question: str, docs: list[Document]) -> str:
    context = "\n\n".join(
        f"[来源 {index}: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for index, doc in enumerate(docs, start=1)
    )
    return f"""你是一个建筑知识库助手。请只根据下面的资料回答问题。
如果资料中没有答案，请明确说“当前文献中没有足够信息”。
回答要简洁、专业，并在关键结论后标注来源编号。

资料：
{context}

问题：
{question}

回答："""


def show_source(doc: Document, index: int) -> None:
    source = doc.metadata.get("source", "unknown")
    with st.expander(f"来源 {index}: {source}"):
        st.write(doc.page_content)


def main() -> None:
    st.set_page_config(page_title="建筑知识库助手", page_icon="🏛️", layout="wide")

    st.title("建筑知识库助手")
    st.caption("基于本地文档检索、ChromaDB 向量库和可切换模型服务的 RAG 原型")

    DOCUMENTS_DIR.mkdir(exist_ok=True)

    with st.sidebar:
        st.header("设置")
        default_provider = "Groq Cloud" if get_secret("GROQ_API_KEY") else "Ollama 本地"
        provider = st.selectbox("模型服务", ["Ollama 本地", "Groq Cloud"], index=0 if default_provider == "Ollama 本地" else 1)
        default_model = DEFAULT_GROQ_MODEL if provider == "Groq Cloud" else DEFAULT_MODEL
        model = st.text_input("模型名称", value=default_model)
        top_k = st.slider("检索片段数量", min_value=2, max_value=8, value=4)

        st.divider()
        st.write("上传文档")
        uploaded_files = st.file_uploader(
            "支持 TXT、MD、PDF",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
        )
        if st.button("保存上传文件", use_container_width=True):
            if uploaded_files:
                saved_files = save_uploaded_files(uploaded_files)
                reset_vector_store()
                st.success(f"已保存 {len(saved_files)} 个文件，并清空旧索引。")
            else:
                st.warning("请先选择要上传的文件。")

        st.divider()
        st.write("当前文档")
        st.code(str(DOCUMENTS_DIR), language="text")
        current_files = sorted(path.name for path in DOCUMENTS_DIR.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES)
        if current_files:
            for filename in current_files:
                st.caption(f"- {filename}")
        else:
            st.caption("还没有可检索文档。")

        rebuild = st.button("重新建立索引", use_container_width=True)
        if st.button("清空本地索引", use_container_width=True):
            reset_vector_store()
            st.success("索引已清空。")

        st.divider()
        embeddings_name = type(get_embeddings()).__name__
        st.caption(f"当前向量方式：{embeddings_name}")

    try:
        vector_store = build_vector_store(force_rebuild=rebuild)
    except Exception as exc:
        st.error(str(exc))
        st.info("请先把建筑相关 TXT、MD 或 PDF 文件放进 documents 文件夹，然后点击“重新建立索引”。")
        st.stop()

    question = st.text_area(
        "输入你的建筑问题",
        value="被动式建筑设计中，应该如何降低建筑能耗？",
        height=90,
    )

    if st.button("提问", type="primary"):
        if not question.strip():
            st.warning("请先输入一个问题。")
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
            st.error(
                "无法连接 Ollama。请确认已经打开 Ollama，并在终端运行过：ollama run "
                f"{model}"
            )
            st.stop()
        except requests.exceptions.HTTPError as exc:
            st.error(f"模型服务返回错误：{exc}")
            if provider == "Ollama 本地":
                st.info(f"如果模型不存在，请先运行：ollama pull {model}")
            else:
                st.info("请检查 GROQ_API_KEY 是否正确，以及模型名称是否在 Groq 支持列表中。")
            st.stop()
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"调用模型失败：{exc}")
            st.stop()

        st.subheader("回答")
        st.write(answer)

        st.subheader("检索来源")
        for index, doc in enumerate(related_docs, start=1):
            show_source(doc, index)


if __name__ == "__main__":
    main()
