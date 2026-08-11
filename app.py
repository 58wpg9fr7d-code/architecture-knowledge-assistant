"""
ArchMind · AI 建筑知识助手  (Streamlit frontend)

RAG-powered Q&A over architectural regulations and literature.
Uses core.py for all backend logic.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

from core import (
    DOCUMENTS_DIR,
    DEFAULT_GROQ_MODEL,
    DEFAULT_OLLAMA_MODEL,
    SUPPORTED_SUFFIXES,
    get_embeddings,
    get_secret,
    list_current_files,
    load_documents,
    save_uploaded_file,
    split_documents,
    build_vector_store,
    reset_vector_store,
    ask_llm,
    make_prompt,
)

# ── Page config ───────────────────────────
st.set_page_config(
    page_title="ArchMind · AI 建筑知识助手",
    page_icon="🏛️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────
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


# ── Main UI ───────────────────────────────
def main():
    DOCUMENTS_DIR.mkdir(exist_ok=True)

    # ── Header ──
    st.markdown('<div class="product-title">ArchMind</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="product-subtitle">'
        'AI 建筑知识助手 · 上传规范与文献，用自然语言提问，基于 RAG 检索回答并标注来源'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Status bar ──
    current_files = list_current_files()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            f'<div style="font-size:0.85rem;color:#6B6B68;">'
            f'📄 已录入 <b style="color:#1A1A1A;">{len(current_files)}</b> 份文档</div>',
            unsafe_allow_html=True,
        )
    with col_b:
        emb_name = type(get_embeddings()).__name__
        emb_display = "语义向量模型（HuggingFace）" if "HuggingFace" in emb_name else "本地哈希向量"
        st.markdown(
            f'<div style="font-size:0.85rem;color:#6B6B68;">'
            f'🧠 向量方式：<b style="color:#1A1A1A;">{emb_display}</b></div>',
            unsafe_allow_html=True,
        )
    with col_c:
        provider_label = "Groq Cloud" if get_secret("GROQ_API_KEY") else "Ollama 本地"
        st.markdown(
            f'<div style="font-size:0.85rem;color:#6B6B68;">'
            f'⚡ 默认模型服务：<b style="color:#1A1A1A;">{provider_label}</b></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        "<hr style='border:0;border-top:1px solid #E8E4DD;margin:0.8rem 0 1.5rem 0'>",
        unsafe_allow_html=True,
    )

    # ── Settings panel ──
    with st.expander("⚙️ 模型设置 · 上传文档 · 索引管理", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            provider = st.selectbox(
                "模型服务",
                ["Groq Cloud", "Ollama 本地"],
                index=0 if get_secret("GROQ_API_KEY") else 1,
            )
            model = st.text_input(
                "模型名称",
                value=DEFAULT_GROQ_MODEL if provider == "Groq Cloud" else DEFAULT_OLLAMA_MODEL,
            )
            top_k = st.slider("检索片段数量", 2, 8, 4)
        with c2:
            uploaded_files = st.file_uploader(
                "上传文档（TXT / MD / PDF）",
                type=["txt", "md", "pdf"],
                accept_multiple_files=True,
            )
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                if st.button("保存并重建索引", use_container_width=True) and uploaded_files:
                    for f in uploaded_files:
                        save_uploaded_file(f.getbuffer(), f.name)
                    reset_vector_store()
                    st.success(f"已保存 {len(uploaded_files)} 个文件")
                    st.rerun()
            with col_u2:
                if st.button("清空索引", use_container_width=True):
                    reset_vector_store()
                    st.success("索引已清空")
                    st.rerun()
            if current_files:
                st.caption("当前文档：" + "、".join(current_files))
            rebuild = st.checkbox("启动时强制重建索引", value=False)

    # ── Question input ──
    question = st.text_area(
        "输入你的建筑问题",
        placeholder="例如：被动式建筑设计中，应该如何降低建筑能耗？",
        height=80,
        label_visibility="collapsed",
    )

    col_q1, col_q2, col_q3 = st.columns([1, 1, 4])
    ask_clicked = col_q1.button("🔍 提问", type="primary", use_container_width=True)

    # ── Execute query ──
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
            st.error(f"无法连接 {provider}。请确认服务已启动。")
            if provider == "Ollama 本地":
                st.info(f"请先运行：ollama run {model}")
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

        # ── Answer ──
        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.85rem;color:#6B6B68;margin-bottom:0.5rem;">📝 回答</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="answer-block">{answer}</div>', unsafe_allow_html=True)

        # ── Sources ──
        st.markdown(
            f'<div style="font-size:0.85rem;color:#6B6B68;margin:1.2rem 0 0.5rem 0;">'
            f'📎 检索来源（共 {len(related_docs)} 条）</div>',
            unsafe_allow_html=True,
        )
        for i, doc in enumerate(related_docs, 1):
            source = doc.metadata.get("source", "unknown")
            with st.expander(f"来源 {i}：{source}"):
                st.markdown(
                    f'<div style="font-size:0.85rem;line-height:1.7;color:#1A1A1A;">'
                    f'{doc.page_content}</div>',
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
