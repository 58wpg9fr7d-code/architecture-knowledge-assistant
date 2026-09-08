# ArchMind · AI 建筑知识助手

把建筑规范、文献和案例资料整理成**可提问、可核验**的知识库，并扩展为**案例分析工作流**（RAG检索 + 按用途模板产出报告）。

## 快速入口

- 🚀 **在线体验**（Streamlit）：`https://architecture-knowledge-assistan-sevbyy7z3nkrurt6twqma9.streamlit.app`
- 🎯 **面试 Demo**（GitHub Pages，免登录）：<https://58wpg9fr7d-code.github.io/architecture-knowledge-assistant/>
- 📄 **产品一页纸**：`ArchMind_产品一页纸_合并版.docx`
- 🎤 **面试材料**：`ArchMind_面试材料_合并版.docx`
- 🎨 **设计原型说明**：`ArchMind_设计原型_整合版.docx`
- 📋 **项目总览**：`00-项目总览.md`

## 项目状态

- **V1（已实现部署）**：RAG检索问答，Streamlit + ChromaDB + Groq/Gemini/Ollama，已预置17份资料
- **V2（PRD+原型完成）**：案例分析工作流——上传 → 选用途/维度 → 结构化报告 → 汇报提纲 → 导出

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

模型配置见 `.env.example` 和 `.streamlit/secrets.toml.example`。

## 目录说明

| 路径 | 说明 |
|---|---|
| `app.py` / `core.py` / `api.py` | Streamlit前端 / RAG核心 / FastAPI接口 |
| `documents/` | 19份建筑资料（规范要点TXT + 文献PDF） |
| `extension/` | Chrome扩展（选中文字右键调用ArchMind） |
| `outputs/` | 竞品分析、部署清单、作品集材料 |
| `截图/` | 6页原型 + 用户流程图截图 |
| `index.html` | GitHub Pages Demo入口 |
| `.streamlit/` | Streamlit配置与密钥示例 |

## 当前边界

公开Demo展示产品形态；真实RAG能力在受控环境验证。不宣称用户数、准确率、节省时间等未验证数据。
