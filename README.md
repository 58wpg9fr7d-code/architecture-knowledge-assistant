# ArchMind · AI 建筑知识助手

ArchMind 是一个面向建筑规范、文献和案例资料的 AI 知识检索产品。用户上传自己的建筑资料后，可以用自然语言提问，系统通过 RAG 检索相关片段，再由大模型生成回答并展示来源。

本项目包含两条产品线：

- **v1.0 技术验证**：Streamlit + ChromaDB 的建筑资料问答助手，支持本地 Ollama 和线上 Groq Cloud。
- **v2.0 产品设计**：从“单次问答”扩展到资料上传、分析维度选择、结构化报告、提纲生成和导出的建筑案例分析工作流。

## 先看什么

| 目的 | 文件 |
|---|---|
| 了解完整产品叙事 | [ArchMind_产品一页纸.docx](./ArchMind_产品一页纸.docx) |
| 了解 RAG 评测与优化 | [ArchMind_产品验证与优化计划.md](./ArchMind_产品验证与优化计划.md) |
| 准备简历和面试 | [ArchMind_面试讲解稿.md](./ArchMind_面试讲解稿.md) |
| 查看 v2.0 用户流程 | [ArchMind_用户流程图.html](./ArchMind_用户流程图.html) |
| 查看 v2.0 六页原型 | [ArchMind_Axure_6页产品原型.html](./ArchMind_Axure_6页产品原型.html) |
| 查看竞品、作品集和部署材料 | [outputs/](./outputs/) |

## 当前能力

### v1.0 RAG 应用

```text
上传 TXT / MD / PDF
        ↓
读取并切分文档
        ↓
Embedding 向量化
        ↓
ChromaDB 建立索引
        ↓
自然语言提问
        ↓
检索相关片段
        ↓
Ollama / Groq 生成回答
        ↓
展示回答与来源片段
```

代码职责：

- `app.py`：Streamlit 页面、上传、索引管理、提问和来源展示。
- `core.py`：文档读取、切分、Embedding、向量库、Prompt、模型调用和统一查询接口。
- `api.py`：FastAPI `/health` 和 `/ask` 接口，供 Chrome 扩展调用。
- `extension/`：Manifest V3 Chrome 扩展，支持选中文字后右键提交给 ArchMind。

### v2.0 案例分析工作流

```text
首页 → 资料上传 → 用途与维度选择 → AI 分析结果
     → 汇报提纲生成 → 导出 / 保存案例
```

相关材料：

- `ArchMind_用户流程图.html`：六步用户流程。
- `ArchMind_Axure_6页产品原型.html`：六页低保真交互原型。
- `截图/`：原型页面截图。
- `outputs/competitor_analysis.md`：竞品分析。

## 本地运行

### 1. 安装依赖

建议使用 Python 3.11 虚拟环境：

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 准备模型

本地运行：

```bash
ollama pull qwen2:7b
```

线上运行：

```bash
export GROQ_API_KEY="你的 Groq API Key"
export GROQ_MODEL="llama-3.1-8b-instant"
```

也可以复制 `.streamlit/secrets.toml.example` 为 `.streamlit/secrets.toml` 后填写密钥。密钥文件不要提交到 Git。

### 3. 启动 Streamlit

```bash
streamlit run app.py
```

打开页面后，在设置面板上传资料，点击“保存并重建索引”，再输入问题。

### 4. 启动 FastAPI 和扩展

```bash
uvicorn api:app --host 0.0.0.0 --port 8765
```

然后在 Chrome 的扩展程序页面打开开发者模式，加载 `extension/` 文件夹。

扩展默认请求 `http://localhost:8765`。生产环境可设置：

```bash
export ALLOWED_ORIGINS="chrome-extension://你的扩展 ID"
```

## 数据目录

当前 `documents/` 目录共有 17 份资料：13 份 TXT 核心要点和 4 份 PDF 文献。它们是本地测试资料，不等于已经完成正式用户评测。

`.chroma_db/` 是运行时生成的向量索引，已加入 `.gitignore`，更换或更新资料后需要重新建立索引。

## 产品决策

### 为什么不直接导入规范全集？

规范全集可能带来相似术语、重复条款、目录附注、PDF 排版和上下文丢失等问题。MVP 阶段先使用少量、主题明确的资料，验证切分、检索噪声、上下文保留和来源可追溯性，再扩充数据。

### 为什么使用 Streamlit？

项目第一阶段的目标是验证 RAG 任务链路，而不是投入大量时间开发复杂前端。Streamlit 可以用纯 Python 快速搭建上传、提问和结果展示界面。

### 为什么使用 ChromaDB？

ChromaDB 开源、本地运行、成本低，适合小规模原型。未来如果数据规模、并发量和权限要求提高，再考虑迁移到更适合生产环境的向量数据库。

## 评测与当前边界

当前项目已经完成：

- 多格式文档读取和上传；
- 文档切分与向量索引；
- 自然语言检索；
- 基于检索片段的模型回答；
- 来源片段展示；
- Ollama / Groq Cloud 双模型服务；
- v2.0 PRD、用户流程和原型。

当前没有虚构或宣称以下结果：

- 用户数量；
- 访谈人数；
- 检索准确率；
- 回答准确率；
- 节省时间比例；
- 用户留存或持续使用情况。

下一轮验证方法见 [ArchMind_产品验证与优化计划.md](./ArchMind_产品验证与优化计划.md)。

## 生产化路线

优先级从高到低：


1. 建立 20 道固定评测题，记录标准证据、检索命中和引用准确性。
2. 将来源从文件级片段升级到页码、章节和条款级引用。
3. 增加文档版本、更新时间、索引重建和删除管理。
4. 增加用户权限、API 限流、调用日志和错误监控。
5. 对检索结果增加重排和人工反馈闭环。

## 作品集表达

> 我独立完成了 ArchMind，一款面向建筑规范与文献检索场景的 AI 知识助手。项目从 v1.0 RAG 技术验证开始，完成多格式资料上传、文档切分、向量检索、模型回答和来源展示；随后基于建筑案例研究场景扩展为 v2.0 工作流，完成用户场景拆解、竞品分析、MVP 定义、PRD、用户流程图和低保真原型。项目当前定位为 MVP，下一步重点是建立评测集、完善条款级引用并进行真实用户验证。
