# 建筑知识库助手

这是一个 RAG 原型：Streamlit 页面 + ChromaDB 向量库 + 本地文档检索。模型服务支持 Ollama 本地模型，也支持部署时切换到 Groq Cloud。

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

如果暂时装不上 `pypdf`，也可以先用 `documents/sample_architecture.txt` 跑通 TXT 检索流程。

## 2. 准备文档

可以直接在页面左侧上传 `.txt`、`.md` 或 `.pdf` 文件。也可以手动把文件放进：

```text
documents/
```

PDF 读取需要安装 `pypdf`。

## 3. 准备 Ollama 模型

确认 Ollama 已打开，然后至少准备一个对话模型，例如：

```bash
ollama pull qwen2:7b
```

如果你使用其他模型，可以在页面左侧修改模型名，或设置环境变量：

```bash
export OLLAMA_MODEL=qwen2:7b
```

## 4. 启动应用

```bash
streamlit run app.py
```

打开页面后，先点击左侧“重新建立索引”，再输入问题并点击“提问”。

## 5. 部署时使用 Groq

如果部署到 Railway 或 Streamlit Cloud，通常不能依赖本机 Ollama。可以在平台环境变量里设置：

```text
GROQ_API_KEY=你的 Groq API Key
GROQ_MODEL=llama-3.1-8b-instant
```

应用检测到 `GROQ_API_KEY` 后，左侧会默认选择 Groq Cloud。模型名称也可以在页面左侧手动改。

Streamlit Cloud 也可以在 App settings 的 Secrets 里填：

```toml
GROQ_API_KEY = "你的 Groq API Key"
GROQ_MODEL = "llama-3.1-8b-instant"
```

如果想启用更强的 HuggingFace 多语言向量模型，可以额外安装 `sentence-transformers`，并设置：

```toml
USE_HF_EMBEDDINGS = true
```

默认不开启它，是为了让线上部署更快、更稳。

## 6. 推到 GitHub 后部署

```bash
git init
git add .
git commit -m "Build architecture knowledge assistant"
git branch -M main
git remote add origin 你的 GitHub 仓库地址
git push -u origin main
```

然后在 Streamlit Cloud 新建 App，选择这个仓库，入口文件填：

```text
app.py
```

在 Secrets 里填入 `GROQ_API_KEY`，部署完成后即可获得公开链接。

## 常见问题

- 无法连接 Ollama：确认 Ollama 正在运行，并且模型名和页面左侧输入一致。
- Groq 调用失败：检查 `GROQ_API_KEY` 是否设置，以及模型名是否仍在 Groq 控制台支持列表中。
- PDF 读取失败：运行 `pip install pypdf`。
- 检索效果一般：默认使用轻量本地哈希向量，适合快速 demo；需要更强语义检索时再开启 `USE_HF_EMBEDDINGS`。
- 换文档后答案没变化：点击左侧“重新建立索引”。
