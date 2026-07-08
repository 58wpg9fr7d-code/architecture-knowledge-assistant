# 部署清单

## 当前状态

- 本地原型已跑通。
- 页面已支持上传 TXT、MD、PDF。
- 应用已支持 Ollama 本地模型和 Groq Cloud。
- 已准备 Streamlit Cloud 配置文件和 secrets 示例。

## 下一步操作

1. 在 GitHub 新建一个空仓库。
2. 在本地项目目录执行：

```bash
git init
git add .
git commit -m "Build architecture knowledge assistant"
git branch -M main
git remote add origin 你的 GitHub 仓库地址
git push -u origin main
```

3. 打开 Streamlit Cloud，新建 App。
4. 选择刚才的 GitHub 仓库。
5. Main file path 填：

```text
app.py
```

6. 在 Secrets 里填：

```toml
GROQ_API_KEY = "你的 Groq API Key"
GROQ_MODEL = "llama-3.1-8b-instant"
```

7. 部署完成后，打开公开链接测试：

- 上传一份建筑资料
- 点击重新建立索引
- 提问
- 检查回答和来源片段

## Railway 备用部署

如果 Streamlit Cloud 因网络或 IP 限制无法访问，可以使用 Railway：

1. 打开 Railway，新建 Project。
2. 选择 Deploy from GitHub repo。
3. 选择 `58wpg9fr7d-code/architecture-knowledge-assistant`。
4. 如果提示安装 Railway GitHub App，按页面提示授权这个仓库。
5. 在项目 Variables 里添加：

```text
GROQ_API_KEY=你的 Groq API Key
GROQ_MODEL=llama-3.1-8b-instant
```

6. Railway 会读取 `Procfile` 并启动 Streamlit。
7. 部署完成后，在 Settings 或 Deployments 中生成公开域名。

## 作品集里可以写的交付状态

已完成一个可运行的建筑知识库助手原型，支持本地文档上传、文档向量化检索、基于来源的回答生成，以及本地/云端模型切换。项目可本地运行，也可部署为公开演示链接。
