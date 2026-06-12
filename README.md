# AI Study Assistant

一个适合作为 AI 学习者首个 GitHub 项目的 PDF 学习助手。上传课程资料后，
应用会自行完成文本切分、向量检索和上下文拼接，并让大模型基于原文回答问题。

## 功能

- 上传并解析 PDF，保留原始页码
- 使用 OpenAI Embeddings 建立内存向量索引
- 通过余弦相似度检索相关内容
- 基于检索结果回答问题，并标注来源页码
- 展示命中的原文片段和相似度
- 自动生成单选题、简答题和答案解析

## 项目亮点

这个项目没有使用 LangChain，而是直接实现了一个精简的 RAG 流程：

```text
PDF -> 分页提取 -> 重叠切块 -> 向量化 -> 相似度检索
    -> 拼接上下文 -> 大模型回答 -> 页码引用
```

这样更容易理解并向面试官解释每一个环节。

## 快速开始

需要 Python 3.10 或更高版本。

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

打开 `.env`，填写自己的 API Key：

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

也可以直接在应用侧边栏临时输入 API Key。

## 测试

```bash
pip install -r requirements-dev.txt
pytest
```

## 目录结构

```text
.
├── app.py
├── src/
│   ├── chatbot.py
│   ├── document.py
│   └── retrieval.py
├── tests/
├── .env.example
├── requirements.txt
└── requirements-dev.txt
```

## 已知限制

- 当前索引只保存在内存中，重启应用后需要重新处理 PDF。
- 扫描版 PDF 没有文本层，需要先进行 OCR。
- 生成答案和向量会产生 OpenAI API 使用费用。
- 第一版只支持单个 PDF。

## 后续计划

- 支持多个 PDF 和持久化向量库
- 增加 OCR 和图片理解
- 加入答案质量评测
- 支持导出复习题和错题本

## License

[MIT](LICENSE)
