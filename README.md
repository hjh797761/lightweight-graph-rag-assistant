# Lightweight Graph RAG Assistant

一个面向中文 PDF、DOCX、Markdown 和纯文本资料的 CPU-first Graph RAG 原型。项目把文档切块、批量生成向量、构建概念共现图和主题索引，并通过统一检索管线生成带 `[S1]`、`[S2]` 来源编号的上下文。

本项目适合本地学习、原型验证和检索消融实验。它不是已经完成生产加固的托管服务。

## 主要变化

- 使用 SQLite 存储文档、处理进度、chunk、`float32` embedding、概念边和 topic。
- 一个批次的 chunk、概念和进度在同一事务中提交，失败时整批回滚。
- embedding 按批编码；检索使用缓存的连续 NumPy 矩阵，而不是逐条 Python 扫描。
- 五种检索 profile 共用同一数据、文档范围、候选预算和输出格式。
- 生产提示和评测器不包含针对私有题目的答案特例。
- 测试与公开评测默认可使用确定性 CPU 后端，不下载模型、不调用 API。

## 安装

需要 Python 3.10 或更高版本。CPU 环境可直接运行：

```powershell
git clone https://github.com/hjh797761/lightweight-graph-rag-assistant.git
cd lightweight-graph-rag-assistant
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

首次实际建库或检索时，默认会从 Hugging Face 下载以下模型并缓存：

- `BAAI/bge-small-zh-v1.5`
- `BAAI/bge-reranker-base`（当 `ENABLE_CROSS_ENCODER=1`）

如果不需要 cross-encoder，可在 `.env` 设置 `ENABLE_CROSS_ENCODER=0`，减少首次下载和 CPU 推理开销。

### 离线模式

联网机器可提前下载模型：

```powershell
python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-zh-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"
```

随后在 `.env` 设置：

```dotenv
OFFLINE_MODE=1
```

离线模式只读取本机缓存；缓存缺失时会抛出明确的 `ModelUnavailableError`，不会静默联网。

## 配置

常用环境变量见 [.env.example](.env.example)：

```dotenv
KB_PATH=knowledge_base.json
OFFLINE_MODE=0
BUILD_BATCH_SIZE=20
EMBEDDING_BACKEND=sentence_transformer
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
ENABLE_CROSS_ENCODER=1
CROSS_ENCODER_MODEL=BAAI/bge-reranker-base
VECTOR_RECALL_K=40
FINAL_TOP_K=6
MOONSHOT_API_KEY=
MOONSHOT_BASE_URL=https://api.moonshot.cn/v1
MOONSHOT_MODEL=moonshot-v1-8k
```

`.env` 会先加载，进程环境变量具有更高优先级。`EMBEDDING_BACKEND=moonshot` 仍受支持，并通过 OpenAI 兼容 embeddings API 批量请求向量。

## 运行

启动兼容的交互菜单：

```powershell
python graphrag_assistant.py
```

菜单保留上传资料、提问、查看知识结构、查看文档列表、备份并清空和退出操作。回答生成需要有效的 Moonshot/Kimi 配置；建库和检索本身可使用本地模型。

旧 Python 入口仍可导入：

```python
from graphrag_assistant import (
    add_document_to_tree,
    generate_answer,
    load_knowledge_base,
    retrieve,
    retrieve_semantic,
)
```

`dingtalk_server.py` 继续通过这些入口工作，启动方式为：

```powershell
python dingtalk_server.py
```

## 存储与旧数据迁移

当 `KB_PATH=knowledge_base.json` 时，实际新库路径为同目录下的 `knowledge_base.sqlite3`。如果 SQLite 不存在而旧 JSON 存在，首次加载会：

1. 在同目录构建临时 SQLite；
2. 迁移并核对 chunk 数量；
3. 成功后原子切换为正式 SQLite；
4. 始终保留原 `knowledge_base.json`。

迁移失败不会覆盖原 JSON 或已有正式数据库。菜单中的“清空”仍要求输入 `RESET`，并在清空前生成 SQLite 备份。

## 公开评测

仓库包含两份原创合成资料和问题集。无需模型下载的可复现烟测：

```powershell
python scripts/retrieval_eval.py --embedding-backend deterministic --out-json .tmp/eval.json --out-md .tmp/eval.md
```

使用真实 embedding 模型：

```powershell
python scripts/retrieval_eval.py --embedding-backend model --out-json .tmp/eval-model.json --out-md .tmp/eval-model.md
```

报告同时运行 `vector`、`vector_keyword`、`vector_graph`、`vector_topic_graph` 和 `full`。各 profile 使用同一个问题限定文档和 `top_k`，输出关键证据覆盖、倒数排名、命中片段、分数和检索耗时。指标只用于报告，不设自动质量阈值。

非门禁 CPU 基准：

```powershell
python scripts/cpu_benchmark.py
```

## 测试

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -v
python -m compileall -q graphrag graphrag_assistant.py dingtalk_server.py scripts tests
```

GitHub Actions 在 Windows 和 Ubuntu、Python 3.10 与 3.12 上运行测试、语法检查和确定性公开评测。

更多细节见 [系统设计](docs/system_design.md) 和 [评测说明](docs/evaluation.md)。

## License

[MIT](LICENSE)
