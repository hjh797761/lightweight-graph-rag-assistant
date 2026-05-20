# Graph RAG 复习助手

一个面向长文档复习问答的轻量 Graph RAG 原型。系统目标不是替代大模型，而是在用户上传 PDF / Word / 文本资料后，尽量从原文中找全证据，再把可追溯的上下文交给 LLM 生成回答。

## 项目定位

这个项目最初的定位是“复习助手”：帮助学生把课件、教材、论文、白皮书等资料变成可提问的知识库。后续测试发现，它在财报、行业报告这类长文档上也有不错表现，尤其适合需要同时命中表格数字、主体、变动原因和跨页证据的问题。

相比更重的 GraphRAG 路线，本项目刻意选择轻量化实现：

- 不依赖图数据库。
- 不生成社区摘要。
- 不强依赖文档目录或章节树。
- 以 `chunk embedding + topic_index + concept graph + evidence guard` 为主流程。

在同一份财报文档上，nano-GraphRAG 完整建图曾耗时约 40 多分钟；本项目没有社区检测和社区报告生成，建库流程更快，更适合课程项目、个人知识库和本地复习助手场景。

## 核心思路

系统采用 v2 流程：

```text
文档上传
→ chunk 切分
→ chunk embedding
→ 语义 topic 聚类
→ 概念抽取与 concept graph
→ 多路召回
→ rerank 融合排序
→ dynamic top-k
→ LLM 基于证据回答
```

与普通 RAG 相比，本项目额外做了几件事：

- `topic_index`：把语义相近的 chunk 归入主题簇，查询时缩小语义范围。
- `knowledge_graph`：记录概念、chunk、概念共现关系，用于图扩展召回。
- `exact evidence guard`：对数字、金额、同比、主体、原因说明等强证据做保底。
- `adjacent chunk bridge`：当表格和原因说明分散在相邻 chunk 时，自动补桥。
- `dynamic top-k`：根据问题复杂度动态决定给 LLM 的证据数量。

## 目录

```text
.
├── graphrag_assistant.py          # 核心命令行 RAG 程序
├── dingtalk_server.py             # 钉钉 / OpenClaw 接入示例
├── requirements.txt               # 依赖
├── .env.example                   # 环境变量示例
├── scripts/
│   └── two_doc_retrieval_eval.py  # 检索证据命中评测脚本
└── docs/
    ├── V2_FOCUSED_EVAL_SUMMARY.md
    ├── DEEPEVAL_TWO_DOC_FOCUSED_SUMMARY.md
    └── EXTERNAL_BASELINE_NANO_SUMMARY.md
```

## 安装

建议使用 Python 3.10+。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

首次运行会下载 embedding / reranker 模型，默认使用：

- `BAAI/bge-small-zh-v1.5`
- `BAAI/bge-reranker-base`

## 配置

复制环境变量模板：

```bash
copy .env.example .env
```

至少需要设置：

```bash
set MOONSHOT_API_KEY=你的_key
```

如果使用钉钉 / OpenClaw 接入，还需要设置：

```bash
set DINGTALK_WEBHOOK=你的钉钉机器人 webhook
```

不要把 `.env`、API key、webhook token 提交到 GitHub。

## 命令行使用

```bash
python graphrag_assistant.py
```

菜单中：

- `1` 上传资料并建库
- `2` 提问
- `3` 查看知识结构
- `4` 查看文档列表
- `9` 备份并清空知识库

默认知识库文件是 `knowledge_base.json`。可用环境变量指定：

```bash
set KB_PATH=./data/my_knowledge_base.json
```

## 钉钉 / OpenClaw 接入

```bash
python dingtalk_server.py
```

服务默认监听 `0.0.0.0:5000`，可配合 ngrok / OpenClaw 把钉钉消息转发到 `/dingtalk`。

## 评测说明

本项目的评测不只看大模型回答流畅度，而是分三层：

1. 检索证据命中：上下文是否包含人工标注的关键证据点。
2. DeepEval 辅助评估：Answer Relevancy 和 Faithfulness。
3. 人工核查：主体、数字、原因、跨页整合、幻觉、可追溯性。

已整理的实验摘要见 `docs/`。

更详细的流程说明见：

- [系统设计说明](docs/system_design.md)
- [评测说明](docs/evaluation.md)

仓库提供了一个极小样例 [examples/sample.txt](examples/sample.txt)，用于快速验证上传、建库和问答流程。

## 开源注意

仓库不应包含：

- API key、webhook、access token
- 个人文档、课程资料、商业 PDF 原文
- `knowledge_base*.json` 等本地知识库
- 大量评测中间产物和缓存
- `.deepeval/`、`__pycache__/`、模型缓存

## License

建议按课程项目用途选择 MIT License；如果文档数据包含第三方版权材料，请不要把原文资料一起开源。
