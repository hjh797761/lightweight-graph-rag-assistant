# 轻量证据关联助手 / Lightweight Evidence Association Assistant

面向中文 PDF、DOCX、Markdown 和 TXT 资料的 CPU-first 本地证据检索助手。默认 evidence 路径从向量与 BM25 召回，经 RRF 融合、可选且有候选上限的 cross-encoder 重排选出核心证据，再补充一跳显式引用目标或同节直接相邻片段，最后按渲染后的上下文预算输出来源。

检索分数不是概率；补充语境的分数为 `None`，不会借用核心分数。结构关联说明材料为何一起出现，不证明事实正确，也不保证回答正确。当前没有新路径优于其他系统的质量结论。项目适合本地使用与原型验证，尚非生产加固的托管服务。

旧仓库 slug `lightweight-graph-rag-assistant`、`graphrag` 包名和 Python 入口保留兼容，不需要重命名 GitHub 仓库。

## 安装与离线配置

需要 Python 3.10 或更高版本，可在 CPU 上运行。新项目请放在自己的项目盘目录，例如 Windows 的 D 盘：

~~~powershell
Set-Location D:\
git clone https://github.com/hjh797761/lightweight-graph-rag-assistant.git
Set-Location D:\lightweight-graph-rag-assistant
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~

安装依赖需要联网。仅在尚无 .env 时复制示例，已有配置请逐项检查，不覆盖：

~~~powershell
if (-not (Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.example -Destination .env }
~~~

新配置示例默认离线（.env 先加载，进程环境变量优先）：

~~~dotenv
KB_PATH=evidence.sqlite3
RETRIEVAL_PROFILE=evidence
OFFLINE_MODE=1
EMBEDDING_BACKEND=sentence_transformer
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
ENABLE_CROSS_ENCODER=0
CROSS_ENCODER_MODEL=BAAI/bge-reranker-base
EVIDENCE_MAX_CHUNKS=6
EVIDENCE_MAX_SUPPLEMENTS=2
EVIDENCE_CONTEXT_BUDGET=6000
EVIDENCE_SUPPLEMENT_FRACTION=0.30
EVIDENCE_BUDGET_UNIT=characters
~~~

[完整配置](.env.example) 中 ENABLE_CROSS_ENCODER=1 会启用本地 BGE reranker；设为 0 可减少 CPU 开销。默认本地 embedding 为 BAAI/bge-small-zh-v1.5，重排模型为 BAAI/bge-reranker-base。缓存不存在或加载失败会明确报错，不静默联网、不自动换模型或检索方式。已有服务配置应显式设置 OFFLINE_MODE=1；新 CLI 默认只读缓存。模型参数也接受本机绝对 snapshot 路径。

## 显式建库与运行

从原始文件创建新索引，输出路径必须不存在：

~~~powershell
python -B scripts/build_evidence_index.py --input D:\资料\手册.md D:\资料\规则.pdf --out D:\索引\evidence.sqlite3
~~~

命令默认只读取本地模型缓存。需要下载时明确加 --allow-download。无模型烟测可用 --embedding-backend deterministic，该后端不代表模型质量。所有文档先完整解析和准备，再分批向量化；只有完成的文档参与检索。

中断后只对未完成的新证据库使用 --resume，沿用原始输入、backend、model；必须包含所有未完成文档，可附原有已完成文档。输入变化、模型身份变化或已完成索引都会拒绝续建。失败留下的文件用于检查，不会覆盖旧库。评估运行目录则不提供 resume，重跑请选择新的目录。

建库成功后将命令打印的 KB_PATH、RETRIEVAL_PROFILE、EMBEDDING_BACKEND、EMBEDDING_MODEL、ENABLE_CROSS_ENCODER、OFFLINE_MODE 逐项放入应用配置。CLI 只打印这些值，不写 .env。尤其 EMBEDDING_MODEL 必须与建库时的字符串身份完全相同；同维度不同模型、模型名与 snapshot 路径互换也不被视为同一身份。

~~~powershell
python graphrag_assistant.py
~~~

菜单保留添加资料、提问、查看结构/文档以及备份清空；清空仍需输入 RESET 并生成备份。菜单“提问”会生成回答，需要 Moonshot/Kimi API 配置。本地模型的建库、检索和新评估不调用回答 API。显式配置 EMBEDDING_BACKEND=moonshot 是远程 embedding API 路径，不属于离线运行。

兼容入口 `graphrag_assistant.retrieve(question)` 仍返回 `(path, comma_separated_ids, context)` 三元组，第二项是逗号分隔的 chunk ID；`GraphRAGService.retrieve()` 返回包含 selection、核心/补充角色、决策、候选细节和耗时的丰富结果。retrieve_semantic 的旧模式是显式 legacy 入口。dingtalk_server.py 继续使用兼容入口。

## 旧索引与迁移

保留原来的 JSON/SQLite。默认 evidence 模式发现旧库会解释缺少的结构能力，并要求从原始文档显式建到新路径。不要把旧索引改名冒充新索引。

需要继续旧行为时显式设 RETRIEVAL_PROFILE=vector、vector_keyword、vector_graph、vector_topic_graph 或 full。旧 KB_PATH=knowledge_base.json 对应实际 knowledge_base.sqlite3；仅在显式 legacy 模式、SQLite 不存在且 JSON 存在时启动旧 JSON 迁移，保留原 JSON。该迁移不能恢复新模式所需的原文结构。

## 证据路径评估

新评估实际调用 EvidenceStore、GraphRAGService 和 EvidenceRetriever，与应用共用候选、核心/补充选择和渲染预算，无回答 API：

~~~powershell
python -B scripts/evidence_eval.py --dataset examples/evidence_smoke.json --out .tmp/evidence-smoke-01 --embedding-backend deterministic
python -B scripts/evidence_eval.py --dataset examples/evidence_smoke.json --out .tmp/evidence-model-01 --embedding-backend model --rerank
~~~

每次 --out 必须是不存在的新目录，内含真实 SQLite 索引、JSON 和 Markdown 报告。第二条命令默认只读本地缓存，需要下载时加 --allow-download。--split dev|test 只选择问题，所有输入文档都建库。源文档与手工引用目标标注均可缺省；缺省明确显示不可用和分母，不会生成虚构准确率。确定性后端始终标记 smoke；合成资料使用真实模型仍是 smoke。

具体格式、指标和预算边界见[证据助手说明](docs/evidence_assistant.md)。只检查已标注来源 ID 不跨 dev/test，近重复文本需要数据准备阶段另行检查。引用标签必须由外部或人工提供，不能从被评估的自动链接反推。

## Legacy 评估与外部对比

以下旧评估命令保留用于旧五种 profile，不评估新 evidence 路径：

~~~powershell
python scripts/retrieval_eval.py --embedding-backend deterministic --out-json .tmp/eval.json --out-md .tmp/eval.md
python scripts/retrieval_eval.py --embedding-backend model --out-json .tmp/eval-model.json --out-md .tmp/eval-model.md
python -B scripts/external_comparison.py --dataset examples/comparison_smoke.json --out .tmp/comparison-smoke-01 --system project --profile vector
python scripts/cpu_benchmark.py
~~~

旧检索烟测限定文档，不加载 cross-encoder；外部对比使用独立 legacy adapters。当前 legacy 启用重排时会先截断候选，历史 full 图/关键词扩展可能突破重排候选上限的行为已调整；关闭重排时不声称所有 profile 总候选数相同。旧报告未重算或改写。已有 400 文档、30 道 dev 题及 GPU 结果属于旧实现，不是新证据路径验证，更不能据此宣称新路径胜出。参见[外部对比说明](docs/external_comparison.md)、[CRUD-RAG 试运行](docs/crud_pilot.md)和[集群说明](cluster/README.md)；GPU 实验独立于本机 CPU 功能检查。

## 测试与设计

~~~powershell
python -m pip install -r requirements-dev.txt
python -B -m pytest -q -W error -p no:cacheprovider
~~~

离线测试用确定性后端或注入假模型，真实 SQLite 与服务用于集成验证。功能测试通过不代表检索质量优势。设计与兼容边界见[系统设计](docs/system_design.md)，旧评估说明见[评测说明](docs/evaluation.md)。

## License

[MIT](LICENSE)
