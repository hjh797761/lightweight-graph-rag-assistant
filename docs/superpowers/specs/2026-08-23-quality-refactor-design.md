# Lightweight Graph RAG 全面质量重构设计

## 1. 目标与范围

在保留现有命令行入口、环境变量和钉钉调用方式的前提下，完成全面工程重构，解决公开评测不可复现和比较不公平、测试材料过拟合、首次运行配置矛盾、单文件职责过重、JSON 整库反复写入、逐 chunk embedding、固定休眠以及缺少测试与 CI 等问题。

本轮不主动处理检索正确性扩展、OCR 与表格理解增强、文档内容变更识别、钉钉鉴权、提示注入和隐私策略。删除过拟合特例及为此引入的通用格式归一化属于明确范围。

## 2. 外部兼容边界

保持兼容：

- 执行入口：`python graphrag_assistant.py`。
- CLI 菜单的上传、提问、查看知识结构、查看文档列表、备份并清空和退出操作。
- 既有 `KB_PATH`、Moonshot、embedding、reranker 和 retrieval 环境变量。
- Python 入口：`load_knowledge_base`、`retrieve`、`retrieve_semantic`、`generate_answer`。
- `dingtalk_server.py` 对上述入口的调用方式。

允许变化：

- 新知识库默认使用 SQLite。
- 旧 `knowledge_base.json` 首次使用时迁移到同目录 SQLite。
- 旧 API 由兼容外壳委托给新的应用服务。
- 错误信息改为明确的配置、模型、解析、迁移或存储错误。

## 3. 目标结构

```text
graphrag_assistant.py
dingtalk_server.py
graphrag/
├── __init__.py
├── config.py
├── errors.py
├── models.py
├── normalization.py
├── ingestion.py
├── embeddings.py
├── storage.py
├── migration.py
├── graph.py
├── retrieval.py
├── answering.py
└── service.py
scripts/
├── retrieval_eval.py
└── cpu_benchmark.py
examples/
├── eval_technical_notes.txt
├── eval_financial_notes.txt
└── eval_questions.json
tests/
├── test_config.py
├── test_normalization.py
├── test_embeddings.py
├── test_storage.py
├── test_migration.py
├── test_retrieval_profiles.py
├── test_answering.py
├── test_compatibility.py
└── test_public_eval.py
```

模块职责：

- `config.py`：加载项目根目录 `.env`，解析并验证配置。
- `errors.py`：定义配置、模型、解析、存储和迁移异常。
- `models.py`：定义文档、chunk、topic、候选和检索结果数据结构。
- `normalization.py`：领域无关的 Unicode、数字、单位和标点归一化。
- `ingestion.py`：读取现有支持格式、切块和批量建库。
- `embeddings.py`：管理 Sentence Transformers 和批量编码。
- `storage.py`：管理 SQLite 数据和向量矩阵缓存。
- `migration.py`：将旧 JSON 安全迁移为 SQLite。
- `graph.py`：构建和查询概念共现图及 topic。
- `retrieval.py`：多路召回、profile 消融、重排和上下文选择。
- `answering.py`：领域无关提示和 LLM 答案生成。
- `service.py`：组合建库、检索和生成，供 CLI 与钉钉共用。

本轮不改变现有 PDF、DOCX 和 TXT 的语义解析能力。

## 4. 存储和迁移

SQLite 保存 schema 元数据、文档、处理进度、chunk、embedding、概念关联、共现边、topic 和 chunk 归属。embedding 使用 `float32` BLOB；检索时按范围加载成连续 NumPy 矩阵并缓存，以矩阵乘法计算余弦相似度，写入后使相关缓存失效。

每批 chunk、概念边和处理进度在同一个 SQLite 批次中写入。具体失败场景是：进程在 chunk 已落盘但进度未更新，或进度已更新但 chunk 未完整落盘时中断，续建会重复累计概念边或跳过片段。Git、schema 版本、主键和普通测试无法恢复用户机器上的运行时半批数据，因此仅在此处使用批次原子写入。

不新增内容 hash、冻结 contract 或质量 gate。

旧 JSON 迁移流程：

1. 读取旧 schema 和数据。
2. 在同目录创建临时 SQLite。
3. 迁移文档、chunk、topic、图和处理进度。
4. 核对记录数量及引用完整性。
5. 成功后将临时数据库切换为正式数据库。
6. 原 JSON 始终保留，不自动删除或覆盖。

迁移失败时保留原 JSON 和临时数据库，抛出明确错误并输出恢复位置。若 `KB_PATH` 仍指向 `.json`，兼容层推导同目录 `.sqlite3` 路径。

## 5. 配置和模型

- 使用 `python-dotenv` 加载项目根目录 `.env`，操作系统环境变量优先。
- 默认允许首次下载 embedding 和 reranker 模型。
- `OFFLINE_MODE=1` 时传递 `local_files_only=True`。
- 离线模型缺失时立即给出模型名和预下载说明。
- 查询 embedding 支持单条编码；建库使用批量编码。
- embedding 持久化前转换为 `float32`。
- cross-encoder 默认开关、模型和候选数保持不变。

依赖拆分为 `requirements.txt`、`requirements-eval.txt` 和 `requirements-dev.txt`。`deepeval` 不再作为普通运行必装依赖。本轮不增加锁文件或依赖冻结。

## 6. 评测设计

所有 profile 使用同一知识库、chunk、embedding、reranker、文档范围、候选池和证据预算：

- `vector`：向量召回与相同 cross-encoder。
- `vector_keyword`：增加通用关键词信号。
- `vector_graph`：增加概念图召回。
- `vector_topic_graph`：增加 topic routing。
- `full`：再启用 exact guard、相邻块 bridge 和 dynamic top-k。

profile 只控制已声明组件，不加载私有 `main.py`、`main_experimental.py` 或冻结知识库。

仓库提供两份自行编写的合成文档和一份问题集：技术笔记测试跨段概念关系，财务风格笔记测试数字、单位和相邻原因说明。评测脚本默认运行公开 fixture，也允许通过参数传入用户文档和问题集。

输出逐题 JSON 和 Markdown，包含 Recall@k、MRR、关键证据覆盖率、检索耗时及选中的 chunk 与分数。指标只生成报告，不形成自动质量 gate。

## 7. 过拟合清理

从生产提示、证据排序和评测器删除：

- 中华第一库。
- 中国科学院合作项目示例。
- Super-N、Fast-N 等测试材料专名。
- 650.33 亿元和 60 亿元回购等预期答案特例。
- 其他只能由现有私有评测材料解释的行业词表或答案短语。

替代规则只允许 Unicode NFKC、大小写、全半角标点与空白、千位分隔符、数字与百分号或单位间空格的归一化，以及通用英文缩写识别。不再为某个预期答案手写候选变体。

答案提示只描述主体、时间、指标、单位、原因、条件和多子问题覆盖等抽象要求。评测 fixture 中的专名不得写入生产检索或答案代码。

## 8. 性能设计

- 切块后按配置批量调用 `model.encode`。
- 删除本地建库循环中的固定 `sleep(1)`。
- 每批只持久化一次。
- 向量召回使用连续 NumPy 矩阵乘法。
- 数据写入后使向量缓存失效。
- 提供 CPU benchmark，报告建库和查询耗时，不设置通过阈值。

本轮不引入图数据库、FAISS、HNSW 或 GPU 路径。

## 9. 错误处理

- 配置无效：`ConfigurationError`。
- 离线模型缺失：`ModelUnavailableError`。
- 文档读取失败：`DocumentParseError`。
- SQLite 操作失败：`StorageError`。
- JSON 迁移失败：`MigrationError`，并保留恢复文件。

底层异常保留为异常链，CLI 输出简洁消息，测试可检查根因。对跨边界操作不再使用无条件宽泛重试。

## 10. 测试

单元测试覆盖：

- `.env` 和系统环境变量优先级。
- 在线与离线模型参数。
- 通用格式归一化。
- 批量 embedding 与 `float32`。
- RetrievalProfile 开关。
- 领域无关答案提示。
- 已知污染短语不再出现在生产代码和评测器。

存储与迁移测试覆盖：

- 新库创建、批量写入、加载和续建。
- 批次中断不会形成半批状态。
- 旧 JSON 完整迁移。
- 迁移失败不覆盖原 JSON。
- 向量缓存随写入失效。

集成测试覆盖：

- 公开 fixture 从导入、建库、检索到上下文生成。
- embedding 与 LLM 使用确定性替身，不下载模型、不调用外部 API。
- 兼容入口仍能被旧 CLI 和钉钉代码导入。
- 各 profile 使用相同文档范围和检索预算。

GitHub Actions 在 Windows 与 Ubuntu 的 Python 3.10、3.12 上运行单元测试、集成 smoke test 和语法编译。本机另外运行真实 CPU 模型 smoke test；若模型缓存或网络不可用，报告须区分自动测试结果与真实模型验证状态。

## 11. 发布流程

用户已明确要求测试后直接推送 `main`：

1. 在 D 盘本地仓库实施。
2. 运行完整自动测试、语法检查、公开评测 smoke test 和污染词扫描。
3. 检查最终 diff、未跟踪文件和提交内容。
4. 获取远端状态并执行 `git pull --ff-only`。
5. 只有本地与远端可快进且验证仍通过时，普通推送 `main`。
6. 不强推、不重写远端历史；远端冲突时停止并报告。

## 12. 完成标准

- 现有 CLI 和 Python 入口保持可用。
- 旧 JSON 能迁移且原文件保留。
- 公开评测无需作者本机文件即可运行。
- 评测比较使用同一管线和相同范围。
- 已知评测材料特例从生产代码和评测器移除。
- `.env`、在线下载和离线模式与 README 一致。
- 建库使用批量 embedding、批次持久化和矩阵召回。
- 自动测试、smoke test、语法检查和 CI 配置齐全。
- 推送前完整验证取得新鲜成功证据。
