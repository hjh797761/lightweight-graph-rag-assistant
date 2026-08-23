# 系统设计

## 1. 边界与兼容层

`graphrag_assistant.py` 只保留旧 Python 函数和交互菜单。真实职责位于 `graphrag/`：

- `config.py`：加载 `.env` 和进程环境变量。
- `ingestion.py`：读取文档、切块、批量建库。
- `embeddings.py`：本地、OpenAI 兼容和确定性测试后端，以及 cross-encoder 精排。
- `storage.py`：SQLite schema、事务写入和向量矩阵缓存。
- `graph.py`：通用概念、共现边和主题聚类。
- `retrieval.py`：统一召回、消融 profile、排序和来源上下文。
- `answering.py`：领域无关的证据约束提示。
- `service.py`：迁移、摄取、检索和回答编排。

模块导入不会加载模型或访问网络。首次需要向量或精排时才懒加载对应模型。

## 2. 建库数据流

文档先按段落和长度切成 `(clean_text, context_text)`。一个批次的净文本通过一次 `encode_many` 生成连续 `float32` 矩阵，然后构造 chunk、抽取通用概念，并在同一个 SQLite 事务中写入：

```text
文档 -> 切块 -> 批量 embedding -> chunk + concepts + edges + progress
                                      \____________事务____________/
```

事务边界解决的具体问题是：进程若在 chunk 已写入但进度未更新时中断，续建会重复累计概念边；若进度先更新而 chunk 未完整写入，则会跳过资料。主键或 Git 无法修复用户运行时产生的半批数据。

最后一个批次完成后，系统按向量余弦相似度进行轻量贪心聚类，将 topic centroid 和 chunk 归属写入 SQLite。整个流程没有逐 chunk 固定休眠，也不会每处理一个 chunk 就重写整库 JSON。

## 3. 存储与矩阵召回

SQLite 保存 metadata、documents、chunks、chunk_concepts、concept_edges、topics 和 topic_chunks。embedding 以维度明确的 `float32` BLOB 保存。

检索时，目标文档范围内的向量一次性加载为连续 NumPy 矩阵：

```text
scores = chunk_matrix @ query_vector
```

矩阵按文档范围缓存在进程内。成功写入后缓存失效；事务失败时数据和缓存状态都不前进。

## 4. 同管线检索 profile

所有 profile 都从同一个 `Retriever` 入口运行，并保持相同文档范围和最终证据预算：

| Profile | 向量 | 关键词 | 图 | Topic | 精确证据 | 相邻桥接 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `vector` | ✓ |  |  |  |  |  |
| `vector_keyword` | ✓ | ✓ |  |  |  |  |
| `vector_graph` | ✓ |  | ✓ |  |  |  |
| `vector_topic_graph` | ✓ |  | ✓ | ✓ |  |  |
| `full` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

若默认服务启用了 cross-encoder，它在候选融合后统一精排。上下文中的每个证据块带 `[S编号]`、文档 ID 和 chunk ID。

## 5. 过拟合控制

生产提示只要求覆盖主体、时间、指标、单位、原因和条件，不包含任何评测材料中的专名或预期答案。证据匹配只进行领域无关的 Unicode NFKC、大小写、空白、千位分隔符和标点归一化。

公开评测资料独立存放在 `examples/`。测试会扫描生产模块和评测器，防止已知私有 fixture 词语重新进入逻辑。

## 6. 迁移与失败行为

旧 JSON 迁移使用唯一临时数据库。核对成功后通过原子改名切换；原 JSON 不删除。失败时抛出带异常链的 `MigrationError`，目标库不被替换，临时库保留用于诊断。

模型在线或离线加载失败会抛出 `ModelUnavailableError`；文档解析错误抛出 `DocumentParseError`。回答 API 缺少密钥时在调用点报告，不影响离线导入测试。
