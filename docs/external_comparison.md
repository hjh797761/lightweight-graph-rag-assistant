# 开源项目检索对比

本文及 `scripts/external_comparison.py` 描述 **legacy adapters**，不包含新的 `evidence` 适配器。当前提供本项目旧 profile 与 LlamaIndex VectorStoreIndex 的可执行适配。LightRAG 与 Microsoft GraphRAG 是候选，尚未跑出对比结果。合成 smoke 分数不代表 CRUD-RAG 成绩。

新默认路径及同真实服务评估见[证据助手说明](evidence_assistant.md)。既有 400 文档、30 道 dev 题及跨系统/GPU 结果属于旧实现条件，不能解释为新证据路径获胜。旧协议和报告均保留，不将新评估指标混入旧结果。

## 输入

参见 `examples/comparison_smoke.json`：documents 含普通文档 ID 与原文，queries 含问题 ID、question、dev/test split 和 relevant_documents。相关性等级为 0–4 的整数，至少一篇文档正相关。ID 唯一，相关文档必须存在。同一正相关来源不能跨开发与留出分组。

分组检查只针对已提供的文档 ID；同源近重复文本仍需在数据准备时审查。所有语料均参与检索，正确来源不传入检索器。split 只选择问题，不改变候选语料。CRUD-RAG 适配前先核对来源标注；答案字符串不能冒充完整相关性标签。正式实验需说明原始标注、选中 ID、抽样方法与数据许可。

## CPU 烟测

```powershell
python -B scripts/external_comparison.py --dataset examples/comparison_smoke.json --out .tmp/project-smoke-01 --system project --profile vector
```

外部依赖单独安装，推荐项目独立环境：

```powershell
python -m pip install -r requirements-comparison.txt
python -B scripts/external_comparison.py --dataset examples/comparison_smoke.json --out .tmp/llamaindex-smoke-01 --system llamaindex
```

每条命令启动独立进程。输出目录必须不存在，防止覆盖。真实模型增加 `--embedding-backend model`，默认只读缓存；明确需要下载时使用 `--allow-download`。`--rerank` 对两套流程启用 BGE reranker，按 candidate-k 限制重排数量并记录实际数量。Qwen3 的 query instruction 与专用 reranker 需另行适配。

## 对比含义

- LlamaIndex 调用真实 VectorStoreIndex，共享分块与 embedding，没有生成模型。这是特定向量检索配置，不代表该框架全部方案。
- 输出预算按 chunk 计算，再按首次出现顺序得到文档排名。Recall/MRR/nDCG 对该排名计算，短排名不补齐，报告同时显示返回 chunk 数。
- project full 默认保持动态选择，可能不足十个 chunk；各配置分别报告。实验参数 `--fixed-output-k` 只覆盖最终返回数量，不改变 top-k 候选/桥接预算。
- 当前 legacy 启用 reranker 时，在图/关键词扩展后按上限截断重排候选；关闭 reranker 时不声称总候选数相同，向量与原始并集数也可能不同。旧版 `full` 的额外候选曾可能突破重排上限，历史报告没有按当前代码重算，比较时必须检查代码版本与实际候选计数。
- warm-up 单独计时。逐题耗时包含编码、检索和可选重排；全部请求与成功请求的 P50/P95 分别记录。串行吞吐不是并发服务吞吐。
- 共享编码与各系统建库开销分列。project 按 profile 构建所需组件：仅图配置构建概念图，仅主题配置构建主题；LlamaIndex 包含持久化。旧版 project 对所有 profile 都建图和主题，新旧建库耗时需结合配置解释。
- GPU 指标是 PyTorch allocator 峰值，不是整卡显存。排队时间不计入查询性能。
- 查询失败保留错误并以零分计入质量均值，命令非零退出。建库或输入失败不生成成功报告。

## 展示

单卡固定返回预算与组件对照见[检索效率与选择策略检查](retrieval_efficiency.md)。该实验复用开发题，不增加独立样本量；各阶段耗时只是定位开销的诊断，不能直接当作 GPU 核函数耗时。

内部结果默认放 `.tmp/`。没有优势可以不发宣传文章；发布优势时附同轮完整指标、失败数与条件。更快但 Recall 降低需要同时展示。不要根据结果删题，或把模型升级归因于图检索。

资料：[LlamaIndex](https://github.com/run-llama/llama_index)、[LightRAG](https://github.com/HKUDS/LightRAG)、[Microsoft GraphRAG](https://github.com/microsoft/graphrag)、[CRUD-RAG](https://github.com/IAAR-Shanghai/CRUD_RAG)。LightRAG 需要真实 LLM 建图与来源映射；Microsoft 提醒建库成本，先做小样本适配再扩展。
