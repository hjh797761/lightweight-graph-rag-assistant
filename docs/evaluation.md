# 可复现评测说明

## 1. 目标

当前评测回答的是“不同检索组件在同一管线中是否找到了所需证据”。它不把语言流畅度混入检索对比，也不把普通 RAG 的未限定检索和 Graph RAG 的限定检索放在一起比较。

## 2. 公开材料

`examples/` 包含两份原创合成文档：

- 技术笔记：温度传感器、控制器和冷却流程之间的跨句关系。
- 财务风格笔记：本期数、上期数、同比变化和原因。

问题及证据点位于 `examples/eval_questions.json`。所有路径均相对仓库，不依赖作者电脑、私有 PDF、冻结知识库或仓库外的 `main.py`。

## 3. 公平比较

`scripts/retrieval_eval.py` 对每道题依次运行五种 `RetrievalProfile`。同一道题的 profile 共用：

- 同一 SQLite 数据与 chunk；
- 同一 embedding 后端；
- 同一文档限定；
- 同一 `top_k` 证据预算；
- 同一指标和报告格式。

profile 只改变其名称声明的组件。

## 4. 指标

- `evidence_recall`：期望证据点中有多少出现在召回上下文。
- `reciprocal_rank`：首个包含任一期望证据的 chunk 的倒数排名。
- `selected_chunks` 与 `scores`：便于逐题审计排序。
- `retrieval_seconds`：单次检索墙钟时间，仅用于观察，不设置性能门禁。

匹配前只使用通用归一化，不为某个金额、专名或预期答案添加特殊候选。

## 5. 运行

无需网络、模型或 API：

```powershell
python scripts/retrieval_eval.py --embedding-backend deterministic --out-json .tmp/eval.json --out-md .tmp/eval.md
```

真实本地模型：

```powershell
python scripts/retrieval_eval.py --embedding-backend model --out-json .tmp/eval-model.json --out-md .tmp/eval-model.md
```

可通过 `--questions` 和重复的 `--document` 参数传入自己的材料。报告会记录 embedding backend，确定性烟测结果不能冒充真实模型质量。

## 6. 历史结果说明

早期仓库文档中的数字来自未公开资料、不同检索范围或无法由当前仓库重建的脚本，因此只属于历史、不可复现快照，不再作为当前版本的性能证据。当前项目不预填“新版优于基线”的结论；请运行公开脚本并检查逐题证据后再形成判断。
