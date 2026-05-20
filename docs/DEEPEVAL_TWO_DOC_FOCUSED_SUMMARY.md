# DeepEval Two-Document Focused Summary

本轮使用 DeepEval 作为第三方自动评估层，题目为 6 道代表题：

- `AIF01` 算力瓶颈逻辑链
- `AIF02` 监管沙盒演化
- `MTF01` 财务表格 + 原因说明
- `MTF02` 经营现金流下降与财务公司
- `MTF04` ESG + 分红 + 回购
- `MTF06` 核心竞争力 + 经营计划 + 风险

系统对比：

- `NaiveRAG`
- `FrozenGraphRAG`
- `V2GraphRAG`

## 1. DeepEval 自动指标结果

| System | Answer Relevancy | Faithfulness | Faithfulness Errors |
|---|---:|---:|---:|
| NaiveRAG | 0.881 / 6 cases | 0.938 / 3 valid cases | 3 |
| FrozenGraphRAG | 1.000 / 6 cases | 0.887 / 4 valid cases | 2 |
| V2GraphRAG | 0.957 / 6 cases | 0.983 / 4 valid cases | 2 |

原始文件：

- `deepeval_two_doc_focused_cases.json`
- `deepeval_two_doc_focused_report.json`
- `deepeval_two_doc_focused_report.md`

## 2. 可以放进汇报的结论

DeepEval 的 Answer Relevancy 指标显示：

- NaiveRAG 在复杂财报题上明显掉分，尤其 `MTF01` 只有 0.471。
- FrozenGraphRAG 和 V2GraphRAG 整体更稳定。
- V2GraphRAG 的答案相关性略低于 Frozen，主要是 `AIF02` 和 `MTF02` 被 judge 认为有部分展开不够直接；这需要人工核查具体答案，不能只看自动分。

DeepEval 的 Faithfulness 指标显示：

- V2GraphRAG 在有效样本中最高，平均 0.983。
- 但 Faithfulness 有多次 invalid JSON，说明 Kimi 作为 DeepEval judge 时并不完全稳定。

## 3. 必须诚实说明的限制

DeepEval 本轮不是唯一裁判，原因：

1. `FaithfulnessMetric` 多次出现 invalid JSON，自动评估存在模型格式不稳定问题。
2. DeepEval 对中文财报数字、表格字段、主体关系的细粒度判断不如人工 key point 核查稳定。
3. 部分答案因为多给背景或解释，被 Answer Relevancy 扣分，但并不一定代表证据错误。

因此更稳妥的表述是：

> 我们采用“检索证据命中率 + DeepEval 自动评估 + 人工核查表”的混合评估。DeepEval 提供第三方指标参考，人工核查负责确认数字、主体、原因和跨章节证据是否真实命中。

## 4. 与检索层结果结合后的判断

检索层 focused 评估显示：

- 白皮书：v2 为 43/46，高于 Frozen 的 42/46 和 Naive 的 40/46。
- 茅台年报：加入相邻 chunk 补桥后，v2 为 44/46，高于 Frozen 的 41/46 和 Naive 的 31/46。

DeepEval 答案层显示：

- GraphRAG 系统整体答案相关性高于 Naive。
- V2 的 Faithfulness 在有效样本中最高。
- 但 DeepEval judge 存在失败样本，所以不能单独作为结论来源。

## 5. 推荐 PPT 话术

> 在评估上，我们没有只依赖 LLM 打分。首先用 key point 检查检索证据是否命中，再用 DeepEval 从答案相关性和忠实性角度做第三方自动评估，最后人工核查数字、主体和跨章节证据。结果显示，v2 在跨 chunk 表格补桥和多证据覆盖上比普通 RAG 更稳定；同时我们也保留 DeepEval judge 失败样本，说明自动评估只作为辅助参考。

