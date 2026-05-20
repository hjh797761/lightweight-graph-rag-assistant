# 外部轻量 GraphRAG baseline：nano-GraphRAG 对比摘要

## 为什么选 nano-GraphRAG

如果需要加入一个现成开源项目作为外部 baseline，更建议选 `nano-GraphRAG`，而不是此时再接入 `LightRAG`。

原因：

- `nano-GraphRAG` 的定位更接近我们的项目：轻量、GraphRAG、支持本地文本构建实体图。
- 之前已经完成过部署和对比，复用成本低。
- `LightRAG` 工程更完整，但接入变量更多，容易让汇报重点从“我们的系统设计”变成“调别人框架”。
- 当前阶段目标是增强验证可信度，不是再开一条大工程线。

## 已完成的对比设置

对比对象：

- 我们的 `v2 tree-off`
- `nano-GraphRAG local mode`
- `nano-GraphRAG naive mode`

测试文档：

- `guanggang.pdf`
- 全文约 256 页

nano 建图情况：

- 输入全文：`guanggang.pdf`
- 切分 chunk：约 456 个
- 图节点：约 674 个
- 图边：约 784 条
- 社区报告：约 45 个
- 全量建图耗时：约 42 分钟

这说明 nano 并不是“几行代码就完成 GraphRAG”。它的核心思路轻量，但完整跑起来仍包含：

- chunking
- entity extraction
- relation extraction
- graph clustering
- community report
- LLM cache
- query mode

因此它可以作为外部 baseline，但不适合直接整体搬进我们的项目。

## 代表题对比结论

我们选了 4 类代表题：

1. 氦气资源壁垒、长期协议、前沿储备技术和风险披露综合题。
2. Super-N/Fast-N、Super-N 50K、杭州基地、ASME、SOD 智驭引擎串联题。
3. 财务费用同比增长 `224.01%` 的官方解释题。
4. 反驳“海外大规模自有开采”的反幻觉题。

人工核查结论：

- v2 在精确财报数字和官方原因说明题上明显更稳。
- nano local 更像“实体社区摘要”，对宏观归纳有帮助，但在细粒度数字、表格附近证据、官方原因说明上不一定占优。
- nano naive 有时能命中原文片段，但整体仍容易漏关键证据或泛化回答。
- 在本轮代表题里，v2 tree-off 没有输给 nano；在 `224.01%` 这种精确证据题上，v2 明显优于 nano。

## 对我们系统的意义

这个对比可以支持三个汇报结论：

### 1. 我们不是只和普通 NaiveRAG 比

除了 NaiveRAG，我们还和开源轻量 GraphRAG 做过对照，说明当前 v2 的收益不是只来自 baseline 太弱。

### 2. 重型或开源 GraphRAG 不一定适合个人资料复习助手

nano 的建图成本并不低，尤其在 Kimi 8k 上下文条件下，还需要额外压缩上下文预算，否则容易超限。

### 3. 我们的轻量路线是有取舍依据的

我们没有照搬完整 community report 流程，而是保留：

- embedding 作为稳定入口
- topic scope 缩小语义范围
- concept graph 做关系扩展
- exact evidence guard 保护数字和主体
- adjacent chunk bridge 修复跨 chunk 表格证据

这比完整复制 nano 更符合当前项目目标。

## 建议放在 PPT 的方式

主 PPT 不必大篇幅讲 nano，可以在“与现有工作对比”或“验证补充”里放一句：

> 除 NaiveRAG 外，我们还用 nano-GraphRAG 作为开源轻量 GraphRAG baseline 做过对比。结果显示，nano 在宏观实体社区摘要上有优势，但在财报细粒度数字、官方原因说明和跨 chunk 表格证据上不稳定；我们的 v2 在代表题上没有输给 nano，且建库成本更低。

如果老师追问，可以展开：

- nano 全文建图约 42 分钟。
- 需要 LLM 做实体/关系抽取和 community report。
- 对精确数字题不天然占优。
- 我们的系统更偏向“课程资料/个人文档问答”的轻量、可解释和可回退。

