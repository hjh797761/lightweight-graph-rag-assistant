# 证据关联助手：范围、预算与评估

默认链路为向量与 BM25 召回 → RRF → 候选上限内可选 cross-encoder → 核心证据 → 一跳显式引用/邻接补充 → 渲染预算。它组织可回查材料，没有核实事实，也没有推断引用材料正确。各阶段的分数不在统一概率尺度上；补充语境不获得相关性分数，输出为 `None`。

## 来源结构与回查

Markdown 识别明确标题，忽略围栏示例内的伪标题。TXT 只识别有限的编号章/节/条格式和明示引用：中文第…章/节/条、英文 chapter/section/clause 与数字编号。它不识别任意自然语言语义引用。DOCX 使用真实 `Heading 1`–`Heading 9` 段落样式；视觉加粗或大字号不自动成为标题。PDF 仅用 PyPDF2 提取逐页文本，不提供 OCR、阅读顺序修复、版面或标题语义；扫描件可能没有可用文本。

TXT/Markdown 区间指向解码后的原文字符；DOCX 指向以换行连接的段落文本，PDF 指向以换行连接的提取页文本。这些都不是 PDF 文件字节偏移。证据块保留行、段落或页定位、标题路径及原文范围。长句按长度拆分时注明片段不完整；不会截断渲染尾部来强行塞入预算。

范围优先精确 ID，再匹配精确文档名或路径。模糊子串不被当作范围；同名歧义要求用 ID 或唯一完整路径。未完成文档不参与召回；显式指定未完成范围会报错。自动引用只在同文档内按编号匹配；重复目标是歧义，缺失目标保持未解析，不猜测补齐。

## 选择与预算

`K=max_chunks` 是核心加补充的总上限，`S=max_supplements` 的有效上限为 `min(S, K-1)`；初始核心配额是 `K-min(S,K-1)`。补充比例为正且有效 S 大于零时，初始核心使用 `(1-supplement_fraction) × context_budget` 的保留预算。比例为零时禁用补充并取消字符预算预留，初始核心数量仍按该配额，然后由未展开核心补足；K=1 时没有补充槽位。

只加载初始核心的一跳链接。显式引用目标优先于同节直接邻接，补充不再继续展开。未用完的数量或预算可由排序池中的未展开核心补足，不保证填满 K。引用目标可能跨多个 chunk；未完全纳入时显式显示覆盖数量和缺失 ID。

字符预算使用最终渲染字符串长度，包括 `[S编号]`、角色、标题、定位、原因、分隔符和部分覆盖说明。它只覆盖 context，不包含问题、system prompt 或模型输出，并非模型 token 窗口保证。token 模式仅通过程序接口提供，必须注入生成模型 tokenizer 的 `count_tokens`，不能使用 embedding tokenizer 代替；CLI 仅支持字符预算。

原文重叠且内容一致的区间可合并展示，但所有底层 ID 和角色保留。一个合并块同时包含核心和补充时，整块保守计入补充比例额度。链接原因等注释也可能使初始核心超预算，此时移除尾部完整核心并记录 `link_annotations` 决策，再考虑未展开核心回填；不会在正文中途截断。最终预算、选择/跳过原因和部分覆盖在结果中可检查。

## 新评估输入

`scripts/evidence_eval.py` 接受原创或有许可的内联 JSON：

```json
{
  "name": "manual evaluation",
  "purpose": "smoke",
  "documents": [{"id": "manual", "text": "# Section 1\nSee section 2.\n# Section 2\nException.", "format": "markdown"}],
  "queries": [{"id": "q1", "question": "What is the exception?", "split": "dev",
    "relevant_documents": {"manual": 1},
    "reference_target_ids": ["manual::chunk_000001"],
    "reference_targets_exhaustive": true}]
}
```

`format` 缺省为 markdown，也支持 txt（md/text 为对应别名）。ID 和正文必须非空；文档及问题 ID 各自唯一。来源相关性是 0–4 的整数映射；重复 JSON 键、未知来源、未知 chunk、重复引用标签、非布尔 exhaustive 都拒绝。所有文档完整解析并验证手工目标 ID 后才创建输出目录或加载模型。标签 ID 如 `manual::chunk_000001` 由解析顺序稳定生成，但修改来源可能改变它，标注者须核对对应原文。

`doc_scope` 可缺省，缺省检索全库；内联文档名称等于 ID，路径记作 `inline:ID`。精确 ID 优先于路径匹配；内联资料中未知范围在建库和模型加载前作为输入错误拒绝，不根据相关性标签补选范围。所有文档都建库，`--split dev|test` 只筛选问题。同一已标注正相关来源 ID 不得跨 dev/test；这不是近重复文本保证。旧 comparison validator 仍要求标签，新入口独立允许缺省标签，不改变旧协议。

`relevant_documents` 和 `reference_target_ids` 都可省略；引用目标是外部/人工预期片段，不从自动 `build_links` 派生。`reference_targets_exhaustive` 默认 false；true 必须显式提供目标列表，允许空列表表达“应当没有引用目标”。没有正标签时 coverage 不可用，不把空列表当 100% 覆盖。

## 指标和可审计报告

来源按返回 chunk 首次出现顺序去重，`source_recall` 是已返回正来源占标注正来源的比例，`source_mrr` 是第一正来源倒数排名。它们只是 source retrieval proxies，不是完整官方 CRUD 指标，更不评价回答质量。`reference_coverage` 是核心与补充所覆盖的手工目标占全部手工正目标的比例，并单列核心和补充贡献。

只有 exhaustive=true 的题目上，带 `explicit_reference` 原因的补充才参与 `false_reference_expansion`：未在手工目标集合里的这类补充数 / 这类补充总数。同一补充多条理由只计一次；只有 adjacency 原因的补充排除。无可评估的显式引用补充时指标不可用，不报告完美零错误。

报告保留总请求数、失败数、标签题数、正标签数、每项指标的实际分母。来源和正引用覆盖按有正标签的题目取均值，失败题为零且保留分母；未标注题保持 null。false expansion 汇总分母为实际可评估补充数，失败题没有可观察的展开，不能当作正确展开。逐题保留原问题 ID、原始人工标签、错误、角色、nullable 分数、核心/补充 ID、关联原因、选择决策、原始候选明细、各级候选数、实际 rerank 输入数、阶段墙钟时间、完整 context 与预算用量。失败时阶段轨迹不可用，不编造成功计数。

输出目录必须不存在，内含 `evidence.sqlite3`、`report.json` 和同样包含完整轨迹的 `report.md`；不转储 embedding 数组。记录 Git revision 与 dirty/unavailable、实际选项、budget unit、backend/model、是否注入组件、重排启停、包版本和可读到的模型设备。JSON 禁止 NaN。查询失败仍写部分报告并以非零状态退出；输入/建库失败不写成功报告，可能保留未完成数据库用于诊断，重跑使用新目录。

## 运行与证据边界

```powershell
python -B scripts/evidence_eval.py --dataset examples/evidence_smoke.json --out .tmp/evidence-smoke-01 --embedding-backend deterministic --max-chunks 2 --max-supplements 1 --context-budget 1800 --supplement-fraction 0.45
python -B scripts/evidence_eval.py --dataset examples/evidence_smoke.json --out .tmp/evidence-model-01 --embedding-backend model --rerank --candidate-k 40 --lexical-k 40 --rrf-k 60
```

默认模型只读缓存；允许下载需显式 `--allow-download`。`--model`、`--reranker-model` 可传本机绝对 snapshot 路径。程序接口 `run` 支持注入 embedder、reranker、生成 tokenizer 计数器；注入 reranker 须同时指定 `rerank=True`，报告区别 disabled/model/injected。注入组件的实际模型身份由调用者负责，不能把假模型运行称为真实模型质量实验。

确定性 embedding 强制将报告 purpose 标为 smoke，即使输入声称 benchmark；真实模型配 smoke 数据仍是 smoke。真实模型可能产生较高 CPU 延迟，需分别观察召回与重排成本，不能由测试运行时间推断生产吞吐。独立 GPU 实验须明确硬件、模型、语料和代码版本。

本次功能检查与旧 400 文档/30 dev 结果分开：现有旧报告不变，新证据路径没有质量优势结论。新增测试使用真实 SQLite 和应用服务比对最终 chunk、角色、context、预算与候选信息；这验证功能一致性而非任务成功率。外部系统比较仍见[legacy 外部对比](external_comparison.md)。

## 已观察到的本地验证（2026-09-18）

在功能代码 `586c25f` 上，Windows / Python 3.12.10 的完整测试以 `-B -W error -p no:cacheprovider` 运行，246 项通过，退出码 0；另对 53 个 Python 文件完成语法解析。测试数量是本次记录，不是质量阈值。

使用已有本地 BGE-small-zh-v1.5 和 BGE-reranker-base 快照、明确的离线设置和 CPU，实际运行了显式建库与新版评估 CLI。原创 smoke 数据的两道 dev 问题均无运行错误：有引用标签的问题返回一条核心和一条引用补充，渲染上下文 401 字符；另一道无标签问题返回两条核心，321 字符。两题上限均为 1800 字符、K=2、S=1、补充比例 0.45。无标签问题未被赋予质量得分，模型运行仍标为 smoke。

本轮另用中文人工材料核对了关闭和开启重排的检索、补充分数为空、显式引用理由、374 字符上下文及旧三元组结果一致；实际 PDF 解析后同进程加载本地模型亦正常退出。新 PDF 路径使用已有的纯 Python 解析依赖，避免了本机曾复现的原生 PDF/模型组合测试退出崩溃。真实模型日志仍出现第三方 `cache_dir` 弃用提示，未屏蔽；模型运行退出码为 0。

这些只是离线集成和实现一致性证据，不是独立检索效果实验、并发性能测试、回答质量评估或新版 GPU 验证，也不支持优于 LlamaIndex 等系统的结论。旧集群任务和历史结果没有因此重跑。

## 设计来源

沿用[已审阅设计的来源说明](superpowers/specs/2026-09-18-evidence-assistant-design.md)：排名融合参考 [Cormack 等的 RRF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)，混合融合的局限参考 [An Analysis of Fusion Functions for Hybrid Retrieval](https://arxiv.org/abs/2210.11934)，词法索引接口参考 [SQLite FTS5](https://www.sqlite.org/fts5.html)。小片段检索与结构语境恢复的职责分离参考 [LlamaIndex AutoMergingRetriever](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/retrievers/auto_merging_retriever.py)。本项目的一跳引用、配额和预算规则不是其原实现，参考这些设计不构成算法原创性或质量优势证明；没有直接复制第三方实现。
