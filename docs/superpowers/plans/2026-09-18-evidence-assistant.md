# Lightweight Evidence Association Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not dispatch agents until the execution approach is selected. Review between dependent tasks.

**Goal:** 将已批准的轻量证据关联设计落到正常问答、建库与评测入口，保留旧模式和旧数据。

**Architecture:** 在现有 SQLite / NumPy / 本地模型基础上增加结构片段、FTS5 词项召回、RRF 融合与有预算的上下文组织。以 `EvidenceStore` 承载新索引、`EvidenceRetriever` 编排新流程，保留旧 `KnowledgeStore` 和 `Retriever` 的显式使用路径。服务在只读检查库能力后分派，不隐式重建旧库。

**Tech Stack:** Python 3.10+、SQLite FTS5、NumPy、现有 sentence-transformers、pytest；PowerShell 本地执行，已有 CPU 测试环境。

---

## 执行位置、验证命令及状态

- 仓库：`D:/lightweight-graph-rag-assistant`。用户已授权测试后提交并推送 main；本计划阶段不发布运行代码。
- 规格：`docs/superpowers/specs/2026-09-18-evidence-assistant-design.md`，已由用户确认。
- 2026-09-18 用户选择子代理实施；执行前工作区干净，内存数据库可创建 FTS5 表，改动前完整测试 58 项通过。以下按任务记录实施进度。
- 不使用 C 盘输出、不调用回答或抽取 API、不自动 SSH、不动已有模型缓存、数据库、集群任务或实验报告。
- 不新增散列、冻结 contract 或基准门槛；只使用事务、已有版本字段、普通状态和测试。

所有 pytest 命令在仓库根目录运行，解释器为：

```powershell
& '.tmp/comparison-venv/Scripts/python.exe' -B -m pytest tests -q -W error -p no:cacheprovider --basetemp=.tmp/pytest-evidence-final
```

任务级执行将 `tests` 换成列出的测试文件，`--basetemp` 使用任务独有的 `.tmp/pytest-evidence-tN`。
每次先观察预期失败，再实现，再观察通过；不得把依赖、路径或语法错误当成功的失败验证。
完整最终命令须在全部实现后重新执行，旧的 58 项通过记录不能替代它。

## 文件职责

| 文件 | 职责 |
|---|---|
| `graphrag/rank_fusion.py` | 去重后的 RRF、稳定排名及通道记录 |
| `graphrag/lexical.py` | 中文／标识符词项、FTS 字面查询构造 |
| `graphrag/structure.py` | 结构解析、来源位置、切块、有限引用解析 |
| `graphrag/evidence_store.py` | 新库创建、事务建库、词项查询、局部关系读取、能力检查 |
| `graphrag/evidence_context.py` | 核心／补充选择、实际渲染预算及原因记录 |
| `graphrag/evidence_retrieval.py` | 范围、召回、融合、重排、上下文编排 |
| `graphrag/models.py` | 新结果和来源数据类型；原 `ChunkRecord` 兼容 |
| `graphrag/retrieval.py` | 旧模式的候选预算修正；不混入新模式实现 |
| `graphrag/service.py`、`config.py` | 新默认、能力分派、参数检查及兼容 |
| `scripts/build_evidence_index.py` | 显式输出新库的离线建库入口 |
| `scripts/evidence_eval.py` | 调用实际服务流程的独立新协议报告 |
| `graphrag_assistant.py`、`.env.example` | 菜单及旧公开函数接入、示例配置 |

下面的 API 名称是本计划内的一致实施接口，不是对外冻结协议。若普通重构调整名称，必须同步调用方和测试。

## Task 1：统一旧检索器的候选预算

完成：`8b08a31`。实现者记录 RED（12 != 2）及完整 71 项通过；独立规格与质量审查各运行 28 项通过，无阻塞项。质量审查建议后续加一例“非向量首位候选通过融合进入重排”的保护测试。

**Files:** Modify `graphrag/retrieval.py`, `scripts/external_comparison.py`; Test `tests/test_retrieval_efficiency.py`。

- [x] 在现有 `fixture` 上增加以下回归测试，观察当前实现失败：

```python
def test_full_rerank_respects_core_candidate_limit(tmp_path):
    store, embedder, _ = fixture(tmp_path)
    counts = []
    def rerank(query, chunks):
        counts.append(len(chunks))
        return [(c, float(len(chunks) - i)) for i, c in enumerate(chunks)]
    Retriever(store, embedder, vector_recall_k=2, reranker=rerank).retrieve(
        "冷却", profile="full", top_k=2)
    assert counts == [2]
```

- [x] 运行 `tests/test_retrieval_efficiency.py`，预期新测试发现重排收到超过二个候选。
- [x] 在 `ranked` 送进重排前截取 `self.vector_recall_k`；候选上限验证为正整数。关键词分支仅在命中分数大于零时加入新 ID，保留已有向量候选的零增量语义。旧模式原打分权重不改。
- [x] 比较脚本的重排包装只记录实际收到的候选数，不再独占预算控制；适配器不得再依靠包装实现正确性。
- [x] 增加无词项命中不扩充候选、关闭重排、top_k 大于实际候选数、非法上限的测试；运行检索和外部比较测试。
- [x] 检查 diff 后提交 `fix: enforce rerank candidate budgets in core retrieval`。

## Task 2：可手算的融合与领域无关词项

完成：`ceb0df3`。先观察 17 项接口缺失失败，再全部通过；独立规格及质量审查各 17 项通过，整库 88 项通过。FTS 使用可逆词项编码保护标识符，无新增 hash。

**Files:** Create `graphrag/rank_fusion.py`, `graphrag/lexical.py`, `tests/test_rank_fusion.py`, `tests/test_lexical.py`。

- [x] 先写 RRF 的输入输出测试：

```python
def test_rrf_deduplicates_each_channel_and_counts_missing_as_absent():
    from graphrag.rank_fusion import reciprocal_rank_fusion
    rows = reciprocal_rank_fusion({"vector": ["a", "a", "b"], "lexical": ["b"]}, k=60)
    assert [row[0] for row in rows] == ["b", "a"]
    assert rows[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert rows[1][1] == pytest.approx(1 / 61)
```

- [x] 观察缺少新接口导致的失败；实现返回 `(chunk_id, fusion_score, ranks_by_channel)` 的 RRF，通道内保序去重，按 `(-score, id)` 排序。空通道不产生条目，k 必须是正整数。
- [x] 先写词项测试：

```python
def test_terms_do_not_cross_punctuation():
    from graphrag.lexical import lexical_terms
    terms = lexical_terms("北京，大学 AB_12")
    assert "北京" in terms and "大学" in terms
    assert "京大" not in terms
    assert "ab_12" in terms
```

- [x] 观察失败后实现 NFKC、小写、中文连续段二元组及单字段、英文数字标识符。FTS 的预分词存储与查询使用同一词项编码，标识符中的下划线／连字符不能被 tokenizer 二次拆分而改变相等关系。
- [x] `fts_query(text)` 返回经过引用和转义的词项 OR 表达式，空输入返回空字符串；测试引号、括号、`OR` 和 `NEAR` 只作为内容进入查询，不提供任意 FTS 语法入口。
- [x] 运行两个文件，测试空输入、同分 ID 稳定顺序、非法 k，提交 `feat: add deterministic lexical terms and rank fusion`。

## Task 3：结构片段和有依据的关系

完成：`71fbc78`、`70cf348`、`3df9f01`。独立审查发现并回归修复标题切片误引用、英文编号截短和跨行引用丢失；增加真实标题结束位置及原文空白字段。最终规格／质量审查各 23 项通过，完整 109 项通过。DOCX 测试使用最小实际 OOXML。后续 Task 7 发现 PDF／模型组合测试的原生退出崩溃，新证据 PDF 路径改用已有纯 Python PyPDF2，旧解析入口与依赖版本不变；测试使用真实 PDF 文本流与 xref。

**Files:** Create `graphrag/structure.py`, `tests/test_structure.py`; Modify `graphrag/models.py`。

- [x] 定义并测试数据形态：`SourceBlock` 保存 text、section_id、title_path、locator、start/end；`EvidenceChunk` 继承兼容 chunk 字段，额外保存同样来源信息；`EvidenceLink` 保存 source_id、target_id（可空）、kind、basis、status。
- [x] 为 `parse_text(text, doc_id, format="markdown", chunk_size=800)` 写最小测试：

```python
def test_repeated_titles_keep_distinct_section_ids():
    from graphrag.structure import parse_text
    chunks = parse_text("# 范围\n甲。\n# 范围\n乙。", doc_id="d")
    assert len({c.section_id for c in chunks}) == 2
    assert all(c.doc_id == "d" for c in chunks)
    assert chunks[0].source_start < chunks[-1].source_start
```

- [x] 观察失败后实现按 Markdown 标题层级、有限条款标记分段，再在段内优先按段落／句界切块。仅超长单句做长度分割并记录 partial 状态。来源偏移始终指解析后的原文，不指归一化词项文本。
- [x] `parse_file(path, doc_id)` 对 TXT/MD 保留行号，对 DOCX 保留段落序号及真实 Heading 样式，对 PDF 保留页码；没有可信章节时 section_id 为空。复用已安装文档读取依赖，不增加 OCR。
- [x] 为 `build_links(chunks)` 写前向引用、重复编号、缺失目标和同章节邻接测试；不相邻或不同章不能产生 adjacent。
- [x] 实现编号引用只在同文档唯一匹配时 resolved，多匹配 ambiguous、零匹配 missing；保存引用所在原文及位置。引用由多个片段组成的条款时保留完整目标 ID 列表，以供部分覆盖判断。
- [x] 运行 `tests/test_structure.py` 及现有 ingestion/normalization 测试；提交 `feat: parse traceable document structure and explicit references`。

## Task 4：独立新索引与事务建库

完成：`c178dd1`。规格及质量独立审查各 21 项通过；整库 125 项通过。分批写入前使用 `prepare_document` 保存完整普通来源记录（无 hash），因此未处理后缀变化也会被拒绝；只读能力检查、独占创建、事务回滚、完整状态过滤及原文元数据往返均有测试。

**Files:** Create `graphrag/evidence_store.py`, `tests/test_evidence_store.py`; Modify `graphrag/errors.py`（增加明确的索引能力异常）。

- [x] 测试 `inspect_index(path)` 对旧库只读检查，不调用 `KnowledgeStore.__init__`；用读取前后文件字节比较验证未修改，不增加文件散列机制。

```python
def test_inspecting_legacy_database_does_not_write(tmp_path):
    from graphrag.storage import KnowledgeStore
    from graphrag.evidence_store import inspect_index
    path = tmp_path / "old.sqlite3"
    KnowledgeStore(path)
    before = path.read_bytes()
    assert inspect_index(path)["evidence_ready"] is False
    assert path.read_bytes() == before
```

- [x] 观察失败后实现只读 URI 打开、已有版本／表能力检查；不存在路径报告 missing，不创建空库。
- [x] `EvidenceStore.create(path)` 要求目标不存在，原子独占创建。沿用可兼容读取向量的基础表，新增来源、文档状态、关系和 FTS 表；新库中旧共现图／主题为空，不现场生成。FTS 不可用时明确失败，保留可诊断的未完成新库，不触碰别的文件。
- [x] `write_document_batch(doc_id, chunks, next_index, total_chunks)` 在同一事务写片段、来源、FTS 与进度；最后一批在同一完成事务中写关系和 complete 状态。失败时进度不前进；成功后使现有矩阵缓存失效。
- [x] 对重入批次按主键保留已验证的相同原文记录，完成时重建关系，不累加重复边。续建时比较全部解析片段和来源元数据，输入不一致明确要求新文件重建，不能将新文本拼到旧进度后。
- [x] 实现 `lexical_search(query, doc_id=None, limit=40)`、`links_for(source_ids, doc_id=None)`、`list_chunks(...)`、`vector_matrix(...)`。新证据路径只检索 complete 文档；空词项不执行 MATCH；范围条件在 SQL 内生效。
- [x] 测试事务中途异常、重复批次、前向引用完成、中断续建、输入变化、作用域、原库保护和 FTS 特殊查询；运行 storage/migration 全部测试。
- [x] 提交 `feat: add transactional evidence index without implicit migration`。

## Task 5：预算和证据角色

完成：`d7a23b8`、`fb04b0b`。独立规格及质量复审各 44 项通过；父任务同命令复核 44 项通过。最终渲染计入引用标签、理由及分隔；混合核心／补充重叠块保守地全计入补充预算。关联注释超额时可移除整个尾部核心并记录原因，后续填充不再扩展。零补充比例不改变首轮名额公式；关联可用性单独记录。

**Files:** Create `graphrag/evidence_context.py`, `tests/test_evidence_context.py`; Modify `graphrag/models.py`。

- [x] 增加 `EvidenceOptions`：candidate_k=40、lexical_k=40、rrf_k=60、max_chunks=6、max_supplements=2、context_budget=6000、supplement_fraction=0.30、budget_unit="characters"。正整数、非负补充上限及 `[0,1)` 比例均校验；token 模式必须提供明确对应回答模型的计数器。
- [x] `EvidenceSelection` 保存 chunks、roles、可空 scores、context、decisions、budget_used、budget_unit、status。测试完整渲染代价：

```python
def test_rendered_context_including_labels_obeys_budget():
    from graphrag.evidence_context import assemble_context
    from graphrag.models import EvidenceOptions
    result = assemble_context([], lambda source_ids: [], options=EvidenceOptions(context_budget=20))
    assert result.context == ""
    assert result.budget_used == 0
    assert result.status == "no_candidates"
```

- [x] 观察失败后实现规格第 6 节贪心过程。按 `min(S, K-1)` 计算补充名额，核心预留比例与实际补充是否启用一致；无任何可容纳核心时报告 budget_exhausted。
- [x] `assemble_context(ranked_core, link_loader, *, options, count_tokens=None)` 在首轮核心选定后，仅调用一次 `link_loader(selected_core_ids)`；回调返回关系及目标片段，不遍历整库。只从首轮选中核心扩展一次，显式引用优先于邻接；剩余预算填核心但不再扩展。
- [x] 每次增添或合并后重新渲染并计数，包含来源标签、章节标题、原因和分隔符。核心 score 来自相关性排序，补充 score 为空。共享目标保留全部来源理由，不重复占用片段名额。
- [x] 去重只按 ID 或同来源重叠区间；通过来源区间合并原文，遇到不连续片段显示明确分隔。多片段引用目标未全部纳入时标记部分覆盖；不能删掉差异数字、日期或否定语句。
- [x] 增加一跳环、两核心引用同目标、K=1、S=0、比例为零、无核心容纳空间、token 计数器缺失、标题很长、合并后计数和部分引用覆盖测试。
- [x] 运行 `tests/test_evidence_context.py`；提交 `feat: assemble budgeted core and supporting evidence`。

## Task 6：新检索编排与完整解释

完成：`57aac80`。规格及质量审查各 100 项通过；完整 193 项通过。重排计数记录实际输入工作量，省略候选明确记录，不以 RRF 恢复核心身份；候选原文使用存储权威记录。已有向量缓存的冷加载路径保留，不声称本轮优化了冷加载 SQL。

**Files:** Create `graphrag/evidence_retrieval.py`, `tests/test_evidence_retrieval.py`。

- [x] `EvidenceRetriever(store, embedder, *, reranker=None, options=None)` 只消费 Task 4 的读取接口；`retrieve(query, doc_scope=None, top_k=None)` 返回包含 selection、通道明细和阶段耗时的 `RetrievalResult`。
- [x] 先以临时真实 EvidenceStore 和确定性 embedding 写集成测试：两个文档共享相似词，只限定一个文档时，核心和补充都不得越界；同名范围要失败而非选第一个。
- [x] 观察失败后实现唯一范围解析、complete 文档向量矩阵、FTS 候选、RRF 截断、有限重排及 Task 5 组装。把已绑定文档范围的局部关系加载函数传给 assembler，由其在核心选择后读取，避免在两处重复选择核心。
- [x] 重排返回结果须属于送入集合、ID 不重复、分数有限；非法输出明确报错，不能将未知或重复候选带入解释。tie 用融合顺序再用 ID。
- [x] 无重排分支直接使用融合次序；按每个候选记录向量／词项排名、原始分数、融合分数、重排值及被舍弃原因。关系目标未参与重排时不编造其分数。
- [x] 测试“直接相关条款＋被引用条件”：条件即使不属于高相关核心，也能以 supplement 角色出现，并有原文引用原因；将同样数字放在无关系文档中不能得到该角色。
- [x] 验证空查询、不完整文档、无结构、无引用、重排关闭／失败、歧义范围、非法参数和所有候选上限。不得用固定 fixture 词语写生产规则。
- [x] 提交 `feat: retrieve explainable evidence with shared resource budgets`。

## Task 7：应用入口、显式建库与兼容

完成：`5875652`、`6623d3c`。独立规格审查 61 项通过，质量审查 54 项通过；质量审查发现建库提示缺少关闭重排开关，已增加复制配置后实际启动服务的回归测试，复审 2 项通过。最终完整测试 227 项通过、退出码 0。父任务另完成真实离线 CPU 建库、开／关重排检索与三元组接口核对，以及同进程 PDF 解析后加载真实模型的验证，均退出 0。以上只证明实现链路，不代表检索质量提升。

**Files:** Modify `graphrag/config.py`, `service.py`, `models.py`, `graphrag_assistant.py`, `.env.example`; Create `scripts/build_evidence_index.py`, `tests/test_evidence_service.py`。

- [x] 为 AppConfig 新增 `retrieval_profile="evidence"` 和 Task 5 配置的加载／校验写测试。原有显式 embedding backend 仍支持；新默认本地模型不引入额外 API。
- [x] 服务先用 `inspect_index` 检查，再打开对应 store；missing 可走明确的首次建库流程，旧库上的新模式请求报能力错误。显式旧模式仍走旧索引，不自动执行 JSON 迁移再伪装为新索引。
- [x] 正常 `retrieve(query)` 使用配置默认；显式 mode/profile 继续生效。兼容函数仍返回 `(path, comma_separated_ids, context)`；菜单打印核心／补充原因、预算单位及限制。
- [x] 原有重置确认与备份边界不移除；新库重置后保持新类型，不能重新创建成旧 schema。既有用户 .env 不改写。
- [x] 建库入口采用以下接口，输出路径必须新建，不提供默认 overwrite：

```powershell
& '.tmp/comparison-venv/Scripts/python.exe' -B scripts/build_evidence_index.py --help
```

帮助需包含 `--input`（一个或多个原文）、`--out`（新 SQLite）、`--batch-size`、`--embedding-backend`、`--model`、`--allow-download`；默认本地缓存。恢复只接受显式 `--resume`，验证未完成新库及来源一致性，拒绝在 complete 或旧库上恢复。

- [x] 用临时 Markdown／DOCX／PDF 测试文件、确定性后端验证建库、恢复、检索以及输出已存在时报错。原库文件内容、旧报告和 .env 不变。模型加载错误不得标记建库完成。
- [x] 运行全部 config/compatibility/answering/evidence_service 测试；提交 `feat: wire evidence mode into local assistant and explicit index builds`。

## Task 8：同实际路径的评测和使用说明

完成：`7f1a74b`、`586c25f`。独立规格审查发现静态未知范围在建库后才失败，已前移到模型／输出创建前校验，回归覆盖合法路径与 ID 优先规则；规格、质量复审各 19 项通过，无阻塞项。父任务完整 246 项通过、退出码 0；实际确定性与离线 CPU 真实模型评估 CLI 均完成并保留为 smoke。质量审查建议后续补多题不同补充数量的聚合分母测试，现有聚合实现正确。

**Files:** Create `scripts/evidence_eval.py`, `tests/test_evidence_eval.py`, `docs/evidence_assistant.md`; Modify `README.md`, `docs/system_design.md`, `docs/external_comparison.md`。

- [x] 新评测入口必须调用 EvidenceRetriever，不另写预算逻辑。独立输出目录要求不存在，保留历史比较脚本和历史结果，不偷偷把旧 profile 改名为 evidence。
- [x] 报告记录代码版本、模型配置、排序开关、预算单位、候选数、重排数、核心／补充 ID 与理由、完整实际渲染上下文、预算已用量、错误和阶段耗时。
- [x] 有来源文档代理标签时继续明确使用 source 指标；有显式引用目标标签时分别记录目标覆盖和误扩展；缺少标注写 unavailable，不从程序自己抽出的关系循环推导“准确率”。没有生成实验不报告答案质量。
- [x] 测试报告与同参数服务直接运行的 chunks、roles、context、预算一致；失败问题保留在分母中。确定性测试称为 smoke，不声称实际模型效果。
- [x] README 更新定位、全新库起步、旧库显式保留／切换、禁用重排、离线模式、引用部分覆盖、CPU 限制及错误处理。解释旧 full 候选预算修正和新模式 score 可空的变化。
- [x] 运行新报告测试和原全部比较测试；提交 `docs: document and evaluate the evidence assistant path`。

## Task 9：全量验证、审查和交付

本地验证及独立整合审查完成。父任务最终完整测试 246 项通过、退出码 0；独立整合审查另复跑 246 项通过、退出码 0，未发现阻塞推送的问题。53 个 Python 文件语法解析通过。真实 CPU 检查使用现有 BGE-small-zh-v1.5 与 BGE-reranker-base 快照，新输出分别位于 `.tmp/evidence-model-smoke-20260918/` 和 `.tmp/evidence-pdf-model-smoke-20260918/`。Markdown 建库分两批写入三个片段；开／关重排均返回一核心、一显式引用补充，完整上下文 374 字符（上限 1800），补充分数为空，公开三元组与服务结果相同。PDF 解析后同进程加载真实模型亦退出 0。最终新评估在 `586c25f` 功能代码上运行到 `.tmp/evidence-eval-final-model-20260918/`，两道 smoke 问题均无错误，实际设备均为 CPU；报告如实记录当时尚有文档修改的 dirty 状态。真实模型运行出现第三方 `cache_dir` 弃用提示，未屏蔽，退出码为 0。旧 `retrieval_eval.py` 的确定性 CLI 另跑到新目录并退出 0。发布前 fetch 已确认没有远端独有提交；实际推送结果在交付消息单列，不把发布前检查当作已推送。

**Files:** Review all task diffs; update this plan's completed boxes and `docs/evidence_assistant.md` verification notes with observed facts only。

- [x] 运行本计划开头完整 pytest 命令，确认退出码、实际通过／跳过数和警告。任何功能失败先定位原因，不能通过删掉断言或关闭测试掩盖。
- [x] 用已有本地模型缓存做小型 CPU 真实模型集成，指定新的 `.tmp/evidence-model-smoke-*` 输出，不下载／修改既有缓存。对离线 embedding、可选重排、引用补全和真实渲染预算核对；没有模型则准确报告缺项。
- [x] 实际 CLI 在临时新目录完成建库和检索，检查旧 API 三元组、说明文档命令与本机兼容。不要调用回答 API 作为隐含测试步骤。
- [x] 使用 requesting-code-review 技能审查变更；按所选执行方式处理审查，不做无关重构。重点检查旧库只读保护、预算一致性、关系来源和半成品状态。
- [x] `git diff --check`，核对只有本任务文件；提交并推送用户已授权的 main 前，重新确认远端状态，不 force push。若远端有新修改先检查差异，不能覆盖。
- [x] 交付记录区分本地实现／测试、真实 CPU 模型验证与尚未进行的 GPU／独立效果评估；GitHub 推送结果以交付消息中的实际远端核对为准。本轮没有提交集群作业，不将准备视作已提交。

## 规格覆盖与执行选择

规格 1–3 → 全部任务的边界；规格 4 → Task 1、2、6；规格 5 → Task 3、4；
规格 6 → Task 5、6；规格 7–8 → Task 4、7；规格 9 → 每项红绿测试及 Task 8、9；
规格 10 → 文档保留出处，不直接复制第三方代码。

执行方式：用户已选择逐任务子代理实现并审查，由主代理整合与验证。
每项依次做规格审查与代码质量审查；仅完成这些检查后更新完成状态。
