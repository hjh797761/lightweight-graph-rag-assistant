# 系统设计

默认应用是证据关联路径；graphrag 包名及旧交互/Python 入口保留兼容。这里把新路径和仍保留的 legacy 组件分开说明，已有旧评估报告不自动成为新实现的证据。

## 默认 evidence 数据流

structure.py 解析 TXT/Markdown 明示结构、DOCX Heading 样式和 PDF 页文本，保留可回查定位与原文区间。EvidenceStore 保存完整解析来源，再通过 GraphRAGService._embed_prepared_document 分批向量化、事务提交进度。完整来源先验证，避免中断后续建时混入已变化的后半文档。只有 complete 文档进入检索；同维度但不同 backend/model 的配置仍拒绝使用。

存储包含来源 chunk、标题路径、定位、结构链接和中文感知词法索引。embedding 为有限值向量，检索用 NumPy 矩阵点积；向量与词法数据都按同一个完整文档范围筛选。范围匹配精确文档 ID，其次精确名称或路径；未知、歧义、未完成范围明确报错。

EvidenceRetriever 先各取有界向量/BM25 候选，再由 rank_fusion.py 做 RRF，融合池截至 candidate_k，可选 BGE cross-encoder 只对该池重排。报告区分各通道候选、原始并集、融合池与实际送入重排数量。模型异常传播；无重排时使用 RRF 排序，重排器遗漏候选会明确记录，不悄悄回退为核心。

evidence_context.py 是唯一核心/补充选择和上下文预算实现。它从排序结果取初始核心，读取一次这些核心的一跳链接，优先显式引用、其次同节直接邻接；最后可用未展开核心补足。节归属只作定位，无递归语义推理。分数只属于核心；补充为 None。标题、来源编号、角色、引用原因、缺失/部分覆盖说明等渲染文本一起计入预算。

数量、比例、重叠块计费与 token 计数详见[证据助手说明](evidence_assistant.md)。这些规则是一套可检查的工程策略，尚未证明质量优势；RRF、混合检索、显式结构链接和重排是已有设计思想，不宣称原创算法。

## 服务、接口和错误

config.py 读配置；默认 profile 为 evidence。service.py 按索引能力选择证据或 legacy 检索器，不静默转换旧索引。graphrag_assistant.py 负责兼容三元组和交互显示；服务返回 RetrievalResult.selection、候选明细与阶段墙钟时间。answering.py 单独负责回答模型，菜单提问会调用回答 API；本地建库、检索、新评估不会。

导入模块不加载模型。embeddings.py 首次需要时加载模型，离线由缓存约束，失败抛出 ModelUnavailableError；解析失败抛出 DocumentParseError。新 CLI 默认 cache-only，显式 --allow-download 才允许下载。.env.example 默认离线，既有配置不会被自动改写。

scripts/build_evidence_index.py 显式建立新文件；仅未完成 evidence 库可用原始配置和输入 --resume。建库先准备所有文档，防止模型失败后丢失待处理文档清单。scripts/evidence_eval.py 复用真实服务、存储和选择器，对内联资料输出索引与报告，无第二套预算逻辑，也不执行回答生成。

## Legacy 组件与迁移

ingestion.py、storage.py、graph.py、retrieval.py 保留旧的通用概念共现图、主题聚类和五种消融 profile：vector、vector_keyword、vector_graph、vector_topic_graph、full。其中 full 仍可动态选择少于 top_k 的输出；旧脚本不等价于新核心/补充选择。启用重排时先截断候选的当前行为限制了旧版额外图/关键词候选扩展；关闭重排时不声称所有 profile 总候选数相同。历史报告保留原状。

旧建库事务使 chunk、概念边与进度同时提交，防止中断后的重复边累积或跳过资料；旧主题聚类在完成文档后构建。这些表与检索策略不是新 evidence 索引的建库必需步骤。

默认 evidence 拒绝将旧 JSON/SQLite 自动升级；需要原始资料显式新建。仅显式 legacy profile 允许原有 JSON 转 SQLite 流程：临时库完成并核对后原子切换，保留原 JSON，不覆盖已有正式库。带图/主题 profile 还需原索引具备对应能力。旧数据库可以继续留在原位置，不应通过重命名绕过能力检查。

## 验证边界

测试覆盖真实 SQLite、来源范围、事务/进度、解析结构、候选上限、预算和三元组兼容。新评估的来源指标只是 source retrieval proxy；可选引用目标标签来自外部人工标注。查询失败保留并计入有正标签的质量分母，未标注指标为不可用。自动结构链接不能同时充当自身正确性的标签。

已有 400 文档、30 道 dev 题与外部/GPU 对比都是 legacy 条件下的结果。合成 CPU 集成、确定性 smoke、程序退出码和测试通过都不能证明新路径的检索优势或回答质量。没有新增质量阈值或正式评测承诺。
