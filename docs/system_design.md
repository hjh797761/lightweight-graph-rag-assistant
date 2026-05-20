# 系统设计说明

本项目实现的是一个轻量 Graph RAG 复习助手。设计目标是：在长文档资料中检索可靠证据，并把可追溯上下文交给大模型回答。

## 1. 建库阶段

建库阶段是离线流程，主要生成三类索引。

### 1.1 chunk 切分

系统读取 PDF、Word 或文本文件后，将文档切分成多个 chunk。chunk 是实际检索和回答的基本单位。

切分时会尽量保留段落边界，并通过 overlap 降低“答案刚好被切断”的风险。

### 1.2 chunk embedding

每个 chunk 会生成向量表示。默认模型：

```text
BAAI/bge-small-zh-v1.5
```

embedding 是召回入口，因为它对表达变化更稳定。即使问题和原文措辞不完全一致，也能先找到语义接近的候选片段。

### 1.3 topic_index

系统根据 chunk embedding 的相似度，将语义接近的 chunk 归入 topic。

topic 不是人工目录，也不要求文档有真实章节。它的作用是查询时缩小语义范围，减少跨主题、跨文档误召回。

### 1.4 knowledge_graph

每个 chunk 会抽取少量概念。概念与 chunk 建立连接；同一 chunk 内共同出现的概念形成共现关系。

图谱主要服务两个目标：

- 图扩展召回：从命中的概念找到相关 chunk。
- 可解释展示：展示回答依赖了哪些概念关系。

## 2. 查询阶段

查询阶段不是只跑一次向量检索，而是多路召回。

### 2.1 原问题 embedding 召回

先用用户问题生成 embedding，在全库中取语义相似 chunk。

这一路保证基础相关性。

### 2.2 topic routing

系统判断问题最接近哪些 topic，并在这些 topic 内增强召回。

topic 不会完全屏蔽全局结果，只是给同主题候选加权，避免过度限制。

### 2.3 concept graph 召回

系统从问题中抽取查询概念，去 knowledge_graph 中寻找相关概念和 chunk。

这一路用于补充普通 embedding 不容易发现的关系型证据。

### 2.4 expanded query 召回

系统会把高置信概念拼入查询，再做一次向量召回，用来补充同义表达或上下文表达不同的片段。

### 2.5 exact evidence guard

对于数字、金额、同比、主体、原因说明等强证据，系统会做额外保底。

这一步用于解决“语义相关但证据不完整”的问题。

### 2.6 adjacent chunk bridge

如果问题需要表格和原因说明，而二者分布在相邻 chunk 中，系统会补入相邻 chunk。

典型场景：

```text
chunk_012：现金流金额表
chunk_013：变动原因说明
```

普通 RAG 可能只召回其中一个；bridge 规则会让两类证据一起进入上下文。

## 3. Rerank 与 Dynamic Top-k

多路召回会带来更多候选，因此需要重新排序。

基础融合分数：

```text
score = 0.70 * embedding_score
      + 0.25 * graph_score
      + 0.05 * keyword_score
```

再叠加：

- topic rerank
- exact evidence floor
- cross-encoder 精排

最后根据问题复杂度动态选择 top-k：

- 简单事实题：较少 chunk
- 原因/比较题：中等 chunk
- 跨页/多证据题：更多 chunk

dynamic top-k 决定最终送给 LLM 的证据数量。

## 4. 答案生成

答案生成阶段本身不复杂。系统只把精排后的 chunk、检索路径和概念关系摘要交给 LLM。

重点是约束 LLM：

- 只能基于给定资料回答。
- 资料不足时说明无法确认。
- 尽量保留关键主体、数字和原因。

## 5. 为什么默认关闭 tree recall

早期版本尝试过章节树，但通用材料不一定有可靠目录结构。对于白皮书、年报、课件和混合资料，强依赖章节树会带来不稳定性。

最终 v2 默认使用：

```text
chunk embedding + topic_index + concept graph
```

tree recall 保留为可选实验路径，但不是主流程。
