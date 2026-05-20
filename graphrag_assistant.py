# “树状知识结构”
knowledge_tree = {}
knowledge_graph = {}
topic_index = {}
documents = {}
chunks_index = {}
processed_chunks = {}
import os
import time
import hashlib
import math
import re
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
KB_PATH = os.getenv(
    "KB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base.json")
)
SCHEMA_VERSION = 3
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY", "")
MOONSHOT_BASE_URL = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")
MOONSHOT_MODEL = os.getenv("MOONSHOT_MODEL", "moonshot-v1-8k")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence_transformer")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "embedding_graph")
VECTOR_RECALL_K = int(os.getenv("VECTOR_RECALL_K", "40"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "6"))
DYNAMIC_MIN_TOP_K = int(os.getenv("DYNAMIC_MIN_TOP_K", "3"))
DYNAMIC_MAX_TOP_K = int(os.getenv("DYNAMIC_MAX_TOP_K", "8"))
GRAPH_EXPAND_HOPS = int(os.getenv("GRAPH_EXPAND_HOPS", "2"))
GRAPH_RECALL_PER_CONCEPT = int(os.getenv("GRAPH_RECALL_PER_CONCEPT", "8"))
TREE_RECALL_PER_CONCEPT = int(os.getenv("TREE_RECALL_PER_CONCEPT", "8"))
ENABLE_TREE_RECALL = os.getenv("ENABLE_TREE_RECALL", "0") == "1"
ENABLE_TOPIC_ROUTING = os.getenv("ENABLE_TOPIC_ROUTING", "1") == "1"
TOPIC_CLUSTER_MIN_SIM = float(os.getenv("TOPIC_CLUSTER_MIN_SIM", "0.78"))
TOPIC_CLUSTER_FAR_MIN_SIM = float(os.getenv("TOPIC_CLUSTER_FAR_MIN_SIM", "0.86"))
TOPIC_CLUSTER_WINDOW = int(os.getenv("TOPIC_CLUSTER_WINDOW", "8"))
TOPIC_RECALL_K = int(os.getenv("TOPIC_RECALL_K", "4"))
TOPIC_RECALL_MIN_SCORE = float(os.getenv("TOPIC_RECALL_MIN_SCORE", "0.34"))
TOPIC_RERANK_WEIGHT = float(os.getenv("TOPIC_RERANK_WEIGHT", "0.08"))
EXACT_EVIDENCE_BOOST = float(os.getenv("EXACT_EVIDENCE_BOOST", "0.32"))
EMBEDDING_WEIGHT = float(os.getenv("EMBEDDING_WEIGHT", "0.70"))
GRAPH_WEIGHT = float(os.getenv("GRAPH_WEIGHT", "0.25"))
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.05"))
ENABLE_CROSS_ENCODER = os.getenv("ENABLE_CROSS_ENCODER", "1") == "1"
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-base")
CROSS_ENCODER_TOP_N = int(os.getenv("CROSS_ENCODER_TOP_N", "20"))
CROSS_ENCODER_WEIGHT = float(os.getenv("CROSS_ENCODER_WEIGHT", "0.65"))
CROSS_ENCODER_TEXT_CHARS = int(os.getenv("CROSS_ENCODER_TEXT_CHARS", "700"))
USE_LLM_QUERY_CONCEPTS = os.getenv("USE_LLM_QUERY_CONCEPTS", "0") == "1"
USE_LLM_BUILD = os.getenv("USE_LLM_BUILD", "0") == "1"
USE_GRAPH_EXPANSION = True
_llm_client = None
_embedding_model = None
_cross_encoder_model = None
def get_llm_client():
    global _llm_client
    if not MOONSHOT_API_KEY:
        raise RuntimeError("请先设置环境变量 MOONSHOT_API_KEY，再调用 Moonshot/Kimi 接口。")
    if _llm_client is None:
        from openai import OpenAI
        _llm_client = OpenAI(
            api_key=MOONSHOT_API_KEY,
            base_url=MOONSHOT_BASE_URL,
            timeout=60
        )
    return _llm_client

def llm_call_with_retry(client, prompt, max_retries=5, temperature=0):
    global _llm_client
    for attempt in range(max_retries):
        try:
            active_client = client or get_llm_client()
            completion = active_client.chat.completions.create(
                model=MOONSHOT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            _llm_client = None
            client = None
            wait = min(30, 3 * (attempt + 1))
            print(f"[Kimi重试 {attempt+1}/{max_retries}] {str(e)[:120]}，等待{wait}秒...")
            time.sleep(wait)
    return ""
def save_knowledge_base(processed_chunks=None):
    import json
    current_processed = processed_chunks if processed_chunks is not None else globals().get("processed_chunks", {})
    data = {
        "schema_version": SCHEMA_VERSION,
        "documents": documents,
        "chunks": chunks_index,
        "knowledge_tree": knowledge_tree,
        "knowledge_graph": knowledge_graph,
        "topic_index": topic_index,
        "processed_chunks": current_processed or {}
    }
    temp_path = f"{KB_PATH}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, KB_PATH)
    print(f"✅ 知识库已保存 → {KB_PATH}")

def load_knowledge_base():
    import json, os
    global knowledge_tree, knowledge_graph, topic_index, documents, chunks_index, processed_chunks
    if not os.path.exists(KB_PATH):
        print("⚠️  未找到知识库文件，从空库开始")
        return
    with open(KB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents = data.get("documents", {})
    chunks_index = data.get("chunks", {})
    knowledge_tree = data.get("knowledge_tree", {})
    knowledge_graph = data.get("knowledge_graph", {})
    topic_index = data.get("topic_index", {})
    processed_chunks = data.get("processed_chunks", {})
    if chunks_index:
        print(f"✅ 知识库加载成功，共 {len(documents)} 个文档，{len(chunks_index)} 个片段")
    else:
        print(f"✅ 旧版知识库加载成功，共 {len(knowledge_tree)} 个章节；建议清空后按新版结构重建")
    return processed_chunks

def make_doc_id(file_path):
    abs_path = os.path.abspath(file_path)
    digest = hashlib.md5(abs_path.encode("utf-8")).hexdigest()[:12]
    name = os.path.splitext(os.path.basename(file_path))[0]
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in name)[:30]
    return f"{safe_name}_{digest}"

def normalize_vector(values):
    norm = math.sqrt(sum(float(v) * float(v) for v in values))
    if norm == 0:
        return [0.0 for _ in values]
    return [float(v) / norm for v in values]

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    return _embedding_model


def get_cross_encoder_model():
    global _cross_encoder_model
    if _cross_encoder_model is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL, local_files_only=True)
    return _cross_encoder_model

def embed_text(text):
    if EMBEDDING_BACKEND == "moonshot":
        client = get_llm_client()
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return normalize_vector(response.data[0].embedding)

    model = get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return [float(x) for x in vector]

def cosine_similarity(vec_a, vec_b):
    if not vec_a or not vec_b:
        return 0.0
    size = min(len(vec_a), len(vec_b))
    return sum(float(vec_a[i]) * float(vec_b[i]) for i in range(size))

def minmax_normalize(items):
    if not items:
        return []
    values = [score for _, score in items]
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return [(item, 1.0 if max_score > 0 else 0.0) for item, _ in items]
    return [(item, (score - min_score) / (max_score - min_score)) for item, score in items]

def reset_knowledge_base(make_backup=True):
    import shutil
    from datetime import datetime
    global knowledge_tree, knowledge_graph, topic_index, documents, chunks_index, processed_chunks

    if make_backup and os.path.exists(KB_PATH):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(os.path.dirname(KB_PATH), f"knowledge_base.backup_{stamp}.json")
        shutil.copy2(KB_PATH, backup_path)
        print(f"✅ 已备份旧知识库 → {backup_path}")

    knowledge_tree = {}
    knowledge_graph = {}
    topic_index = {}
    documents = {}
    chunks_index = {}
    processed_chunks = {}
    save_knowledge_base(processed_chunks)
    print("✅ 知识库已清空，可以重新导入文档")

def is_bad_concept(concept):
    import re

    concept = (concept or "").strip()
    if len(concept) < 2 or len(concept) > 18:
        return True

    lowered = concept.lower()
    generic = {
        "公司", "本公司", "报告", "年度报告", "报告期内", "情况", "说明", "其他说明",
        "不适用", "适用", "单位", "合计", "金额", "项目", "内容", "数据", "信息",
        "万元", "元", "人民币", "有限", "有限公司", "以下", "以下简称", "其他",
        "董事", "监事", "股东", "子公司", "全资子公司", "公司全资子公司",
        "期末余额", "期初余额", "本期数", "上年同期数"
    }
    if lowered in generic or concept in generic:
        return True
    if re.fullmatch(r"[\d,._/%-]+", concept):
        return True
    if re.fullmatch(r"\d{4}年?", concept):
        return True
    if re.fullmatch(r"\d+(\.\d+)?", concept):
        return True
    if re.fullmatch(r"[A-Za-z]?\d+(\.\d+)?[A-Za-z]?", concept):
        return True
    if re.fullmatch(r"[\d,]+(\.\d+)?(万元|亿元|元|%|吨|项|年|月|日)?", concept):
        return True
    if any(bad in concept for bad in ["000.00", "不适用", "其他说明", "期末余额", "期初余额"]):
        return True
    if len(concept) <= 3 and any(ch.isdigit() for ch in concept):
        return True
    return False


def clean_chapter_title(title):
    import re

    title = re.sub(r"^\[page\s+\d+\]\s*", "", title or "", flags=re.IGNORECASE).strip()
    title = re.sub(r"\s+", " ", title)
    title = title.strip(" ：:，,。；;、|")
    if not title or "年度报告" in title:
        return "未分类"
    if re.fullmatch(r"[\d,./% -]+", title):
        return "未分类"
    if is_bad_concept(title):
        return "未分类"
    return title[:24]


def looks_like_table_or_structured_block(chunk):
    import re

    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in (chunk or "").splitlines()
        if line.strip()
    ]
    if len(lines) < 3:
        return False

    text = "\n".join(lines)
    short_lines = sum(1 for line in lines if len(line) <= 42)
    numeric_lines = sum(1 for line in lines if re.search(r"\d", line))
    multi_column_lines = sum(
        1 for line in lines
        if len(re.split(r"\s{2,}|\t+| {1,}(?=[\d,-]+(?:\.\d+)?)", line)) >= 3
    )
    repeated_labels = len(re.findall(
        r"(单位|项目|合计|金额|余额|本期|上期|期末|期初|增长|比例|原因|适用)",
        text
    ))

    numeric_ratio = numeric_lines / max(len(lines), 1)
    short_ratio = short_lines / max(len(lines), 1)
    has_dense_numbers = len(re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?%?", text)) >= 8
    has_layout_signal = multi_column_lines >= 2 or repeated_labels >= 3
    return has_layout_signal and (has_dense_numbers or numeric_ratio >= 0.45 or short_ratio >= 0.65)


def expand_graph(concept, query_terms=None, limit=5, min_weight=1):
    """返回该概念的高质量邻居，按共现次数从高到低排序"""
    if is_bad_concept(concept):
        return []
    if concept not in knowledge_graph:
        return []
    query_terms = [term for term in (query_terms or []) if term and not is_bad_concept(term)]
    neighbors = knowledge_graph[concept]
    filtered = []
    for neighbor, weight in neighbors.items():
        if weight < min_weight or is_bad_concept(neighbor):
            continue
        if query_terms and not any(
            term in concept or term in neighbor or concept in term or neighbor in term
            for term in query_terms
        ):
            continue
        filtered.append((neighbor, weight))
    sorted_nb = sorted(filtered, key=lambda x: x[1], reverse=True)
    return [f"{concept} -> {nb} (共现{w}次)" for nb, w in sorted_nb[:limit]]
def llm_extract_concepts_query(query):
    client = get_llm_client()

    prompt = f"""
请从用户问题中提取“核心查询概念”。

要求：
1. 只保留关键知识点（名词）
2. 一律用中文（如：特征值、正交性、边界条件）
3. 删除无意义词（如：什么、为什么、如何、有何）
4. 最多返回3个

问题：
{query}

只返回概念列表（中文逗号分隔）：
"""

    raw = llm_call_with_retry(client, prompt)

    concepts = raw.replace("，", ",") \
              .replace("、", ",") \
              .replace("\n", ",") \
              .split(",")

    cleaned = []
    for c in concepts:
        c = c.strip()

        # 过滤垃圾词
        if len(c) <= 1:
           continue
        if c in ["是什么", "有什么", "如何", "为什么", "问题", "定理"]:
           continue
        if any(x in c for x in ["(", ")", "=", "+", "-", "Æ", "Ç"]):
           continue

        cleaned.append(c)


    return list(set(cleaned))[:3]

def is_noise(chunk):
    noise_keywords = [
        "前言", "目录", "致谢", "版权", "Contents",
        "笔者水平", "考研辅导讲义", "习题课讲义",
        "广大师生批评", "感谢", "撰写"
    ]
    if looks_like_table_or_structured_block(chunk):
        return False
    # 同时过滤掉明显是目录的chunk（包含大量页码）
    import re
    page_numbers = re.findall(r'\.\s*\d+\s*$', chunk, re.MULTILINE)
    if len(page_numbers) > 3:
        return True
    return any(kw in chunk for kw in noise_keywords)
def resolve_doc_scope(doc_scope):
    if not doc_scope:
        return None
    doc_scope = doc_scope.strip()
    if doc_scope in documents:
        return doc_scope
    for doc_id, meta in documents.items():
        haystack = f"{doc_id} {meta.get('name', '')} {meta.get('path', '')}"
        if doc_scope in haystack:
            return doc_id
    return None

def vector_recall(query, doc_scope=None, top_k=VECTOR_RECALL_K, query_embedding=None, allowed_chunk_ids=None):
    doc_id = resolve_doc_scope(doc_scope)
    query_embedding = query_embedding or embed_text(query)
    candidates = []

    for chunk_id, record in chunks_index.items():
        if allowed_chunk_ids and chunk_id not in allowed_chunk_ids:
            continue
        if doc_id and record.get("doc_id") != doc_id:
            continue
        embedding = record.get("embedding")
        if not embedding:
            continue
        score = cosine_similarity(query_embedding, embedding)
        candidates.append((chunk_id, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]


def expand_query_concepts(query_concepts, hops=GRAPH_EXPAND_HOPS, limit=24):
    expanded = {}
    frontier = [(concept, 1.0, 0) for concept in query_concepts]

    while frontier:
        concept, score, depth = frontier.pop(0)
        if not concept or score <= expanded.get(concept, 0.0):
            continue
        expanded[concept] = score
        if depth >= hops:
            continue

        neighbors = sorted(
            knowledge_graph.get(concept, {}).items(),
            key=lambda item: item[1],
            reverse=True
        )[:8]
        for neighbor, weight in neighbors:
            next_score = score * (0.72 ** (depth + 1)) * (1 + min(float(weight), 5.0) / 10)
            frontier.append((neighbor, next_score, depth + 1))

    return [
        concept for concept, _ in sorted(expanded.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]


def tree_recall(query_concepts, per_concept=TREE_RECALL_PER_CONCEPT):
    recalled = {}
    for doc_tree in knowledge_tree.values():
        if not isinstance(doc_tree, dict):
            continue
        for chapter_nodes in doc_tree.values():
            if not isinstance(chapter_nodes, dict):
                continue
            for concept in query_concepts:
                for tree_concept, chunk_ids in chapter_nodes.items():
                    if concept != tree_concept and concept not in tree_concept and tree_concept not in concept:
                        continue
                    if not isinstance(chunk_ids, list):
                        continue
                    for rank, chunk_id in enumerate(chunk_ids[:per_concept]):
                        if chunk_id in chunks_index:
                            recalled[chunk_id] = max(recalled.get(chunk_id, 0.0), 0.88 - rank * 0.02)
    return list(recalled.items())


def graph_recall(query_concepts, per_concept=GRAPH_RECALL_PER_CONCEPT):
    expanded_concepts = expand_query_concepts(query_concepts)
    recalled = {}

    for chunk_id, record in chunks_index.items():
        chunk_concepts = record.get("concepts", [])
        if not chunk_concepts:
            continue

        hits = [concept for concept in expanded_concepts if concept in chunk_concepts]
        if not hits:
            text = record.get("text", "")
            hits = [concept for concept in expanded_concepts if concept and concept in text]
        if not hits:
            continue

        direct_hits = sum(1 for concept in hits if concept in query_concepts)
        score = 0.70 + 0.10 * min(len(hits), 3) + 0.08 * min(direct_hits, 2)
        recalled[chunk_id] = min(score, 1.0)

    ranked = sorted(recalled.items(), key=lambda item: item[1], reverse=True)
    return ranked[:max(per_concept * max(len(query_concepts), 1), per_concept)]


def build_topic_title(records):
    import collections
    import re

    entity_counter = collections.Counter()
    concept_counter = collections.Counter()
    chapter_counter = collections.Counter()
    entity_pattern = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+){1,}\b")

    for record in records:
        text = record.get("clean_text") or record.get("text", "")
        chapter = record.get("chapter")
        if chapter and chapter != "未分类":
            chapter_counter[chapter] += 1
        for entity in entity_pattern.findall(text[:1200]):
            if not is_bad_concept(entity):
                entity_counter[entity] += 1
        for concept in record.get("concepts", []):
            if concept and not is_bad_concept(concept):
                if entity_pattern.fullmatch(concept):
                    entity_counter[concept] += 2
                else:
                    concept_counter[concept] += 1

    title_parts = []
    for entity, _ in entity_counter.most_common(2):
        if entity not in title_parts:
            title_parts.append(entity)
    for concept, _ in concept_counter.most_common(3):
        if concept not in title_parts and not any(concept in part or part in concept for part in title_parts):
            title_parts.append(concept)
    if title_parts:
        return " / ".join(title_parts[:4])
    if chapter_counter:
        return chapter_counter.most_common(1)[0][0]
    return "未命名主题"


TOPIC_ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+){1,}\b")


def extract_topic_entity_anchors(record):
    text = (record.get("clean_text") or record.get("text", ""))[:1600]
    text += " " + " ".join(record.get("concepts", []))
    anchors = set()
    for entity in TOPIC_ENTITY_PATTERN.findall(text):
        if len(entity) >= 5 and not is_bad_concept(entity):
            anchors.add(entity)
    return anchors


def expand_topic_anchor_aliases(anchor):
    aliases = {anchor}
    parts = re.split(r"[-_/]", anchor)
    if len(parts) >= 3:
        tail = "-".join(parts[1:])
        if len(tail) >= 5:
            aliases.add(tail)
    return aliases


def anchors_match_query(anchors, query_anchors):
    if not anchors or not query_anchors:
        return True
    expanded = set()
    for anchor in anchors:
        expanded.update(expand_topic_anchor_aliases(anchor))
    return bool(expanded & query_anchors)


def can_merge_topic_cluster(chunk_anchors, cluster, position):
    cluster_anchors = cluster.get("anchors", set())
    if chunk_anchors and cluster_anchors and chunk_anchors.isdisjoint(cluster_anchors):
        return False
    return True


def rebuild_topic_index_for_doc(doc_id):
    global topic_index

    doc = documents.get(doc_id, {})
    chunk_ids = [
        chunk_id for chunk_id in doc.get("chunk_ids", [])
        if chunk_id in chunks_index and chunks_index[chunk_id].get("embedding")
    ]
    if not chunk_ids:
        return

    clusters = []
    for position, chunk_id in enumerate(chunk_ids):
        record = chunks_index[chunk_id]
        embedding = record.get("embedding")
        chunk_anchors = extract_topic_entity_anchors(record)
        best_index = -1
        best_score = -1.0
        for index, cluster in enumerate(clusters):
            if not can_merge_topic_cluster(chunk_anchors, cluster, position):
                continue
            score = cosine_similarity(embedding, cluster["centroid"])
            required_score = TOPIC_CLUSTER_MIN_SIM
            if position - cluster.get("last_position", position) > TOPIC_CLUSTER_WINDOW:
                required_score = TOPIC_CLUSTER_FAR_MIN_SIM
            if score >= required_score and score > best_score:
                best_index = index
                best_score = score

        if best_index >= 0:
            cluster = clusters[best_index]
            cluster["chunk_ids"].append(chunk_id)
            cluster["count"] += 1
            cluster["anchors"].update(chunk_anchors)
            cluster["last_position"] = position
            cluster["sum"] = [a + b for a, b in zip(cluster["sum"], embedding)]
            cluster["centroid"] = normalize_vector(cluster["sum"])
        else:
            clusters.append({
                "chunk_ids": [chunk_id],
                "count": 1,
                "anchors": set(chunk_anchors),
                "last_position": position,
                "sum": list(embedding),
                "centroid": list(embedding),
            })

    topics = {}
    chunk_to_topic = {}
    for index, cluster in enumerate(clusters):
        topic_id = f"{doc_id}::topic_{index:04d}"
        records = [chunks_index[chunk_id] for chunk_id in cluster["chunk_ids"]]
        concept_counts = {}
        for record in records:
            for concept in record.get("concepts", []):
                if concept and not is_bad_concept(concept):
                    concept_counts[concept] = concept_counts.get(concept, 0) + 1
        concepts = [
            concept for concept, _ in sorted(
                concept_counts.items(),
                key=lambda item: (-item[1], item[0])
            )[:12]
        ]
        title = build_topic_title(records)
        topics[topic_id] = {
            "id": topic_id,
            "doc_id": doc_id,
            "title": title,
            "chunk_ids": cluster["chunk_ids"],
            "centroid": cluster["centroid"],
            "concepts": concepts,
            "anchors": sorted(cluster.get("anchors", set())),
            "size": len(cluster["chunk_ids"]),
        }
        for chunk_id in cluster["chunk_ids"]:
            chunk_to_topic[chunk_id] = topic_id
            chunks_index[chunk_id]["topic_id"] = topic_id
            chunks_index[chunk_id]["topic_title"] = title

    topic_index[doc_id] = {
        "topics": topics,
        "chunk_to_topic": chunk_to_topic,
    }


def get_topic_scope(query, doc_scope=None, query_embedding=None):
    if not ENABLE_TOPIC_ROUTING or not topic_index:
        return None

    query_embedding = query_embedding or embed_text(query)
    scoped_doc_id = resolve_doc_scope(doc_scope)
    candidates = []
    for doc_id, doc_topics in topic_index.items():
        if scoped_doc_id and doc_id != scoped_doc_id:
            continue
        for topic_id, topic in doc_topics.get("topics", {}).items():
            centroid = topic.get("centroid")
            if not centroid:
                continue
            score = cosine_similarity(query_embedding, centroid)
            candidates.append((topic_id, score, topic))

    if not candidates:
        return None

    query_anchors = set()
    for anchor in TOPIC_ENTITY_PATTERN.findall(query or ""):
        if len(anchor) >= 5 and not is_bad_concept(anchor):
            query_anchors.update(expand_topic_anchor_aliases(anchor))

    strict_scope = False
    if query_anchors:
        matched = []
        for topic_id, score, topic in candidates:
            topic_haystack = " ".join(
                [topic.get("title", "")] +
                topic.get("concepts", []) +
                topic.get("anchors", [])
            )
            if any(anchor in topic_haystack for anchor in query_anchors):
                matched.append((topic_id, score + 0.20, topic))
        specific_matches = [
            item for item in matched
            if len(item[2].get("anchors", [])) <= 4
        ]
        if specific_matches:
            matched = specific_matches
        if matched:
            candidates = matched
            strict_scope = True

    candidates.sort(key=lambda item: item[1], reverse=True)
    top_score = candidates[0][1]
    selected = [
        item for item in candidates[:TOPIC_RECALL_K]
        if item[1] >= TOPIC_RECALL_MIN_SCORE or item[1] >= top_score - 0.08
    ]
    if not selected:
        selected = candidates[:1]

    allowed_chunk_ids = set()
    topic_scores = {}
    topic_titles = {}
    for topic_id, score, topic in selected:
        topic_scores[topic_id] = score
        topic_titles[topic_id] = topic.get("title", topic_id)
        allowed_chunk_ids.update(topic.get("chunk_ids", []))

    if strict_scope:
        expanded_allowed = set(allowed_chunk_ids)
        for chunk_id in list(allowed_chunk_ids):
            match = re.match(r"^(?P<doc>.+)::chunk_(?P<num>\d+)$", chunk_id)
            if not match:
                continue
            doc_prefix = match.group("doc")
            number = int(match.group("num"))
            for offset in (1, 2):
                neighbor_id = f"{doc_prefix}::chunk_{number + offset:06d}"
                if neighbor_id in chunks_index and anchors_match_query(
                    extract_topic_entity_anchors(chunks_index[neighbor_id]),
                    query_anchors
                ):
                    expanded_allowed.add(neighbor_id)
        allowed_chunk_ids = expanded_allowed

    return {
        "topic_ids": set(topic_scores.keys()),
        "topic_scores": topic_scores,
        "topic_titles": topic_titles,
        "allowed_chunk_ids": allowed_chunk_ids,
        "top_score": top_score,
        "strict": strict_scope,
    }


def filter_recall_by_topic(hits, topic_scope):
    if not topic_scope:
        return hits
    allowed = topic_scope.get("allowed_chunk_ids") or set()
    if not allowed:
        return hits
    filtered = [(chunk_id, score) for chunk_id, score in hits if chunk_id in allowed]
    return filtered if filtered else hits


def term_recall(query, query_concepts=None, top_k=24):
    terms = []
    for term in list(query_concepts or []) + extract_query_terms(query, limit=20):
        if not term or term in terms:
            continue
        if len(term) <= 1:
            continue
        terms.append(term)

    if not terms:
        return []

    recalled = []
    for chunk_id, record in chunks_index.items():
        text = record.get("clean_text") or record.get("text", "")
        hits = [term for term in terms if term in text]
        if not hits:
            continue
        numeric_hits = sum(1 for term in hits if any(ch.isdigit() for ch in term))
        score = 0.58 + 0.06 * min(len(hits), 5) + 0.06 * min(numeric_hits, 2)
        recalled.append((chunk_id, min(score, 0.95)))

    recalled.sort(key=lambda item: item[1], reverse=True)
    return recalled[:top_k]


def extract_query_terms(query, limit=12):
    import re

    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._%/-]{1,}|[\u4e00-\u9fff]{2,}", query or "")
    stopwords = {
        "\u4ec0\u4e48", "\u54ea\u4e9b", "\u5982\u4f55", "\u4e3a\u4ec0\u4e48", "\u662f\u5426",
        "\u4ee5\u53ca", "\u5206\u522b", "\u8bf4\u660e", "\u5206\u6790", "\u6bd4\u8f83",
        "\u6839\u636e", "\u8d44\u6599", "\u6587\u6863", "\u95ee\u9898", "\u56de\u7b54",
        "\u81f3\u5c11", "\u4e3b\u8981", "\u5177\u4f53"
    }
    terms = []
    for token in tokens:
        token = token.strip()
        if not token or token in stopwords:
            continue
        candidates = [token]
        if re.search(r"[\u4e00-\u9fff]", token) and len(token) > 6:
            suffixes = (
                "风险|费用|收入|资金|商誉|收购|原因|解释|增长|项目|政策|目标|"
                "监督|管理|主体|类型|业务|壁垒|供应|储备|现金流|毛利率|净利润|"
                "认证|基地|装置|合同|协议|暴露"
            )
            for match in re.finditer(rf"[\u4e00-\u9fff]{{0,4}}(?:{suffixes})", token):
                phrase = match.group(0).lstrip("年月日的和及与对从请在为是把将")
                if 2 <= len(phrase) <= 8:
                    candidates.append(phrase)
            for phrase in re.findall(r"[“\"']?([\u4e00-\u9fff]{2,8})(?:[”\"'])?", token):
                if len(phrase) <= 8 and phrase not in stopwords:
                    candidates.append(phrase)
            if "解释" in token or "原因" in token:
                candidates.extend(["原因说明", "变动原因说明"])
        expanded = []
        for candidate in candidates:
            expanded.append(candidate)
            if candidate.startswith("公司") and len(candidate) > 3:
                stripped = candidate[2:]
                expanded.append(stripped)
                for suffix in ("业务", "风险", "项目", "协议", "合同"):
                    if stripped.endswith(suffix) and len(stripped) > len(suffix) + 1:
                        expanded.append(stripped[: -len(suffix)])
            if candidate.startswith("本公司") and len(candidate) > 4:
                stripped = candidate[3:]
                expanded.append(stripped)
                for suffix in ("业务", "风险", "项目", "协议", "合同"):
                    if stripped.endswith(suffix) and len(stripped) > len(suffix) + 1:
                        expanded.append(stripped[: -len(suffix)])
            for suffix in ("业务", "风险", "项目", "协议", "合同"):
                if candidate.endswith(suffix) and len(candidate) > len(suffix) + 1:
                    expanded.append(candidate[: -len(suffix)])
        for candidate in expanded:
            if candidate and candidate not in stopwords and candidate not in terms:
                terms.append(candidate)
        if len(terms) >= limit:
            break
    return terms


def extract_exact_evidence_terms(query, limit=10):
    import re

    terms = []
    for raw in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?%?", query or ""):
        cleaned = raw.replace(",", "").strip()
        if not cleaned:
            continue
        if re.fullmatch(r"(19|20)\d{2}%?", cleaned):
            continue

        variants = [cleaned]
        if cleaned.endswith("%"):
            variants.append(cleaned[:-1])

        for variant in variants:
            if variant and variant not in terms:
                terms.append(variant)
            if len(terms) >= limit:
                return terms
    return terms


def exact_evidence_raw_score(query, record):
    exact_terms = extract_exact_evidence_terms(query)
    if not exact_terms:
        return 0.0

    text = record.get("clean_text") or record.get("text", "")
    concepts = set(record.get("concepts", []) or [])
    exact_hits = [term for term in exact_terms if term and term in text]
    if not exact_hits:
        return 0.0

    generic_terms = {
        "\u516c\u53f8", "\u672c\u516c\u53f8", "\u62a5\u544a", "\u8bf4\u660e",
        "\u539f\u56e0", "\u5b98\u65b9", "\u7b54\u6848", "\u62ab\u9732",
        "\u5185\u5bb9", "\u95ee\u9898", "\u56de\u7b54", "\u8bf7"
    }
    subject_terms = []
    for term in extract_query_terms(query, limit=16):
        if not term or term in generic_terms:
            continue
        if any(ch.isdigit() for ch in term):
            continue
        if 2 <= len(term) <= 12 and term not in subject_terms:
            subject_terms.append(term)

    exact_part = 1.0
    if not subject_terms:
        return 0.75 * exact_part

    subject_hits = sum(1 for term in subject_terms if term in text or term in concepts)
    if not subject_hits:
        return 0.75 * exact_part

    subject_part = min(1.0, subject_hits / min(len(subject_terms), 3))
    return 0.65 * exact_part + 0.35 * subject_part


def local_choose_chapter(text, existing_chapters=None):
    import re

    text = text or ""
    existing_chapters = [
        clean_chapter_title(chapter) for chapter in (existing_chapters or [])
        if chapter and clean_chapter_title(chapter) != "未分类"
    ]
    for chapter in existing_chapters:
        if chapter in text:
            return chapter

    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\[page\s+\d+\]\s*", "", line).strip()
        line = re.sub(r"\s+", " ", line)
        if not line:
            continue
        if "年度报告" in line or re.fullmatch(r"\d+\s*/\s*\d+", line):
            continue
        is_heading = (
            re.match(r"^[一二三四五六七八九十]+[、.)）]", line)
            or re.match(r"^第[一二三四五六七八九十\d]+[章节部分]", line)
            or re.match(r"^[0-9]+[、.)）]", line)
        )
        if is_heading:
            return clean_chapter_title(line)
        if len(line) > 36:
            continue
        if not re.search(r"[\u4e00-\u9fffA-Za-z]", line):
            continue
        cleaned = clean_chapter_title(line)
        if cleaned != "未分类":
            lines.append(cleaned)

    return (lines[0][:12] if lines else "未分类")


def local_extract_chunk_concepts(text, limit=4):
    import re

    text = text or ""
    stopwords = {
        "公司", "本公司", "报告", "年度报告", "情况", "说明", "相关", "主要", "进行",
        "以及", "根据", "包括", "如下", "合计", "单位", "人民币", "项目", "内容",
        "资料", "问题", "信息", "数据", "分析", "page"
    }
    candidates = []
    normalized_text = re.sub(r"\[page\s+\d+\]", " ", text or "", flags=re.IGNORECASE)

    priority_terms = []

    def add_priority(term):
        term = re.sub(r"\s+", "", (term or "").strip())
        if not term:
            return
        if len(term) > 16:
            return
        if term in stopwords or term.lower() in stopwords:
            return
        if is_bad_concept(term):
            return
        if term not in priority_terms:
            priority_terms.append(term)
            candidates.append(term)

    for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)+\b|\b[A-Z]{2,}[A-Za-z0-9]*\b", normalized_text):
        add_priority(match.group(0))
    for match in re.finditer(r"\b([A-Z]{2,}[A-Za-z0-9]*)\s*([\u4e00-\u9fff]{2,8})", normalized_text):
        add_priority(match.group(1) + match.group(2))
    for match in re.finditer(r"[“\"']([^“”\"'\n]{2,12})[”\"']", normalized_text):
        phrase = match.group(1).strip()
        if re.search(r"[\u4e00-\u9fffA-Za-z]", phrase):
            add_priority(phrase)

    for token in re.findall(r"[A-Za-z][A-Za-z0-9._/-]{2,}|[\u4e00-\u9fff]{2,12}", normalized_text):
        token = token.strip()
        if is_bad_concept(token):
            continue
        if token.lower() in stopwords or token in stopwords:
            continue
        if re.fullmatch(r"\d+", token):
            continue
        if len(token) > 12:
            continue
        candidates.append(token)

    for phrase in re.findall(r"[\u4e00-\u9fffA-Za-z0-9._/-]{2,24}", normalized_text):
        for sep in ["、", "，", ",", "；", ";", "：", ":", "（", "）", "(", ")"]:
            phrase = phrase.replace(sep, " ")
        for token in phrase.split():
            token = token.strip()
            if (
                2 <= len(token) <= 12
                and token not in stopwords
                and token.lower() not in stopwords
                and not is_bad_concept(token)
            ):
                candidates.append(token)

    scored = []
    priority_set = set(priority_terms)
    for token in set(candidates):
        freq = candidates.count(token)
        score = freq * 2 + min(len(token), 8) / 8
        if token in priority_set:
            score += 8 + min(len(token), 12)
        scored.append((score, token))
    scored.sort(key=lambda item: item[0], reverse=True)

    concepts = []
    for _, token in scored:
        if is_bad_concept(token):
            continue
        if any(token in existing or existing in token for existing in concepts):
            continue
        concepts.append(token)
        if len(concepts) >= limit:
            break
    return concepts


def merge_recalled(*recall_lists):
    merged = {}
    for hits in recall_lists:
        for chunk_id, score in hits:
            if chunk_id in chunks_index:
                merged[chunk_id] = max(merged.get(chunk_id, 0.0), score)
    return sorted(merged.items(), key=lambda x: x[1], reverse=True)


def multi_source_recall(query, query_concepts, doc_scope=None, query_embedding=None, topic_scope=None):
    doc_id = resolve_doc_scope(doc_scope)
    strict_allowed = (topic_scope or {}).get("allowed_chunk_ids") if (topic_scope or {}).get("strict") else None
    vector_hits = vector_recall(
        query,
        doc_scope=doc_scope,
        top_k=VECTOR_RECALL_K,
        query_embedding=query_embedding,
        allowed_chunk_ids=strict_allowed
    )

    expanded_concepts = expand_query_concepts(query_concepts, hops=1, limit=10)
    expanded_query = " ".join([query] + expanded_concepts[:8])
    expanded_vector_hits = []
    if expanded_query.strip() != query.strip():
        expanded_vector_hits = vector_recall(
            expanded_query,
            doc_scope=doc_scope,
            top_k=max(VECTOR_RECALL_K // 2, 10),
            allowed_chunk_ids=(topic_scope or {}).get("allowed_chunk_ids")
        )

    tree_hits = filter_recall_by_topic(tree_recall(query_concepts), topic_scope) if ENABLE_TREE_RECALL else []
    graph_hits = filter_recall_by_topic(graph_recall(query_concepts), topic_scope)
    term_hits = filter_recall_by_topic(term_recall(query, query_concepts), topic_scope)
    merged = merge_recalled(vector_hits, expanded_vector_hits, tree_hits, graph_hits, term_hits)
    if doc_id:
        merged = [
            (chunk_id, score)
            for chunk_id, score in merged
            if chunks_index.get(chunk_id, {}).get("doc_id") == doc_id
        ]
    return merged


def keyword_raw_score(query, query_concepts, record):
    text = record.get("clean_text") or record.get("text", "")
    terms = [concept for concept in query_concepts if concept]
    terms.extend(extract_query_terms(query, limit=12))

    deduped_terms = []
    for term in terms:
        if term and term not in deduped_terms:
            deduped_terms.append(term)

    if not deduped_terms:
        return 1.0 if query and query in text else 0.0
    hits = sum(1 for term in deduped_terms if term in text)
    return hits / min(len(deduped_terms), 8)


def graph_raw_score(query_concepts, record):
    chunk_concepts = record.get("concepts", [])
    if not query_concepts or not chunk_concepts:
        return 0.0

    score = 0.0
    for concept in query_concepts:
        if concept in chunk_concepts:
            score += 1.0
        neighbors = knowledge_graph.get(concept, {})
        for chunk_concept in chunk_concepts:
            score += float(neighbors.get(chunk_concept, 0))
    return score

def local_extract_query_concepts(query, limit=8):
    generic_concepts = {
        "公司", "内容", "信息", "数据", "问题", "方法", "结果", "分析"
    }
    scored = []
    query_chars = set(query)
    for concept, neighbors in knowledge_graph.items():
        if not concept:
            continue
        if concept in generic_concepts and concept not in query:
            continue
        if concept in query:
            score = 100 + len(concept) + len(neighbors)
        else:
            if len(concept) <= 2:
                continue
            overlap = len(query_chars & set(concept))
            if overlap < 3:
                continue
            score = overlap / max(len(set(concept)), 1)
            if score < 0.60:
                continue
            score += min(len(neighbors), 20) / 100
        scored.append((score, concept))

    scored.sort(key=lambda item: item[0], reverse=True)
    concepts = []
    for _, concept in scored:
        if not any(concept in existing or existing in concept for existing in concepts):
            concepts.append(concept)
        if len(concepts) >= limit:
            break
    return concepts


def get_query_concepts_safe(query):
    local_concepts = local_extract_query_concepts(query)
    if not USE_LLM_QUERY_CONCEPTS:
        return local_concepts

    try:
        llm_concepts = llm_extract_concepts_query(query)
    except Exception as e:
        print(f"⚠️ 查询概念抽取失败，使用本地图概念匹配：{e}")
        return local_concepts

    merged = []
    for concept in llm_concepts + local_concepts:
        if concept and concept not in merged:
            merged.append(concept)
    return merged[:8]

def rerank_candidates(query, recalled, mode=RETRIEVAL_MODE, query_concepts=None, query_embedding=None, topic_scope=None):
    if not recalled:
        return [], []

    mode = mode or RETRIEVAL_MODE
    use_graph = mode == "embedding_graph"
    use_keyword = mode in ["embedding_graph", "embedding_keyword"]
    if query_concepts is None:
        query_concepts = get_query_concepts_safe(query) if (use_graph or use_keyword) else []

    query_embedding = query_embedding or embed_text(query)
    semantic_raw = []
    for chunk_id, _ in recalled:
        record = chunks_index[chunk_id]
        semantic_raw.append((chunk_id, cosine_similarity(query_embedding, record.get("embedding"))))

    embedding_norm = dict(minmax_normalize(semantic_raw))
    graph_raw = []
    keyword_raw = []

    for chunk_id, _ in recalled:
        record = chunks_index[chunk_id]
        graph_raw.append((chunk_id, graph_raw_score(query_concepts, record)))
        keyword_raw.append((chunk_id, keyword_raw_score(query, query_concepts, record)))

    graph_norm = dict(minmax_normalize(graph_raw))
    keyword_norm = dict(minmax_normalize(keyword_raw))
    ranked = []

    semantic_score_by_chunk = dict(semantic_raw)
    topic_scores = (topic_scope or {}).get("topic_scores", {})
    topic_min = min(topic_scores.values()) if topic_scores else 0.0
    topic_max = max(topic_scores.values()) if topic_scores else 0.0
    for chunk_id, _ in recalled:
        record = chunks_index[chunk_id]
        emb_score = embedding_norm.get(chunk_id, 0.0)
        graph_score = graph_norm.get(chunk_id, 0.0)
        keyword_score = keyword_norm.get(chunk_id, 0.0)

        if mode == "embedding":
            final_score = emb_score
        elif mode == "embedding_keyword":
            final_score = 0.90 * emb_score + 0.10 * keyword_score
        else:
            total_weight = EMBEDDING_WEIGHT + GRAPH_WEIGHT + KEYWORD_WEIGHT
            final_score = (
                EMBEDDING_WEIGHT * emb_score +
                GRAPH_WEIGHT * graph_score +
                KEYWORD_WEIGHT * keyword_score
            ) / total_weight

        topic_norm = 0.0
        if topic_scores:
            record_topic_id = record.get("topic_id")
            raw_topic_score = topic_scores.get(record_topic_id)
            if raw_topic_score is not None:
                if topic_max == topic_min:
                    topic_norm = 1.0
                else:
                    topic_norm = (raw_topic_score - topic_min) / (topic_max - topic_min)
            final_score = (1 - TOPIC_RERANK_WEIGHT) * final_score + TOPIC_RERANK_WEIGHT * topic_norm

        exact_evidence_score = exact_evidence_raw_score(query, record)
        if exact_evidence_score:
            evidence_floor = min(0.92, 0.55 + EXACT_EVIDENCE_BOOST * exact_evidence_score)
            final_score = max(final_score, evidence_floor)

        ranked.append({
            "chunk_id": chunk_id,
            "final_score": final_score,
            "embedding_score": semantic_score_by_chunk.get(chunk_id, 0.0),
            "embedding_norm": emb_score,
            "graph_norm": graph_score,
            "keyword_norm": keyword_score,
            "topic_norm": topic_norm,
            "exact_evidence_norm": exact_evidence_score,
            "record": record
        })

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked, query_concepts

def dynamic_select_ranked(ranked, query):
    if not ranked:
        return []

    complexity = query.count("？") + query.count("?")
    complexity += sum(
        1 for marker in ["1)", "2)", "3)", "至少", "分别", "哪些", "和", "及"]
        if marker in query
    )
    min_k = min(DYNAMIC_MAX_TOP_K, max(DYNAMIC_MIN_TOP_K, 4 if complexity >= 2 else DYNAMIC_MIN_TOP_K))
    max_k = min(DYNAMIC_MAX_TOP_K, max(FINAL_TOP_K, min_k))

    selected = []
    top_score = ranked[0]["final_score"] or 1.0
    previous_score = top_score

    for index, item in enumerate(ranked[:max_k]):
        score = item["final_score"]
        if index < min_k:
            selected.append(item)
            previous_score = score
            continue

        gap = previous_score - score
        relative = score / top_score if top_score else 0.0
        if gap > 0.18 and relative < 0.72:
            break
        if relative < 0.50:
            break
        selected.append(item)
        previous_score = score

    return selected


def add_concept_coverage(selected, ranked, query_concepts, query):
    selected_ids = {item["chunk_id"] for item in selected}
    max_k = DYNAMIC_MAX_TOP_K + 2

    coverage_terms = []
    for term in extract_exact_evidence_terms(query):
        if term and term not in coverage_terms:
            coverage_terms.append(term)
    generic_terms = {"公司", "内容", "信息", "数据", "问题", "方法", "结果", "分析", "亿元", "费用"}
    for term in extract_query_terms(query, limit=20) + list(query_concepts or []):
        if not term or term in generic_terms:
            continue
        if len(term) > 12:
            continue
        if term.isdigit() and len(term) == 4:
            continue
        if term and term not in coverage_terms:
            coverage_terms.append(term)

    for term in coverage_terms:
        if len(selected) >= max_k:
            break
        if any(term in (item["record"].get("clean_text") or item["record"].get("text", "")) or term in item["record"].get("concepts", []) for item in selected):
            continue

        best = None
        for item in ranked:
            if item["chunk_id"] in selected_ids:
                continue
            record = item["record"]
            search_text = record.get("clean_text") or record.get("text", "")
            if term in search_text or term in record.get("concepts", []):
                best = item
                break

        if best is not None:
            selected.append(best)
            selected_ids.add(best["chunk_id"])

    selected.sort(key=lambda item: item["final_score"], reverse=True)
    return selected


CHUNK_ID_PATTERN = re.compile(r"^(?P<doc>.+)::chunk_(?P<num>\d+)$")


def parse_chunk_id(chunk_id):
    match = CHUNK_ID_PATTERN.match(chunk_id or "")
    if not match:
        return None, None
    return match.group("doc"), int(match.group("num"))


def make_context_bridge_item(chunk_id, ranked_by_id):
    if chunk_id in ranked_by_id:
        return dict(ranked_by_id[chunk_id])
    record = chunks_index.get(chunk_id)
    if not record:
        return None
    return {
        "chunk_id": chunk_id,
        "final_score": 0.01,
        "embedding_score": 0.0,
        "embedding_norm": 0.0,
        "graph_norm": 0.0,
        "keyword_norm": 0.0,
        "cross_encoder_norm": 0.0,
        "record": record,
        "context_bridge": True,
    }


def add_contiguous_context_bridges(selected, ranked):
    """Add missing chunks between selected neighbors from the same document.

    Financial tables and dense PDF sections are often split across chunk
    boundaries. If retrieval selects both sides of a short gap, the omitted
    middle chunk is usually evidence continuity rather than noise.
    """
    if len(selected) < 2:
        return selected

    selected_ids = {item["chunk_id"] for item in selected}
    ranked_by_id = {item["chunk_id"]: item for item in ranked}
    additions = []
    max_additions = 2

    by_doc = {}
    for item in selected:
        doc_id, seq = parse_chunk_id(item["chunk_id"])
        if doc_id is None:
            continue
        by_doc.setdefault(doc_id, []).append(seq)

    for doc_id, seqs in by_doc.items():
        seqs = sorted(set(seqs))
        for left, right in zip(seqs, seqs[1:]):
            if len(additions) >= max_additions:
                break
            if right - left != 2:
                continue
            for seq in range(left + 1, right):
                chunk_id = f"{doc_id}::chunk_{seq:06d}"
                if chunk_id in selected_ids:
                    continue
                bridge = make_context_bridge_item(chunk_id, ranked_by_id)
                if bridge is not None:
                    additions.append(bridge)
                    selected_ids.add(chunk_id)

    if additions:
        selected = selected + additions
        selected.sort(key=lambda item: item["final_score"], reverse=True)
    return selected


def query_needs_table_neighbor(query):
    terms = [
        "\u672c\u671f\u6570", "\u4e0a\u671f\u6570", "\u4e0a\u5e74\u540c\u671f", "\u540c\u6bd4",
        "\u53d8\u52a8\u6bd4\u4f8b", "\u53d8\u52a8\u539f\u56e0", "\u539f\u56e0\u8bf4\u660e",
        "\u91d1\u989d", "\u79d1\u76ee", "\u8868", "\u73b0\u91d1\u6d41", "\u51c0\u989d",
    ]
    return any(term in (query or "") for term in terms)


def looks_like_reason_or_table_text(text):
    text = text or ""
    if "\u53d8\u52a8\u539f\u56e0\u8bf4\u660e" in text or "\u539f\u56e0\u8bf4\u660e" in text:
        return True
    if "\u5355\u4f4d\uff1a" in text or "\u9879\u76ee" in text and "\u672c\u671f" in text:
        return True
    number_hits = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?%?", text)
    return len(number_hits) >= 8


def add_adjacent_table_reason_bridges(selected, ranked, query):
    """Add immediate neighbor chunks for table/reason evidence split by chunking.

    This is intentionally conservative: it only fires when the query asks for
    tabular amounts/changes/reasons, and at least one side looks like a dense
    table or official reason explanation.
    """
    if not selected or not query_needs_table_neighbor(query):
        return selected

    selected_ids = {item["chunk_id"] for item in selected}
    ranked_by_id = {item["chunk_id"]: item for item in ranked}
    additions = []
    max_additions = 2

    for item in list(selected):
        if len(additions) >= max_additions:
            break
        doc_id, seq = parse_chunk_id(item["chunk_id"])
        if doc_id is None:
            continue
        current_text = record_search_text(item["record"])
        current_is_bridge_anchor = looks_like_reason_or_table_text(current_text)
        for offset in (-1, 1):
            if len(additions) >= max_additions:
                break
            neighbor_id = f"{doc_id}::chunk_{seq + offset:06d}"
            if neighbor_id in selected_ids:
                continue
            neighbor_record = chunks_index.get(neighbor_id)
            if not neighbor_record:
                continue
            neighbor_text = record_search_text(neighbor_record)
            if not (current_is_bridge_anchor or looks_like_reason_or_table_text(neighbor_text)):
                continue
            bridge = make_context_bridge_item(neighbor_id, ranked_by_id)
            if bridge is None:
                continue
            bridge["context_bridge"] = True
            bridge["table_reason_bridge"] = True
            bridge["final_score"] = max(bridge.get("final_score", 0.0), item.get("final_score", 0.0) * 0.985)
            additions.append(bridge)
            selected_ids.add(neighbor_id)

    if additions:
        selected = selected + additions
        selected.sort(key=lambda item: item["final_score"], reverse=True)
    return selected


def record_search_text(record):
    return "\n".join([
        record.get("clean_text", ""),
        record.get("text", ""),
        " ".join(record.get("concepts", [])),
        record.get("chapter", ""),
    ])


def trim_selected_for_context(selected, query, max_chunks=9):
    if len(selected) <= max_chunks:
        return selected

    generic_terms = {
        "公司", "业务", "分析", "层面", "风险", "原因", "影响", "核心", "竞争力",
        "资源", "壁垒", "长期", "供应", "前沿", "储备", "暴露", "项目", "技术",
        "情况", "资料", "报告", "问题", "回答", "为什么", "哪些", "如何",
    }
    terms = [
        term for term in extract_query_terms(query, limit=24)
        if 2 <= len(term) <= 10 and term not in generic_terms and not term.isdigit()
    ]
    if not terms:
        return selected[:max_chunks]

    term_counts = {}
    for term in terms:
        count = sum(1 for item in selected if term in record_search_text(item["record"]))
        if count >= 3:
            term_counts[term] = count

    if not term_counts:
        return selected[:max_chunks]

    anchors = sorted(term_counts, key=lambda term: (-term_counts[term], -len(term)))[:3]

    anchored = []
    unanchored = []
    for item in selected:
        text = record_search_text(item["record"])
        if any(anchor in text for anchor in anchors):
            anchored.append(item)
        else:
            unanchored.append(item)

    if len(anchored) >= min(5, max_chunks):
        anchored = anchored[:max_chunks]
        anchored.sort(key=lambda item: item["final_score"], reverse=True)
        return anchored

    trimmed = anchored[:max_chunks]
    for item in unanchored:
        if len(trimmed) >= max_chunks:
            break
        trimmed.append(item)
    trimmed.sort(key=lambda item: item["final_score"], reverse=True)
    return trimmed


def cross_encoder_rerank(query, ranked):
    if not ENABLE_CROSS_ENCODER or not ranked:
        return ranked

    head = ranked[:CROSS_ENCODER_TOP_N]
    tail = ranked[CROSS_ENCODER_TOP_N:]
    pairs = [
        (query, (item["record"].get("clean_text") or item["record"].get("text", ""))[:CROSS_ENCODER_TEXT_CHARS])
        for item in head
    ]

    try:
        model = get_cross_encoder_model()
        scores = [float(score) for score in model.predict(pairs)]
    except Exception as e:
        print(f"⚠️ cross-encoder rerank 失败，保留 graph rerank 结果：{e}")
        return ranked

    normalized_scores = dict(minmax_normalize([
        (item["chunk_id"], score) for item, score in zip(head, scores)
    ]))
    heuristic_weight = max(0.0, 1.0 - CROSS_ENCODER_WEIGHT)

    reranked_head = []
    for item, raw_score in zip(head, scores):
        updated = dict(item)
        cross_score = normalized_scores.get(item["chunk_id"], 0.0)
        updated["cross_encoder_score"] = raw_score
        updated["cross_encoder_norm"] = cross_score
        updated["heuristic_score"] = item["final_score"]
        updated["final_score"] = (
            CROSS_ENCODER_WEIGHT * cross_score +
            heuristic_weight * item["final_score"]
        )
        exact_evidence_score = updated.get("exact_evidence_norm", 0.0)
        if exact_evidence_score:
            evidence_floor = min(0.92, 0.55 + EXACT_EVIDENCE_BOOST * exact_evidence_score)
            updated["final_score"] = max(updated["final_score"], evidence_floor)
        reranked_head.append(updated)

    for item in tail:
        item.setdefault("cross_encoder_score", None)
        item.setdefault("cross_encoder_norm", 0.0)
        item.setdefault("heuristic_score", item["final_score"])

    merged = reranked_head + tail
    merged.sort(key=lambda item: item["final_score"], reverse=True)
    return merged


def build_semantic_context(ranked, query_concepts, mode, query):
    selected = dynamic_select_ranked(ranked, query)
    selected = add_concept_coverage(selected, ranked, query_concepts, query)
    selected = add_contiguous_context_bridges(selected, ranked)
    selected = add_adjacent_table_reason_bridges(selected, ranked, query)
    selected = trim_selected_for_context(selected, query)
    content_blocks = []
    relation_lines = []
    seen_relations = set()
    relation_terms = []
    for term in extract_query_terms(query, limit=12):
        if term and not is_bad_concept(term) and term not in relation_terms:
            relation_terms.append(term)
    for term in list(query_concepts or []):
        if term not in (query or ""):
            continue
        if term and not is_bad_concept(term) and term not in relation_terms:
            relation_terms.append(term)

    for item in selected:
        record = item["record"]
        doc = documents.get(record.get("doc_id"), {})
        clean_concepts = [c for c in record.get("concepts", []) if not is_bad_concept(c)]
        concepts = "、".join(clean_concepts[:6])
        header = (
            f"【文档】{doc.get('name', record.get('doc_id'))}\n"
            f"【章节】{record.get('chapter', '未分类')}\n"
            f"【概念】{concepts}\n"
            f"【评分】final={item['final_score']:.3f}, "
            f"emb={item['embedding_norm']:.3f}, graph={item['graph_norm']:.3f}, "
            f"keyword={item['keyword_norm']:.3f}, "
            f"cross={item.get('cross_encoder_norm', 0.0):.3f}"
        )
        content_blocks.append(header + "\n" + record.get("text", ""))

        for concept in relation_terms + clean_concepts[:3]:
            for relation in expand_graph(concept, query_terms=relation_terms, limit=4, min_weight=2):
                if relation not in seen_relations:
                    seen_relations.add(relation)
                    relation_lines.append(relation)

    context = "\n\n".join(content_blocks)
    context += "\n\n【关系】\n" + "\n".join(relation_lines[:8])
    recall_flags = []
    if not ENABLE_TREE_RECALL:
        recall_flags.append("tree-off")
    if ENABLE_TOPIC_ROUTING and topic_index:
        recall_flags.append("topic-on")
    flag_text = f" ({', '.join(recall_flags)})" if recall_flags else ""
    path = f"{mode}召回-top-{len(ranked)} -> 选出-top-{len(selected)}{flag_text}"
    return path, ",".join(item["chunk_id"] for item in selected), context


def retrieve_semantic(query, doc_scope=None, mode=RETRIEVAL_MODE):
    query_concepts = get_query_concepts_safe(query)
    query_embedding = embed_text(query)
    topic_scope = get_topic_scope(query, doc_scope=doc_scope, query_embedding=query_embedding)
    recalled = multi_source_recall(
        query,
        query_concepts,
        doc_scope=doc_scope,
        query_embedding=query_embedding,
        topic_scope=topic_scope
    )
    if not recalled:
        return None, None, "没有找到可用于检索的片段，请先重新导入文档并生成 embedding。"

    ranked, query_concepts = rerank_candidates(
        query,
        recalled,
        mode=mode,
        query_concepts=query_concepts,
        query_embedding=query_embedding,
        topic_scope=topic_scope
    )
    ranked = cross_encoder_rerank(query, ranked)
    path, selected_chunks, context = build_semantic_context(ranked, query_concepts, mode, query)
    return path, selected_chunks, context

def retrieve(query):
    mode = "embedding_graph"
    if not chunks_index:
        raise RuntimeError("当前知识库没有新版 chunks/embedding 索引，请先运行 build_kb.py 重建知识库。")
    return retrieve_semantic(query, doc_scope=None, mode=mode)


def preserve_context_aliases(answer, context, limit=12):
    if not answer or not context:
        return answer
    pairs = []
    pattern = r"([\u4e00-\u9fffA-Za-z0-9·\-]{2,24})[（(]([A-Za-z][A-Za-z0-9._/-]{1,})[）)]"
    for base, alias in re.findall(pattern, context):
        base = re.sub(r"\s+", "", base or "").strip("，。；;:：、")
        alias = (alias or "").strip()
        if not base or not alias or alias in answer:
            continue
        if (base, alias) not in pairs:
            pairs.append((base, alias))
        if len(pairs) >= limit:
            break

    patched = answer
    for base, alias in pairs:
        full = f"{base}（{alias}）"
        if base in patched and full not in patched:
            patched = patched.replace(base, full, 1)
            continue

        chinese_parts = re.findall(r"[\u4e00-\u9fff]{2,}", base)
        if not chinese_parts:
            continue
        tail = chinese_parts[-1]
        for size in range(min(8, len(tail)), 3, -1):
            suffix = tail[-size:]
            if suffix in patched and alias not in patched:
                patched = patched.replace(suffix, f"{suffix}（{alias}）", 1)
                break
    return patched


def extract_priority_evidence_snippets(query, context, limit=10):
    import re

    query_terms = extract_query_terms(query, limit=16)
    text = context or ""
    raw_units = []
    clean_lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if re.sub(r"\s+", " ", line).strip()
    ]
    for idx, line in enumerate(clean_lines):
        if line.startswith("【评分】") or line.startswith("【文档】"):
            continue
        raw_units.append(line[:360])
        if idx + 1 < len(clean_lines):
            raw_units.append((line + " " + clean_lines[idx + 1])[:420])
        if idx + 2 < len(clean_lines):
            raw_units.append((line + " " + clean_lines[idx + 1] + " " + clean_lines[idx + 2])[:480])
    for block in re.split(r"[\n。；;]+", text):
        unit = re.sub(r"\s+", " ", block).strip()
        if len(unit) < 12:
            continue
        if unit.startswith("【评分】") or unit.startswith("【文档】"):
            continue
        raw_units.append(unit[:360])

    signal_words = (
        "原因", "说明", "增长", "同比", "收入", "毛利率", "现金流", "财务费用",
        "协议", "合同", "认证", "基地", "装置", "项目", "合作", "供应链",
        "风险", "进口", "采购", "减值", "摊销", "周期", "壁垒", "储备",
        "第一库", "科学院", "能源", "液氦冷箱", "近百", "ISO", "ASME", "SOD", "Super-N", "Fast-N",
    )
    scored = []
    for unit in raw_units:
        score = 0.0
        score += sum(1 for term in query_terms if term and term in unit) * 4
        score += min(len(re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?%?", unit)), 4) * 1.5
        score += min(len(re.findall(r"\b[A-Z][A-Za-z0-9._/-]{1,}\b", unit)), 4) * 1.2
        score += sum(1.0 for word in signal_words if word in unit)
        if re.search(r"[“\"']([^“”\"'\n]{2,20})[”\"']", unit):
            score += 2.0
        if score > 0:
            scored.append((score, unit))

    snippets = []
    seen = set()
    for _, unit in sorted(scored, key=lambda item: item[0], reverse=True):
        key = unit[:80]
        if key in seen:
            continue
        if any(unit in old or old in unit for old in snippets):
            continue
        seen.add(key)
        snippets.append(unit)
        if len(snippets) >= limit:
            break
    return snippets


def print_ablation_retrieval(query, doc_scope=None):
    if not chunks_index:
        print("没有新版 chunk/embedding 索引，请先清空并重新导入文档。")
        return

    recalled = vector_recall(query, doc_scope=doc_scope, top_k=VECTOR_RECALL_K)
    if not recalled:
        print("没有召回到候选片段。")
        return

    modes = ["embedding", "embedding_keyword", "embedding_graph"]
    for mode in modes:
        ranked, query_concepts = rerank_candidates(query, recalled, mode=mode)
        print(f"\n=== {mode} ===")
        if query_concepts:
            print("查询概念：", "、".join(query_concepts))
        for rank, item in enumerate(ranked[:FINAL_TOP_K], start=1):
            record = item["record"]
            doc = documents.get(record.get("doc_id"), {})
            preview = record.get("text", "")[:80].replace("\n", " ")
            print(
                f"{rank}. final={item['final_score']:.3f} "
                f"emb={item['embedding_norm']:.3f} "
                f"graph={item['graph_norm']:.3f} "
                f"keyword={item['keyword_norm']:.3f}"
            )
            print(f"   文档：{doc.get('name', record.get('doc_id'))}")
            print(f"   章节：{record.get('chapter', '未分类')}")
            print(f"   片段：{item['chunk_id']}")
            print(f"   预览：{preview}...")
def load_pdf(file_path):
    pages = []
    try:
        import fitz
        with fitz.open(file_path) as doc:
            for index, page in enumerate(doc, start=1):
                page_text = page.get_text("text") or ""
                if page_text.strip():
                    pages.append(f"[[PAGE:{index}]]\n{page_text.strip()}")
    except Exception:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        for index, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"[[PAGE:{index}]]\n{page_text.strip()}")
    return "\n\n".join(pages)


def load_text(file_path):
    lower_path = file_path.lower()
    if lower_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    if lower_path.endswith(".docx"):
        from docx import Document
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    if lower_path.endswith(".pdf"):
        return load_pdf(file_path)

    print("⚠️ 不支持的文件格式，请使用 .txt / .pdf / .docx")
    return ""


def split_text(text, chunk_size=None):
    import re

    text = (text or "").strip()
    total = len(text)
    if not text:
        return []

    if chunk_size is None:
        if total < 3000:
            chunk_size = 300
        elif total < 20000:
            chunk_size = 500
        else:
            chunk_size = 800

    def split_plain(segment):
        patterns = [
            r"\n(?=\d+[\.\u3001\uff09)]\s*)",
            r"\n(?=[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+[\u3001\uff09)]\s*)",
            r"\n(?=#{1,6}\s+)",
            r"\n(?=[A-Z][\.\u3001\uff09)]\s*)",
            r"\n\n+",
        ]
        labels = [
            "\u6570\u5b57\u7f16\u53f7",
            "\u4e2d\u6587\u7f16\u53f7",
            "\u6807\u9898",
            "\u5b57\u6bcd\u7f16\u53f7",
            "\u7a7a\u884c",
        ]
        parts = None
        used = "\u5b57\u6570\u5207\u5272"
        for label, pattern in zip(labels, patterns):
            pieces = re.split(pattern, segment)
            if len(pieces) > 2:
                parts = pieces
                used = label
                break
        if not parts:
            parts = [segment[i:i + chunk_size] for i in range(0, len(segment), chunk_size)]
        merged = []
        current = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(current) + len(part) <= chunk_size:
                current = f"{current}\n{part}" if current else part
                continue
            if current:
                merged.append(current)
            if len(part) > chunk_size:
                merged.extend(
                    part[i:i + chunk_size].strip()
                    for i in range(0, len(part), chunk_size)
                    if part[i:i + chunk_size].strip()
                )
                current = ""
            else:
                current = part
        if current:
            merged.append(current)
        return [chunk for chunk in merged if not is_noise(chunk)], used

    page_matches = list(re.finditer(r"\[\[PAGE:(\d+)\]\]", text))
    chunks = []
    used_labels = []
    if page_matches:
        for idx, match in enumerate(page_matches):
            page_no = match.group(1)
            start = match.end()
            end = page_matches[idx + 1].start() if idx + 1 < len(page_matches) else len(text)
            page_text = text[start:end].strip()
            if not page_text:
                continue
            if len(page_text) <= int(chunk_size * 1.25):
                page_chunks = [page_text]
                used = "\u9875\u8fb9\u754c"
            else:
                page_chunks, used = split_plain(page_text)
            used_labels.append(used)
            for page_chunk in page_chunks:
                chunks.append(f"[page {page_no}]\n{page_chunk}")
    else:
        chunks, used = split_plain(text)
        used_labels.append(used)

    chunks = [chunk for chunk in chunks if chunk and not is_noise(chunk)]
    overlap = min(120, max(60, chunk_size // 6))
    result = []
    for index, chunk in enumerate(chunks):
        if index > 0:
            context = chunks[index - 1][-overlap:] + "\n" + chunk
        else:
            context = chunk
        result.append((chunk, context))

    used_pattern = "+".join(sorted(set(used_labels))) if used_labels else "\u5b57\u6570\u5207\u5272"
    print(f"\n\u2705 \u5207\u5272\u5b8c\u6210\uff08{used_pattern}\uff09\uff0c\u5171 {len(result)} \u4e2achunk\uff0coverlap={overlap}\u5b57")
    print("\u2500" * 40)
    for i, (clean, ctx) in enumerate(result[:3]):
        preview = ctx[:50].replace("\n", " ")
        print(f"chunk{i+1}: \u300c{preview}...\u300d(\u4e0a\u4e0b\u6587{len(ctx)}\u5b57/\u51c0{len(clean)}\u5b57)")
    print("\u2500" * 40)
    return result

def add_document_to_tree(file_path, batch_size=20):
    import json
    global knowledge_tree, knowledge_graph, topic_index, documents, chunks_index, processed_chunks
    
    # 加载已处理进度
    if os.path.exists(KB_PATH):
        with open(KB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            processed_chunks = data.get("processed_chunks", {})
            if data.get("schema_version") == SCHEMA_VERSION:
                documents = documents or data.get("documents", {})
                chunks_index = chunks_index or data.get("chunks", {})
                knowledge_tree = knowledge_tree or data.get("knowledge_tree", {})
                knowledge_graph = knowledge_graph or data.get("knowledge_graph", {})
                topic_index = topic_index or data.get("topic_index", {})
    
    doc_id = make_doc_id(file_path)
    abs_path = os.path.abspath(file_path)
    doc_name = os.path.basename(file_path)
    if knowledge_tree and not chunks_index and not documents:
        print("⚠️ 检测到旧版知识库。当前阶段建议先清空重建，否则新旧结构会混在一起。")

    if doc_id not in documents:
        documents[doc_id] = {
            "id": doc_id,
            "name": doc_name,
            "path": abs_path,
            "chunk_ids": []
        }
    if doc_id not in knowledge_tree:
        knowledge_tree[doc_id] = {}

    text = load_text(file_path)
    chunk_pairs = split_text(text)
    total = len(chunk_pairs)

    # 找到上次处理到哪里
    start_index = processed_chunks.get(doc_id, processed_chunks.get(file_path, 0))

    if start_index >= total:
        print(f"✅ 该文件已全部处理完成（共{total}个片段）")
        return

    end_index = min(start_index + batch_size, total)
    print(f"\n📄 文档共 {total} 个片段，本次处理第 {start_index+1} 到 {end_index} 个\n")

    for i, (clean_chunk, context_chunk) in enumerate(chunk_pairs[start_index:end_index], start=start_index):
        if not clean_chunk.strip():
            continue

        doc_tree = knowledge_tree[doc_id]
        # 默认用本地规则建库，避免每个 chunk 都调用 LLM 导致构建极慢。
        if USE_LLM_BUILD:
            chapter = llm_choose_chapter(context_chunk, existing_chapters=list(doc_tree.keys()))
        else:
            chapter = local_choose_chapter(clean_chunk, existing_chapters=list(doc_tree.keys()))

        if chapter not in doc_tree:
            doc_tree[chapter] = {}

        # 概念提取用净文本，避免 overlap 引入上一 chunk 的干扰。
        if USE_LLM_BUILD:
            concepts = list(llm_extract_concepts(clean_chunk))
            concepts = [concept for concept in concepts if concept]
            if not concepts:
                concepts = local_extract_chunk_concepts(clean_chunk)
        else:
            concepts = local_extract_chunk_concepts(clean_chunk)
        concepts = list(dict.fromkeys(concepts))
        chunk_id = f"{doc_id}::chunk_{i:06d}"
        # embedding 用净文本，保证向量信号纯净
        embedding = embed_text(clean_chunk)
        chunks_index[chunk_id] = {
            "id": chunk_id,
            "doc_id": doc_id,
            "doc_name": doc_name,
            "chapter": chapter,
            "concepts": concepts,
            "clean_text": clean_chunk,
            "text": context_chunk,   # 带 overlap，给 LLM 读
            "embedding": embedding   # 净文本向量，给检索用
        }
        if chunk_id not in documents[doc_id]["chunk_ids"]:
            documents[doc_id]["chunk_ids"].append(chunk_id)

        for concept in concepts:
            if concept not in knowledge_graph:
                knowledge_graph[concept] = {}
            for other_concept in concepts:
                if concept != other_concept:
                    knowledge_graph[concept][other_concept] = \
                        knowledge_graph[concept].get(other_concept, 0) + 1
            if concept not in doc_tree[chapter]:
                doc_tree[chapter][concept] = []
            if chunk_id not in doc_tree[chapter][concept]:
                doc_tree[chapter][concept].append(chunk_id)

        processed_chunks[doc_id] = i + 1
        save_knowledge_base(processed_chunks)
        print(f"片段 {i} → 分类到：{chapter}/ {concepts}")
        time.sleep(1)

    if end_index < total:
        print(f"\n⏸ 本批处理完成，还剩 {total - end_index} 个片段未处理")
        print(f"再次选择'上传资料'并输入同一文件路径，可继续处理下一批")
    else:
        rebuild_topic_index_for_doc(doc_id)
        save_knowledge_base(processed_chunks)
        print(f"\n✅ 文件全部处理完成！")
def llm_choose_chapter(text, existing_chapters=None):
    client = get_llm_client()
    chapter_candidates = existing_chapters if existing_chapters is not None else list(knowledge_tree.keys())

    prompt = f"""
请判断以下文本属于哪个章节/主题分类。

已有分类：
{chapter_candidates}

文本：
{text}

要求：
1. 优先从已有分类中选择，只要内容有关联就应该归入已有分类
2. 只有当文本内容与所有已有分类完全无关时，才创建新分类
3. 新分类名不超过10个字
4. 只返回分类名，不要解释
5. 同一本教材/文档的内容，尽量归入同一个分类
"""

    result = llm_call_with_retry(client, prompt)

    # ===== 清洗（防止模型乱说话）=====
    result = result.replace("：", ":").replace("\n", "").strip()
    for prefix in ["分类:", "章节:", "主题:", "分类为", "章节为", "主题为"]:
        if result.startswith(prefix):
            result = result[len(prefix):].strip()
    result = result.replace("。", "").strip()
    if not result:
        result = local_choose_chapter(text, existing_chapters=chapter_candidates)
    return result[:20] if result else "未分类"
def llm_extract_concepts(text):
    client = get_llm_client()

    prompt = f"""
请从以下文本中提取"核心知识概念"（用于构建知识图谱）。

要求：
1. 只保留该文本所属学科的核心专业术语
2. 必须是该学科的专业术语，不要提取通用词（如：归纳法、结合律、行、列）
3. 不要提取章节标题、目录名（如：第一章）
4. 一律使用中文
5. 去掉变量、符号、乱码
6. 最多返回4个，宁少勿滥

文本：
{text}

只返回概念列表，用中文逗号分隔：
"""

    raw = llm_call_with_retry(client, prompt)

    # ===== 强化清洗（关键）=====
    concepts = raw.replace("，", ",") \
              .replace("\n", ",") \
              .split(",")
    concepts = [c for c in concepts if c.isascii() or any('\u4e00' <= ch <= '\u9fff' for ch in c)]
    # 去空格 + 去空项 + 去太长的垃圾
    cleaned = []
    for c in concepts:
        c = c.strip()

        if not c:
           continue
        if c in ["定义", "性质"]:
           continue
        if any(x in c for x in ["Æ", "Ç"]):
           continue

        cleaned.append(c)

    if not cleaned:
        cleaned = local_extract_chunk_concepts(text)
    return list(dict.fromkeys(cleaned))[:4]

def generate_answer(query, context):
    client = get_llm_client()
    priority_snippets = extract_priority_evidence_snippets(query, context)
    priority_text = "\n".join(f"- {snippet}" for snippet in priority_snippets)
    prompt = f"""你是一个基于资料的问答助手。请只依据给定资料回答用户问题，不要编造资料外信息。

【用户问题】
{query}

【高优先级证据摘录】
{priority_text}

【参考资料】
{context}

请严格遵守：
1. 先给出直接结论，再用资料依据简要解释。
2. 多子问题按 1) 2) 3) 分点回答，不能漏答。
3. 优先保留资料中的关键术语、定义、公式、步骤、条件、结论、编号、缩写、英文简称和必要数字；若资料写成“中文名（缩写/编号）”或同一实体同时出现中文名和缩写，回答中也要同时保留。
4. 如果资料中没有直接证据，明确写“资料未提供”，不要用常识推测。
5. 如果问题询问多个主体、流程、责任分工、能力、措施、证据、要点、风险类型或原因，必须逐一列出与问题限定范围直接相关的内容，不要合并省略。
6. 回答前检查问题中的每个编号、问号和责任主体是否都已覆盖；若资料中有直接主体名称，必须写出。
7. 责任分工题必须区分“制定/统筹/执行/监督/报告”等动作；如果资料写“A通过B负责C”，回答时要把 A 和 B 的角色分开，不要把 B 的执行职责合并成 A 的职责。
8. 涉及表格数字时，必须确认科目名称、单位、本期数、上年同期数、变动比例属于同一行，不要把相邻行数字串错。
9. 涉及合作方、项目、基地、装置、认证、奖项等专名时，必须保留资料中的完整名称；不要用“相关技术”“某项目”等泛化说法替代。
10. 回答每个维度前，先在参考资料中寻找与该维度最直接、最具体的证据；若同时有上位概括和带专名/数字的证据，优先使用带专名/数字的证据。
11. 如果用户问题限定了某个业务、主体、项目或时间范围，每个分点都必须回到该限定范围；不要用其他业务、其他项目或泛化技术材料替代。
12. 风险、原因和战略意义只能列资料明确写出的内容；不要额外扩展“可能研发失败”“可能市场转化失败”等资料外推断。
13. 不要输出“根据提供的资料”这类空话，直接回答。
"""
    prompt += f"""

【必须优先核对的证据摘录（重复提醒）】
{priority_text}
"""
    prompt += """

【最终自检要求】
回答前请再次检查：如果问题要求从多个角度作答，每个角度都必须优先使用资料中最直接、带专名或数字的证据。
不要用同一段落附近但不属于该角度的其他项目替代。尤其是合作方、基地、认证、装置、协议、储备项目、风险原因等，必须保留原文专名。
如果资料中同时出现泛化表述和具体表述，优先写具体表述；例如“与中国科学院合作推进国内首个小分子深地存储项目，打造氦气储备‘中华第一库’”不能改写成其他技术项目。
如果问题包含“前沿储备技术、储备项目、合作项目、战略储备、供应链能力”等维度，必须优先查找并复述包含“合作、科学院、第一库、储备、协议、供应链”等词的直接证据，不得用同一上下文里的其他研发项目、其他基地或其他产品案例替代。
涉及风险时，优先复述资料中的直接风险短语，例如“进口无法持续或进口量不达预期”“采购价格上涨”“合同权益摊销费用增加或出现减值”“市场周期波动导致销售价格下滑”。
"""
    answer = llm_call_with_retry(client, prompt, max_retries=5, temperature=0.0)
    return preserve_context_aliases(answer, context)
if __name__ == "__main__":
    load_knowledge_base()
    while True:
        print("\n请选择操作:")
        print("1. 上传资料")
        print("2. 提问")
        print("3. 查看知识树")
        print("4. 查看文档列表")
        print("9. 备份并清空知识库")
        print("0. 退出")

        choice = input("请输入选项：").strip()

        if choice == "1":
            file_path = input("请输入文件路径：").strip()
            add_document_to_tree(file_path)

        elif choice == "2":
            query = input("请输入你的问题：").strip()
            chapter, sub, context = retrieve(query)
            answer = generate_answer(query, context)

            print("\n【检索路径】")
            if sub:
                print(f"{chapter} -> {sub}")
            else:
                print(f"{chapter}")

            if "【关系】" in context:
                relations_part = context.split("【关系】", 1)[1].strip()
                print("\n【概念关系】")
                print(relations_part if relations_part else "（暂无关系数据）")

            print("\n【回答】")
            print(answer)

        elif choice == "3":
            print("\n当前知识树")
            if documents:
                for doc_id, meta in documents.items():
                    print(f"\n文档：{meta.get('name')} ({doc_id})")
                    for chapter, concepts in knowledge_tree.get(doc_id, {}).items():
                        print(f"  {chapter}:")
                        for concept in concepts:
                            print(f"    - {concept}")
            else:
                for chapter in knowledge_tree:
                    print(f"\n{chapter}:")
                    for sub in knowledge_tree[chapter]:
                        print(f"  - {sub}")

        elif choice == "4":
            print("\n当前文档列表:")
            if not documents:
                print("（暂无新版文档索引）")
            for doc_id, meta in documents.items():
                print(f"- {doc_id}")
                print(f"  名称：{meta.get('name')}")
                print(f"  路径：{meta.get('path')}")
                print(f"  片段数：{len(meta.get('chunk_ids', []))}")

        elif choice == "9":
            confirm = input("这会备份后清空 knowledge_base.json。确认请输入 RESET：").strip()
            if confirm == "RESET":
                reset_knowledge_base(make_backup=True)
            else:
                print("已取消清空操作")

        elif choice == "0":
            break

        else:
            print("输入错误，请重试")
