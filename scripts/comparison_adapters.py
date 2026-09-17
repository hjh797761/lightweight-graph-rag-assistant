"""Optional external framework integration; importing this module needs no LlamaIndex."""
from __future__ import annotations


def llamaindex_retriever(chunks, embedder, *, candidate_k, top_k, reranker):
    from llama_index.core import VectorStoreIndex
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.schema import TextNode

    class SharedEmbedding(BaseEmbedding):
        def _get_query_embedding(self, query):
            return embedder.encode_one(query).tolist()

        async def _aget_query_embedding(self, query):
            return self._get_query_embedding(query)

        def _get_text_embedding(self, text):
            return embedder.encode_one(text).tolist()

        def _get_text_embeddings(self, texts):
            return embedder.encode_many(texts, batch_size=32).tolist()

    nodes = [TextNode(id_=chunk.id, text=chunk.clean_text, embedding=chunk.embedding.tolist())
             for chunk in chunks]
    index = VectorStoreIndex(nodes, embed_model=SharedEmbedding(model_name="shared-backend"))
    retriever = index.as_retriever(similarity_top_k=candidate_k if reranker else top_k)
    by_id = {chunk.id: chunk for chunk in chunks}

    def retrieve(query):
        found = retriever.retrieve(query)
        pairs = [(by_id[item.node.node_id], float(item.score)) for item in found]
        if reranker:
            pairs = reranker(query, [chunk for chunk, _ in pairs])
        return pairs[:top_k]

    return retrieve, index
