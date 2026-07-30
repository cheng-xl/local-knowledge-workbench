import os
import time

from openai import OpenAI
from config import settings
from loguru import logger
import chromadb
from chromadb.config import Settings as ChromaSettings
from shared import Document
from typing import List, Optional


class VectorStoreManager:
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        logger.info(f"Embedding API: {settings.embedding_model} via SiliconFlow")
        self._client = OpenAI(
            api_key=settings.siliconflow_api_key,
            base_url=settings.siliconflow_base_url,
        )
        self._chroma = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._chroma.get_or_create_collection(name="langchain")
        logger.info(f"Chroma initialized at {self.persist_dir}")

    def _encode(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self._client.embeddings.create(
                model=settings.embedding_model,
                input=batch,
            )
            all_embeddings.extend(d.embedding for d in resp.data)
            if i + batch_size < len(texts):
                time.sleep(0.2)  # rate limit guard
        return all_embeddings

    def add_documents(self, documents: List[Document]) -> List[str]:
        texts = [d.page_content for d in documents]
        embeddings = self._encode(texts)
        ids = [f"doc_{hash(d.page_content) % (10**10)}_{i}"
               for i, d in enumerate(documents)]
        self._collection.add(
            ids=ids, embeddings=embeddings, documents=texts,
            metadatas=[d.metadata for d in documents])
        logger.info(f"Added {len(documents)} documents to Chroma")
        return ids

    def search(self, query: str, top_k: int = 5,
               filter: Optional[dict] = None) -> List[Document]:
        query_emb = self._encode([query])
        results = self._collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
            where=filter if filter else None,
        )
        ids = results.get("ids", [[]])[0]
        docs_raw = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [None])[0] or [{}]
        dists = results.get("distances", [[]])[0] or [0.0]
        docs = []
        for i, text in enumerate(docs_raw):
            meta = dict(metas[i]) if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else 0.0
            doc = Document(page_content=text, metadata=meta)
            doc.metadata["score"] = 1.0 / (1.0 + dist)
            docs.append(doc)
        return docs

    # ── Knowledge base management ─────────────────────────

    def get_stats(self) -> dict:
        total = self._collection.count()
        if total == 0:
            return {"total": 0, "sources": []}
        batch = self._collection.get(
            limit=min(total, 5000), include=["metadatas"])
        sources: dict[str, int] = {}
        for m in batch.get("metadatas", []):
            key = m.get("source_file", "unknown")
            sources[key] = sources.get(key, 0) + 1
        return {"total": total, "sources": sorted(
            sources.items(), key=lambda x: x[1], reverse=True)}

    def delete_by_source(self, source_file: str) -> int:
        results = self._collection.get(
            where={"source_file": source_file}, include=[])
        ids = results.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunks from '{source_file}'")
        return len(ids)


# Global singleton — reused across all RAGPipeline instances to avoid
# reloading the 400MB embedding model on every query.
_global_vs: Optional[VectorStoreManager] = None


def _get_global_vs() -> VectorStoreManager:
    global _global_vs
    if _global_vs is None:
        _global_vs = VectorStoreManager()
    return _global_vs
