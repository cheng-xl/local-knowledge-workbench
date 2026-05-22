import os

from config import settings
from loguru import logger
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings as ChromaSettings
from shared import Document
from typing import List, Optional

os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint or "https://hf-mirror.com")


class VectorStoreManager:
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        try:
            self._model = SentenceTransformer(settings.embedding_model,
                                              local_files_only=False)
        except Exception:
            logger.warning("Online load failed, trying offline cache only")
            self._model = SentenceTransformer(settings.embedding_model,
                                              local_files_only=True)
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(name="langchain")
        logger.info(f"Chroma initialized at {self.persist_dir}")

    def _encode(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

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
