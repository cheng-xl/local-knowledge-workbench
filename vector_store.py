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
        # 保持 "langchain" 名称以兼容现有数据
        self._collection = self._client.get_or_create_collection(name="langchain")
        logger.info(f"Chroma initialized at {self.persist_dir}")

    def _encode(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def add_documents(self, documents: List[Document]) -> List[str]:
        texts = [d.page_content for d in documents]
        embeddings = self._encode(texts)
        ids = [f"doc_{hash(d.page_content) % (10**10)}_{i}"
               for i, d in enumerate(documents)]
        metadatas = [d.metadata for d in documents]
        self._collection.add(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
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
        docs = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [None])[0] or [{}] * len(ids)
        distances = results.get("distances", [[]])[0] or [0.0] * len(ids)
        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc = Document(page_content=documents[i], metadata=dict(meta))
            doc.metadata["score"] = 1.0 / (1.0 + distances[i])
            docs.append(doc)
        return docs

    def search_with_score(self, query: str, top_k: int = 5,
                          filter: Optional[dict] = None) -> List[tuple]:
        docs = self.search(query, top_k=top_k, filter=filter)
        return [(d, d.metadata.get("score", 0.0)) for d in docs]

    def delete_by_filter(self, filter: dict) -> None:
        results = self._collection.get(where=filter)
        if results.get("ids"):
            self._collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} documents matching {filter}")
