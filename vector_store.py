import os

from config import settings

os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint or "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from loguru import logger
from typing import List, Optional


class STEmbeddings(Embeddings):
    """LangChain-compatible wrapper around SentenceTransformer."""

    def __init__(self, model_name: str, local_files_only: bool = False):
        self._st = SentenceTransformer(model_name, local_files_only=local_files_only)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._st.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._st.encode(text, normalize_embeddings=True).tolist()


class VectorStoreManager:
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        try:
            self.embedding_fn = STEmbeddings(settings.embedding_model)
        except Exception:
            logger.warning("Online load failed, trying offline cache only")
            self.embedding_fn = STEmbeddings(settings.embedding_model, local_files_only=True)
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_fn,
        )
        logger.info(f"Chroma initialized at {self.persist_dir}")

    def add_documents(self, documents: List[Document]) -> List[str]:
        ids = self.vectorstore.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to Chroma")
        return ids

    def search(
        self, query: str, top_k: int = 5, filter: Optional[dict] = None
    ) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=top_k, filter=filter)

    def search_with_score(
        self, query: str, top_k: int = 5, filter: Optional[dict] = None
    ) -> List[tuple]:
        return self.vectorstore.similarity_search_with_relevance_scores(
            query, k=top_k, filter=filter
        )

    def delete_by_filter(self, filter: dict) -> None:
        ids = self.vectorstore.get(where=filter).get("ids", [])
        if ids:
            self.vectorstore.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents matching {filter}")
