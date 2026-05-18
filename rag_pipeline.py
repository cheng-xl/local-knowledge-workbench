from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.retrievers import BM25Retriever
from vector_store import VectorStoreManager
from config import settings
from loguru import logger
from typing import List, Optional
import hashlib
import os


class RAGPipeline:
    def __init__(self, persist_dir: Optional[str] = None):
        self.vs = VectorStoreManager(persist_dir)
        self._doc_texts: List[str] = []
        self._bm25: Optional[BM25Retriever] = None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    # ── Document loading ──────────────────────────────────

    _LOADERS = {
        "pdf": PyPDFLoader,
        "docx": Docx2txtLoader,
        "md": UnstructuredMarkdownLoader,
        "txt": TextLoader,
    }

    def load_document(self, file_path: str) -> List[Document]:
        ext = file_path.rsplit(".", 1)[-1].lower()
        loader_cls = self._LOADERS.get(ext, TextLoader)
        logger.info(f"Loading {file_path} with {loader_cls.__name__}")
        docs = loader_cls(file_path).load()
        for d in docs:
            d.metadata.setdefault("source_file", os.path.basename(file_path))
            d.metadata.setdefault("file_type", ext)
        return docs

    # ── Chunking ───────────────────────────────────────────

    def chunk_documents(
        self, docs: List[Document], chunk_size: int = None, overlap: int = None
    ) -> List[Document]:
        cs = chunk_size or settings.chunk_size
        ov = overlap or settings.chunk_overlap
        if cs != self.splitter._chunk_size:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=cs, chunk_overlap=ov,
                separators=["\n\n", "\n", "。", ".", " ", ""],
            )
        chunks = self.splitter.split_documents(docs)
        for i, c in enumerate(chunks):
            c.metadata["chunk_index"] = i
            c.metadata["chunk_hash"] = hashlib.md5(
                c.page_content.encode()
            ).hexdigest()[:8]
        logger.info(f"Split {len(docs)} docs into {len(chunks)} chunks")
        return chunks

    # ── Indexing ───────────────────────────────────────────

    def add_to_store(self, chunks: List[Document]) -> List[str]:
        ids = self.vs.add_documents(chunks)
        self._doc_texts.extend(c.page_content for c in chunks)
        self._bm25 = None
        return ids

    @property
    def bm25(self) -> Optional[BM25Retriever]:
        if self._bm25 is None and self._doc_texts:
            logger.info(f"Building BM25 index on {len(self._doc_texts)} chunks")
            self._bm25 = BM25Retriever.from_texts(
                self._doc_texts,
                preprocess_func=lambda x: x,
            )
        return self._bm25

    # ── Retrieval ──────────────────────────────────────────

    def hybrid_search(
        self, query: str, top_k: int = None, use_rerank: bool = True
    ) -> List[Document]:
        k = top_k or settings.retrieval_top_k

        # Vector search
        vec_results = self.vs.search(query, top_k=k * 2)
        logger.debug(f"Vector search returned {len(vec_results)} results")

        # BM25 search
        bm25_results = []
        if self.bm25 is not None:
            bm25_results = self.bm25.get_relevant_documents(query, k=k * 2)
            logger.debug(f"BM25 returned {len(bm25_results)} results")

        # RRF fusion
        merged = self._rrf(vec_results, bm25_results)

        # Optional rerank
        if use_rerank and len(merged) > k:
            merged = self._rerank(query, merged)

        return merged[:k]

    # ── RRF Fusion ─────────────────────────────────────────

    @staticmethod
    def _rrf(
        vec: List[Document], bm25: List[Document], k: int = 60
    ) -> List[Document]:
        scores: dict[str, tuple[Document, float]] = {}

        for rank, doc in enumerate(vec):
            key = doc.page_content[:200]
            scores[key] = (doc, 1.0 / (k + rank + 1))

        for rank, doc in enumerate(bm25):
            key = doc.page_content[:200]
            rrf = 1.0 / (k + rank + 1)
            if key in scores:
                scores[key] = (scores[key][0], scores[key][1] + rrf)
            else:
                scores[key] = (doc, rrf)

        sorted_items = sorted(scores.values(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items]

    # ── Rerank (optional, requires FlagEmbedding installed) ─

    @staticmethod
    def _rerank(query: str, docs: List[Document]) -> List[Document]:
        try:
            from FlagEmbedding import FlagReranker
            reranker = FlagReranker("BAAI/bge-reranker-base")
            pairs = [[query, d.page_content] for d in docs]
            scores = reranker.compute_score(pairs, normalize=True)
            scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            logger.debug(f"Reranking: top score={scored[0][1]:.3f}")
            return [d for d, _ in scored]
        except ImportError:
            logger.warning("FlagEmbedding not installed, skipping rerank")
            return docs

    # ── Full ingestion pipeline ────────────────────────────

    def ingest_file(self, file_path: str) -> int:
        docs = self.load_document(file_path)
        chunks = self.chunk_documents(docs)
        self.add_to_store(chunks)
        return len(chunks)
