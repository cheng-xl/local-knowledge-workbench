from rag_pipeline import RAGPipeline
from shared import Document


def test_chunk_documents():
    rag = RAGPipeline()
    docs = [Document(page_content="Hello world. " * 200, metadata={"source_file": "test.txt"})]
    chunks = rag.chunk_documents(docs, chunk_size=256, overlap=32)
    assert len(chunks) > 1
    for c in chunks:
        assert "chunk_index" in c.metadata
        assert "chunk_hash" in c.metadata


def test_rrf_fusion():
    docs_a = [Document(page_content=f"doc A {i}", metadata={}) for i in range(5)]
    docs_b = [Document(page_content=f"doc B {i}", metadata={}) for i in range(5)]
    merged = RAGPipeline._rrf(docs_a, docs_b)
    assert len(merged) >= 5
