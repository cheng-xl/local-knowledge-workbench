from rag_pipeline import (
    RAGPipeline, RecursiveCharacterTextSplitter, _load_text,
)
from shared import Document


class TestTextSplitter:
    def test_short_text(self):
        s = RecursiveCharacterTextSplitter(chunk_size=512)
        chunks = s.split_text("hello")
        assert len(chunks) == 1
        assert chunks[0] == "hello"

    def test_empty_text(self):
        s = RecursiveCharacterTextSplitter(chunk_size=512)
        chunks = s.split_text("")
        assert len(chunks) <= 1

    def test_paragraph_split(self):
        s = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
        text = "short\n\n" + "x" * 100
        chunks = s.split_text(text)
        assert len(chunks) >= 2

    def test_split_preserves_metadata(self):
        s = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)
        doc = Document(page_content="Hello world. " * 30,
                       metadata={"source_file": "test.txt"})
        chunks = s.split_documents([doc])
        assert len(chunks) > 0
        for c in chunks:
            assert c.metadata["source_file"] == "test.txt"


class TestRAGPipeline:
    def test_chunk_documents(self):
        rag = RAGPipeline()
        docs = [Document(page_content="Hello world. " * 200,
                         metadata={"source_file": "test.txt"})]
        chunks = rag.chunk_documents(docs, chunk_size=256, overlap=32)
        assert len(chunks) > 1
        for c in chunks:
            assert "chunk_index" in c.metadata
            assert "chunk_hash" in c.metadata

    def test_rrf_fusion(self):
        docs_a = [Document(page_content=f"doc A {i}", metadata={})
                  for i in range(5)]
        docs_b = [Document(page_content=f"doc B {i}", metadata={})
                  for i in range(5)]
        merged = RAGPipeline._rrf(docs_a, docs_b)
        assert len(merged) >= 5

    def test_rrf_dedup(self):
        """Overlapping docs should be merged, not duplicated."""
        same = Document(page_content="same content here", metadata={})
        docs_a = [same, Document(page_content="unique A", metadata={})]
        docs_b = [same, Document(page_content="unique B", metadata={})]
        merged = RAGPipeline._rrf(docs_a, docs_b)
        assert len(merged) == 3  # same (merged) + unique A + unique B

    def test_load_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world", encoding="utf-8")
        docs = _load_text(str(f))
        assert len(docs) == 1
        assert "Hello world" in docs[0].page_content
        assert docs[0].metadata["source_file"] == "test.txt"

    def test_tokenize(self):
        tokens = RAGPipeline._tokenize("hello world")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
