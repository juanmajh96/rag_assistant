"""Ingestion sub-package: loading, chunking, and embedding documents."""
from __future__ import annotations


def __getattr__(name: str):
    if name == "DocumentLoader":
        from rag_assistant.ingestion.loader import DocumentLoader
        return DocumentLoader
    if name == "DocumentChunker":
        from rag_assistant.ingestion.chunker import DocumentChunker
        return DocumentChunker
    if name == "EmbeddingManager":
        from rag_assistant.ingestion.embedder import EmbeddingManager
        return EmbeddingManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DocumentLoader", "DocumentChunker", "EmbeddingManager"]
