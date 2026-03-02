"""Retrieval sub-package: semantic search over the vector store."""
from __future__ import annotations


def __getattr__(name: str):
    if name == "ContextRetriever":
        from rag_assistant.retrieval.retriever import ContextRetriever
        return ContextRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ContextRetriever"]
