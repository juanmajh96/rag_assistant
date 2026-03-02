"""Pipeline sub-package: end-to-end RAG orchestration."""
from __future__ import annotations


def __getattr__(name: str):
    if name in ("RAGPipeline", "RAGResponse"):
        from rag_assistant.pipeline.orchestrator import RAGPipeline, RAGResponse
        return {"RAGPipeline": RAGPipeline, "RAGResponse": RAGResponse}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["RAGPipeline", "RAGResponse"]
