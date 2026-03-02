"""Evaluation sub-package: LLM-as-judge scoring of RAG answers."""
from __future__ import annotations


def __getattr__(name: str):
    if name in ("RAGEvaluator", "EvaluationResult"):
        from rag_assistant.evaluation.scorer import RAGEvaluator, EvaluationResult
        return {"RAGEvaluator": RAGEvaluator, "EvaluationResult": EvaluationResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["RAGEvaluator", "EvaluationResult"]
