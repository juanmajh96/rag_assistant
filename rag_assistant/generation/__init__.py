"""Generation sub-package: prompt templates and Claude-powered answer generation."""
from __future__ import annotations


def __getattr__(name: str):
    if name in ("AnswerGenerator", "GenerationError"):
        from rag_assistant.generation.generator import AnswerGenerator, GenerationError
        return {"AnswerGenerator": AnswerGenerator, "GenerationError": GenerationError}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AnswerGenerator", "GenerationError"]
