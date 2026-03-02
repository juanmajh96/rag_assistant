"""Fallback sub-package: decision logic for the four fallback scenarios."""
from __future__ import annotations


def __getattr__(name: str):
    if name in ("FallbackHandler", "FallbackDecision"):
        from rag_assistant.fallback.handler import FallbackHandler, FallbackDecision
        return {"FallbackHandler": FallbackHandler, "FallbackDecision": FallbackDecision}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FallbackHandler", "FallbackDecision"]
