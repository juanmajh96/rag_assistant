"""Semantic retrieval from the ChromaDB vector store."""
from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag_assistant.config import RetrievalConfig

logger = logging.getLogger(__name__)


class ContextRetriever:
    """Retrieves the most semantically similar chunks for a query."""

    def __init__(
        self,
        vectorstore: Chroma,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._vectorstore = vectorstore
        self._config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Return the top-*k* (Document, similarity_score) pairs, sorted descending.

        Similarity scores are cosine similarities in [0, 1] when embeddings are
        normalised (which is the default).
        """
        k = top_k if top_k is not None else self._config.top_k
        results: list[tuple[Document, float]] = (
            self._vectorstore.similarity_search_with_relevance_scores(query, k=k)
        )
        # Sort descending by score (Chroma may already do this, but be explicit)
        results.sort(key=lambda pair: pair[1], reverse=True)
        logger.debug(
            "Retrieved %d chunk(s) for query: %r (top score: %.3f)",
            len(results),
            query[:80],
            results[0][1] if results else 0.0,
        )
        return results

    def has_relevant_context(
        self,
        results: list[tuple[Document, float]],
        threshold: float | None = None,
    ) -> bool:
        """Return False if *all* retrieved scores fall below *threshold*.

        A False result triggers the "no relevant context" fallback in the pipeline.
        """
        if not results:
            return False
        cutoff = threshold if threshold is not None else self._config.similarity_threshold
        return any(score >= cutoff for _, score in results)
