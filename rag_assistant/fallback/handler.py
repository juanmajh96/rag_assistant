"""Four-scenario fallback decision logic."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.documents import Document

from rag_assistant.config import FallbackConfig, GenerationConfig, RetrievalConfig

logger = logging.getLogger(__name__)


@dataclass
class FallbackDecision:
    triggered: bool
    reason: str  # "oos" | "no_context" | "low_score" | "api_error" | ""
    action: str  # "abstain" | "requery" | "model_downgrade" | "redirect" | "proceed"
    modified_query: str | None = None


class FallbackHandler:
    """Encapsulates all four fallback scenarios."""

    def __init__(
        self,
        fallback_config: FallbackConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        self._fb_cfg = fallback_config or FallbackConfig()
        self._ret_cfg = retrieval_config or RetrievalConfig()

    # ------------------------------------------------------------------
    # 1. Out-of-scope check
    # ------------------------------------------------------------------

    def check_oos(
        self,
        query: str,
        generator,  # AnswerGenerator — avoid circular import
        generation_config: GenerationConfig | None = None,
        fallback_config: FallbackConfig | None = None,
    ) -> FallbackDecision:
        """Return a 'redirect' decision if query is outside the configured domain.

        If no domain_description is configured this check is skipped.
        """
        cfg = fallback_config or self._fb_cfg
        if not cfg.domain_description:
            return FallbackDecision(triggered=False, reason="", action="proceed")

        is_oos = generator.detect_oos(
            query, cfg.domain_description, generation_config
        )
        if is_oos:
            logger.info("OOS fallback triggered for query: %r", query[:80])
            return FallbackDecision(
                triggered=True,
                reason="oos",
                action="redirect",
            )
        return FallbackDecision(triggered=False, reason="", action="proceed")

    # ------------------------------------------------------------------
    # 2. No-context check
    # ------------------------------------------------------------------

    def check_no_context(
        self,
        retrieval_results: list[tuple[Document, float]],
        fallback_config: FallbackConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> FallbackDecision:
        """Return an 'abstain' decision if no chunk exceeds the similarity threshold."""
        ret_cfg = retrieval_config or self._ret_cfg

        has_context = bool(retrieval_results) and any(
            score >= ret_cfg.similarity_threshold
            for _, score in retrieval_results
        )
        if not has_context:
            logger.info("No-context fallback triggered (threshold=%.2f)", ret_cfg.similarity_threshold)
            return FallbackDecision(
                triggered=True,
                reason="no_context",
                action="abstain",
            )
        return FallbackDecision(triggered=False, reason="", action="proceed")

    # ------------------------------------------------------------------
    # 3. Low-score check
    # ------------------------------------------------------------------

    def check_low_score(
        self,
        eval_result,  # EvaluationResult — avoid circular import
        query: str,
        is_retry: bool = False,
    ) -> FallbackDecision:
        """Return a 'requery' decision on first failure, 'abstain' on second.

        On requery, a modified (expanded) version of the query is provided.
        """
        if eval_result.passed:
            return FallbackDecision(triggered=False, reason="", action="proceed")

        if is_retry:
            logger.info(
                "Low-score fallback (retry): score=%.3f — abstaining.",
                eval_result.weighted_score,
            )
            return FallbackDecision(
                triggered=True,
                reason="low_score",
                action="abstain",
            )

        expanded = f"Please explain in detail: {query}"
        logger.info(
            "Low-score fallback (first attempt): score=%.3f — requerying.",
            eval_result.weighted_score,
        )
        return FallbackDecision(
            triggered=True,
            reason="low_score",
            action="requery",
            modified_query=expanded,
        )

    # ------------------------------------------------------------------
    # 4. API-error check
    # ------------------------------------------------------------------

    @staticmethod
    def check_api_error(exception: Exception) -> FallbackDecision:
        """Surface an API error as a structured FallbackDecision for logging.

        Actual retry/model-downgrade logic lives inside AnswerGenerator;
        this method exists so the pipeline can log and surface the event.
        """
        logger.warning("API error fallback: %s — %s", type(exception).__name__, exception)
        return FallbackDecision(
            triggered=True,
            reason="api_error",
            action="model_downgrade",
        )
