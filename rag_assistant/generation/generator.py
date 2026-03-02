"""Claude-powered answer generation with retry and model-downgrade fallback."""
from __future__ import annotations

import logging
import time

import anthropic
from langchain_core.documents import Document

from rag_assistant.config import GenerationConfig
from rag_assistant.generation.prompts import build_oos_prompt, build_rag_prompt

logger = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    """Raised when all generation attempts (including fallback model) fail."""


class AnswerGenerator:
    """Wraps the Anthropic SDK to generate answers with resilience built in."""

    def __init__(self, config: GenerationConfig | None = None) -> None:
        self._config = config or GenerationConfig()
        self._client = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        context_chunks: list[Document],
        config: GenerationConfig | None = None,
    ) -> tuple[str, str]:
        """Generate an answer grounded in *context_chunks*.

        Returns:
            (answer_text, model_name_used) — so the pipeline can surface which
            model produced the response (primary vs fallback).

        Raises:
            GenerationError: if the primary model exhausts retries *and* the
                fallback model also fails.
        """
        cfg = config or self._config
        context = self._format_context(context_chunks)
        system_prompt, user_message = build_rag_prompt(query, context)

        # --- Try primary model with exponential-backoff retries ---
        last_exc: Exception | None = None
        for attempt in range(cfg.max_retries):
            try:
                answer = self._call_model(
                    cfg.primary_model, system_prompt, user_message, cfg.max_tokens
                )
                logger.debug(
                    "Primary model '%s' succeeded on attempt %d",
                    cfg.primary_model,
                    attempt + 1,
                )
                return answer, cfg.primary_model
            except (anthropic.RateLimitError, anthropic.APIStatusError) as exc:
                last_exc = exc
                delay = cfg.retry_base_delay * (2**attempt)
                logger.warning(
                    "Generation attempt %d/%d failed (%s). Retrying in %.1fs…",
                    attempt + 1,
                    cfg.max_retries,
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)

        logger.warning(
            "Primary model '%s' exhausted %d retries. Trying fallback model '%s'.",
            cfg.primary_model,
            cfg.max_retries,
            cfg.fallback_model,
        )

        # --- One final attempt on fallback model ---
        try:
            answer = self._call_model(
                cfg.fallback_model, system_prompt, user_message, cfg.max_tokens
            )
            logger.info("Fallback model '%s' succeeded.", cfg.fallback_model)
            return answer, cfg.fallback_model
        except Exception as exc:
            raise GenerationError(
                f"All generation attempts failed. "
                f"Primary error: {last_exc!r}. Fallback error: {exc!r}"
            ) from exc

    def detect_oos(
        self,
        query: str,
        domain_description: str,
        config: GenerationConfig | None = None,
    ) -> bool:
        """Return True if *query* is out-of-scope for *domain_description*.

        Uses the judge (haiku) model for cost efficiency.
        """
        cfg = config or self._config
        system_prompt, user_message = build_oos_prompt(query, domain_description)
        try:
            raw = self._call_model(
                cfg.fallback_model,  # haiku — cheap classifier
                system_prompt,
                user_message,
                max_tokens=10,
            )
            verdict = raw.strip().upper()
            is_oos = verdict == "OUT_OF_SCOPE"
            logger.debug("OOS detection for query %r → %s", query[:60], verdict)
            return is_oos
        except Exception as exc:
            logger.warning("OOS detection failed (%s); treating as in-scope.", exc)
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_context(chunks: list[Document]) -> str:
        """Concatenate chunk page_content with separator lines."""
        parts = []
        for i, doc in enumerate(chunks, start=1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[Chunk {i} | source: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _call_model(
        self,
        model: str,
        system: str,
        user_message: str,
        max_tokens: int,
    ) -> str:
        """Single synchronous call to the Anthropic Messages API."""
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text
