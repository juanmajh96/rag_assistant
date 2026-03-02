"""LLM-as-judge evaluation of RAG answers across four metrics."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import anthropic
from langchain_core.documents import Document

from rag_assistant.config import EvaluationConfig
from rag_assistant.generation.prompts import build_evaluation_prompt

logger = logging.getLogger(__name__)

_METRICS = ("relevance", "faithfulness", "retrieval_precision", "completeness")


@dataclass
class EvaluationResult:
    answer_relevance: float
    context_faithfulness: float
    retrieval_precision: float
    answer_completeness: float
    weighted_score: float
    explanations: dict[str, str] = field(default_factory=dict)
    passed: bool = False

    def to_dict(self) -> dict:
        return {
            "answer_relevance": self.answer_relevance,
            "context_faithfulness": self.context_faithfulness,
            "retrieval_precision": self.retrieval_precision,
            "answer_completeness": self.answer_completeness,
            "weighted_score": self.weighted_score,
            "explanations": self.explanations,
            "passed": self.passed,
        }


class RAGEvaluator:
    """Scores RAG answers using parallel LLM-as-judge calls."""

    # Map internal metric names → EvaluationResult attribute names
    _ATTR_MAP = {
        "relevance": "answer_relevance",
        "faithfulness": "context_faithfulness",
        "retrieval_precision": "retrieval_precision",
        "completeness": "answer_completeness",
    }
    # Map internal metric names → EvaluationConfig.weights keys
    _WEIGHT_KEY_MAP = {
        "relevance": "relevance",
        "faithfulness": "faithfulness",
        "retrieval_precision": "retrieval_precision",
        "completeness": "completeness",
    }

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self._config = config or EvaluationConfig()
        self._client = anthropic.Anthropic()

    def evaluate(
        self,
        query: str,
        answer: str,
        chunks: list[Document],
        config: EvaluationConfig | None = None,
    ) -> EvaluationResult:
        """Score *answer* on four metrics in parallel; return an EvaluationResult."""
        cfg = config or self._config
        context = self._format_context(chunks)

        scores: dict[str, float] = {}
        explanations: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    self._score_metric, metric, query, answer, context, cfg
                ): metric
                for metric in _METRICS
            }
            for future in as_completed(futures):
                metric = futures[future]
                try:
                    score, explanation = future.result()
                    scores[metric] = score
                    explanations[metric] = explanation
                except Exception as exc:
                    logger.warning(
                        "Metric '%s' evaluation failed (%s); defaulting to 0.0", metric, exc
                    )
                    scores[metric] = 0.0
                    explanations[metric] = f"Evaluation failed: {exc}"

        weights = cfg.weights
        weighted = (
            scores["relevance"] * weights.get("relevance", 0.3)
            + scores["faithfulness"] * weights.get("faithfulness", 0.3)
            + scores["completeness"] * weights.get("completeness", 0.2)
            + scores["retrieval_precision"] * weights.get("retrieval_precision", 0.2)
        )

        result = EvaluationResult(
            answer_relevance=scores["relevance"],
            context_faithfulness=scores["faithfulness"],
            retrieval_precision=scores["retrieval_precision"],
            answer_completeness=scores["completeness"],
            weighted_score=round(weighted, 4),
            explanations=explanations,
            passed=weighted >= cfg.min_acceptable_score,
        )
        logger.info(
            "Evaluation complete — weighted_score=%.3f passed=%s",
            result.weighted_score,
            result.passed,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_metric(
        self,
        metric: str,
        query: str,
        answer: str,
        context: str,
        cfg: EvaluationConfig,
    ) -> tuple[float, str]:
        """Call the judge model for a single metric; return (normalised_score, explanation)."""
        system_prompt, user_message = build_evaluation_prompt(
            metric, query, answer, context
        )
        response = self._client.messages.create(
            model=cfg.judge_model,
            max_tokens=256,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        raw_text = response.content[0].text.strip()
        return self._parse_score(raw_text, metric)

    @staticmethod
    def _parse_score(raw: str, metric: str) -> tuple[float, str]:
        """Parse the JSON response from the judge; return (score_0_to_1, explanation)."""
        try:
            # The judge returns JSON; strip any markdown fences if present
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)
            raw_score = float(data.get("score", 0))
            explanation = data.get("explanation", "")
            normalised = max(0.0, min(1.0, raw_score / 10.0))
            return normalised, explanation
        except Exception as exc:
            logger.warning(
                "Could not parse judge response for metric '%s': %r — %s",
                metric,
                raw[:120],
                exc,
            )
            return 0.0, f"Parse error: {exc}"

    @staticmethod
    def _format_context(chunks: list[Document]) -> str:
        parts = []
        for i, doc in enumerate(chunks, start=1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[Chunk {i} | source: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)
