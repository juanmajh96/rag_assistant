"""End-to-end RAG pipeline orchestrator."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from rag_assistant.config import RAGConfig
from rag_assistant.evaluation.scorer import EvaluationResult, RAGEvaluator
from rag_assistant.fallback.handler import FallbackDecision, FallbackHandler
from rag_assistant.generation.generator import AnswerGenerator, GenerationError
from rag_assistant.ingestion.chunker import DocumentChunker
from rag_assistant.ingestion.embedder import EmbeddingManager
from rag_assistant.ingestion.loader import DocumentLoader
from rag_assistant.retrieval.retriever import ContextRetriever

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    answer: str
    retrieved_chunks: list[Document]
    retrieval_scores: list[float]
    evaluation: EvaluationResult | None
    fallback_triggered: bool
    fallback_reason: str
    model_used: str

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "retrieved_chunks": [
                {
                    "content": d.page_content[:300],
                    "source": d.metadata.get("source", ""),
                }
                for d in self.retrieved_chunks
            ],
            "retrieval_scores": self.retrieval_scores,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "fallback_triggered": self.fallback_triggered,
            "fallback_reason": self.fallback_reason,
            "model_used": self.model_used,
        }


class RAGPipeline:
    """Assembles all components and runs the full RAG pipeline."""

    def __init__(self, config: RAGConfig | None = None) -> None:
        self._config = config or RAGConfig()
        cfg = self._config

        self._loader = DocumentLoader(cfg.ingestion)
        self._chunker = DocumentChunker(cfg.ingestion)
        self._embedder = EmbeddingManager(cfg.embedding, cfg.vector_store)
        self._retriever = ContextRetriever(
            self._embedder.get_vectorstore(), cfg.retrieval
        )
        self._generator = AnswerGenerator(cfg.generation)
        self._evaluator = RAGEvaluator(cfg.evaluation)
        self._fallback = FallbackHandler(cfg.fallback, cfg.retrieval)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, path: str | Path) -> int:
        """Load, chunk, and embed documents from *path* (file or directory).

        Returns the number of chunks stored.
        """
        path = Path(path)
        if path.is_dir():
            docs = self._loader.load_directory(path)
        else:
            docs = self._loader.load(path)

        chunks = self._chunker.split(docs)
        if not chunks:
            logger.warning("No supported content found at '%s' — nothing ingested.", path)
            return 0
        self._embedder.add_documents(chunks)
        logger.info("Ingestion complete: %d chunk(s) stored.", len(chunks))
        return len(chunks)

    def query(self, question: str) -> RAGResponse:
        """Run the full RAG pipeline for *question*.

        Pipeline steps:
          1. OOS check — redirect immediately if out-of-scope.
          2. Retrieve top-k chunks — abstain if no relevant context.
          3. Generate answer (generator handles retry + model downgrade).
          4. Evaluate answer on four metrics.
          5. If low score on first attempt — re-retrieve with expanded query
             and re-generate once; use the better of the two answers.
          6. Return RAGResponse.
        """
        cfg = self._config

        # ---- Step 1: OOS check ----------------------------------------
        oos_decision = self._fallback.check_oos(
            question,
            self._generator,
            cfg.generation,
            cfg.fallback,
        )
        if oos_decision.triggered:
            return self._abstain_response(
                cfg.fallback.abstain_message, "oos", reason_detail=oos_decision
            )

        # ---- Step 2: Retrieval -----------------------------------------
        results = self._retriever.retrieve(question, cfg.retrieval.top_k)
        no_ctx_decision = self._fallback.check_no_context(
            results, cfg.fallback, cfg.retrieval
        )
        if no_ctx_decision.triggered:
            return self._abstain_response(
                cfg.fallback.abstain_message, "no_context"
            )

        chunks = [doc for doc, _ in results]
        scores = [score for _, score in results]

        # ---- Step 3: Generation ----------------------------------------
        try:
            answer, model_used = self._generator.generate(
                question, chunks, cfg.generation
            )
        except GenerationError as exc:
            fallback_dec = FallbackHandler.check_api_error(exc)
            return self._abstain_response(
                cfg.fallback.abstain_message, fallback_dec.reason
            )

        # ---- Step 4: Evaluation ----------------------------------------
        eval_result = self._evaluator.evaluate(
            question, answer, chunks, cfg.evaluation
        )

        # ---- Step 5: Low-score retry -----------------------------------
        low_score_decision = self._fallback.check_low_score(
            eval_result, question, is_retry=False
        )
        if low_score_decision.triggered and low_score_decision.action == "requery":
            expanded_q = low_score_decision.modified_query or question
            logger.info("Re-querying with expanded query: %r", expanded_q[:100])

            retry_results = self._retriever.retrieve(expanded_q, cfg.retrieval.top_k)
            if self._retriever.has_relevant_context(
                retry_results, cfg.retrieval.similarity_threshold
            ):
                retry_chunks = [doc for doc, _ in retry_results]
                retry_scores = [score for _, score in retry_results]
                try:
                    retry_answer, retry_model = self._generator.generate(
                        expanded_q, retry_chunks, cfg.generation
                    )
                    retry_eval = self._evaluator.evaluate(
                        question, retry_answer, retry_chunks, cfg.evaluation
                    )
                    # Keep whichever attempt scored higher
                    if retry_eval.weighted_score > eval_result.weighted_score:
                        logger.info(
                            "Retry produced better answer (%.3f > %.3f) — using retry.",
                            retry_eval.weighted_score,
                            eval_result.weighted_score,
                        )
                        answer, model_used = retry_answer, retry_model
                        eval_result = retry_eval
                        chunks, scores = retry_chunks, retry_scores
                except GenerationError:
                    logger.warning("Retry generation failed; keeping first attempt.")

            # If score still below threshold after retry → abstain
            retry_low_score = self._fallback.check_low_score(
                eval_result, question, is_retry=True
            )
            if retry_low_score.triggered and retry_low_score.action == "abstain":
                return RAGResponse(
                    answer=cfg.fallback.abstain_message,
                    retrieved_chunks=chunks,
                    retrieval_scores=scores,
                    evaluation=eval_result,
                    fallback_triggered=True,
                    fallback_reason="low_score",
                    model_used=model_used,
                )

        # ---- Step 6: Return --------------------------------------------
        return RAGResponse(
            answer=answer,
            retrieved_chunks=chunks,
            retrieval_scores=scores,
            evaluation=eval_result,
            fallback_triggered=low_score_decision.triggered,
            fallback_reason=low_score_decision.reason if low_score_decision.triggered else "",
            model_used=model_used,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _abstain_response(
        self,
        message: str,
        reason: str,
        reason_detail: FallbackDecision | None = None,
    ) -> RAGResponse:
        return RAGResponse(
            answer=message,
            retrieved_chunks=[],
            retrieval_scores=[],
            evaluation=None,
            fallback_triggered=True,
            fallback_reason=reason,
            model_used="none",
        )
