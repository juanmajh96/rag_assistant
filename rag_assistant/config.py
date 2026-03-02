"""Typed configuration dataclasses for the RAG Assistant.

All tuneable knobs live here; no hard-coded values elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IngestionConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    supported_extensions: list[str] = field(
        default_factory=lambda: [".txt", ".pdf", ".md", ".html", ".docx"]
    )


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    normalize_embeddings: bool = True


@dataclass
class VectorStoreConfig:
    persist_directory: str = "./chroma_db"
    collection_name: str = "rag_documents"


@dataclass
class RetrievalConfig:
    top_k: int = 5
    similarity_threshold: float = 0.30  # scores below this → "no relevant context"


@dataclass
class GenerationConfig:
    primary_model: str = "claude-sonnet-4-6"
    fallback_model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 1024
    max_retries: int = 3
    retry_base_delay: float = 1.0


@dataclass
class EvaluationConfig:
    judge_model: str = "claude-haiku-4-5-20251001"
    min_acceptable_score: float = 0.6  # below this triggers low-score fallback
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "relevance": 0.3,
            "faithfulness": 0.3,
            "completeness": 0.2,
            "retrieval_precision": 0.2,
        }
    )


@dataclass
class FallbackConfig:
    domain_description: str = ""  # used for OOS detection; empty = no OOS check
    oos_similarity_threshold: float = 0.25
    abstain_message: str = "I don't have enough information to answer that."


@dataclass
class RAGConfig:
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
