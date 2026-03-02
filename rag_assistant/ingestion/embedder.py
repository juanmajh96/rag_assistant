"""HuggingFace embeddings + ChromaDB vector store management."""
from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from rag_assistant.config import EmbeddingConfig, VectorStoreConfig

logger = logging.getLogger(__name__)


def _l2sq_to_similarity(distance: float) -> float:
    """Convert a squared-L2 distance to cosine similarity.

    ChromaDB's L2 metric returns squared L2 distances.  For unit-normalised
    embeddings the relationship is: cosine_sim = 1 - L2² / 2, which sits in
    [-1, 1].  We clamp to [0, 1] so LangChain's range check never fires.
    """
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


class EmbeddingManager:
    """Manages embedding model and ChromaDB vector store lifecycle."""

    def __init__(
        self,
        embedding_config: EmbeddingConfig | None = None,
        vector_store_config: VectorStoreConfig | None = None,
    ) -> None:
        self._emb_cfg = embedding_config or EmbeddingConfig()
        self._vs_cfg = vector_store_config or VectorStoreConfig()

        logger.info(
            "Loading embedding model '%s' on device '%s'",
            self._emb_cfg.model_name,
            self._emb_cfg.device,
        )
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self._emb_cfg.model_name,
            model_kwargs={"device": self._emb_cfg.device},
            encode_kwargs={"normalize_embeddings": self._emb_cfg.normalize_embeddings},
        )

        self._vectorstore = Chroma(
            collection_name=self._vs_cfg.collection_name,
            embedding_function=self._embeddings,
            persist_directory=self._vs_cfg.persist_directory,
            relevance_score_fn=_l2sq_to_similarity,
        )
        logger.info(
            "ChromaDB collection '%s' opened at '%s'",
            self._vs_cfg.collection_name,
            self._vs_cfg.persist_directory,
        )

    def add_documents(self, chunks: list[Document]) -> None:
        """Embed *chunks* and persist them to ChromaDB."""
        if not chunks:
            logger.warning("add_documents called with empty chunk list — skipping")
            return
        self._vectorstore.add_documents(chunks)
        logger.info("Added %d chunk(s) to vector store", len(chunks))

    def get_vectorstore(self) -> Chroma:
        """Return the underlying Chroma instance for use by the retriever."""
        return self._vectorstore
