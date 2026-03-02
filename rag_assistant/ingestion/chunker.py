"""Recursive character-based document chunking."""
from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_assistant.config import IngestionConfig

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Splits documents into overlapping chunks for embedding."""

    def __init__(self, config: IngestionConfig | None = None) -> None:
        self._config = config or IngestionConfig()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
            length_function=len,
        )

    def split(self, docs: list[Document]) -> list[Document]:
        """Split a list of Documents into smaller chunks.

        Each chunk inherits the metadata from its source document.
        """
        chunks = self._splitter.split_documents(docs)
        logger.info(
            "Split %d document(s) into %d chunk(s) "
            "(chunk_size=%d, overlap=%d)",
            len(docs),
            len(chunks),
            self._config.chunk_size,
            self._config.chunk_overlap,
        )
        return chunks
