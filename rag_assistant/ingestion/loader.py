"""Document loading from various file formats."""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import (
    BSHTMLLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

from rag_assistant.config import IngestionConfig

logger = logging.getLogger(__name__)

_LOADER_MAP = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,
    ".html": BSHTMLLoader,
    ".docx": Docx2txtLoader,
}


class DocumentLoader:
    """Loads documents from disk using extension-appropriate LangChain loaders."""

    def __init__(self, config: IngestionConfig | None = None) -> None:
        self._config = config or IngestionConfig()

    def load(self, path: str | Path) -> list[Document]:
        """Load a single file and return its LangChain Documents."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in self._config.supported_extensions:
            raise ValueError(
                f"Unsupported extension '{ext}'. "
                f"Supported: {self._config.supported_extensions}"
            )

        loader_cls = _LOADER_MAP[ext]
        try:
            loader = loader_cls(str(path))
            docs = loader.load()
            logger.info("Loaded %d document(s) from %s", len(docs), path)
            return docs
        except Exception as exc:
            logger.error("Failed to load %s: %s", path, exc)
            raise

    def load_directory(self, directory: str | Path) -> list[Document]:
        """Recursively load all supported files from *directory*."""
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        all_docs: list[Document] = []
        for ext in self._config.supported_extensions:
            for file_path in sorted(directory.rglob(f"*{ext}")):
                try:
                    docs = self.load(file_path)
                    all_docs.extend(docs)
                except Exception as exc:
                    logger.warning("Skipping %s — %s", file_path, exc)

        logger.info(
            "Loaded %d total document(s) from directory %s", len(all_docs), directory
        )
        return all_docs
