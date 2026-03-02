"""CLI entry point for the RAG Assistant.

Usage:
    python -m rag_assistant ingest <path_or_dir> [--config config.json]
    python -m rag_assistant query "<question>" [--config config.json] [--verbose]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path


def _load_config(config_path: str | None):
    """Load RAGConfig from a JSON file, or return a default instance."""
    # Import here to avoid loading heavy ML deps at import time
    from rag_assistant.config import (
        EmbeddingConfig,
        EvaluationConfig,
        FallbackConfig,
        GenerationConfig,
        IngestionConfig,
        RAGConfig,
        RetrievalConfig,
        VectorStoreConfig,
    )

    if not config_path:
        return RAGConfig()

    with open(config_path) as f:
        raw: dict = json.load(f)

    def _from_dict(cls, data: dict):
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})

    return RAGConfig(
        ingestion=_from_dict(IngestionConfig, raw.get("ingestion", {})),
        embedding=_from_dict(EmbeddingConfig, raw.get("embedding", {})),
        vector_store=_from_dict(VectorStoreConfig, raw.get("vector_store", {})),
        retrieval=_from_dict(RetrievalConfig, raw.get("retrieval", {})),
        generation=_from_dict(GenerationConfig, raw.get("generation", {})),
        evaluation=_from_dict(EvaluationConfig, raw.get("evaluation", {})),
        fallback=_from_dict(FallbackConfig, raw.get("fallback", {})),
    )


def _check_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Error: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Export it before running:\n"
            "  export ANTHROPIC_API_KEY=your_key_here",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_ingest(args: argparse.Namespace) -> None:
    _check_api_key()
    config = _load_config(args.config)
    from rag_assistant.pipeline.orchestrator import RAGPipeline

    pipeline = RAGPipeline(config)
    path = Path(args.path)
    print(f"Ingesting: {path} …")
    count = pipeline.ingest(path)
    print(f"Done. {count} chunk(s) stored in vector store.")


def cmd_query(args: argparse.Namespace) -> None:
    _check_api_key()
    config = _load_config(args.config)
    from rag_assistant.pipeline.orchestrator import RAGPipeline

    pipeline = RAGPipeline(config)
    print(f"Query: {args.question}\n")
    response = pipeline.query(args.question)

    print("=" * 60)
    print(response.answer)
    print("=" * 60)

    if args.verbose:
        import json as _json

        print("\n--- Full RAGResponse (JSON) ---")
        print(_json.dumps(response.to_dict(), indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag_assistant",
        description="RAG-Based Assistant — ingest documents and query them with Claude.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- ingest sub-command ---
    ingest_parser = subparsers.add_parser(
        "ingest", help="Load, chunk, and embed documents into the vector store."
    )
    ingest_parser.add_argument(
        "path", help="Path to a file or directory containing documents to ingest."
    )
    ingest_parser.add_argument(
        "--config", default=None, help="Path to a JSON config file."
    )

    # --- query sub-command ---
    query_parser = subparsers.add_parser(
        "query", help="Ask a question and retrieve a grounded answer."
    )
    query_parser.add_argument("question", help="The question to answer.")
    query_parser.add_argument(
        "--config", default=None, help="Path to a JSON config file."
    )
    query_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full RAGResponse JSON including evaluation scores.",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)


if __name__ == "__main__":
    main()
