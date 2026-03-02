# RAG Assistant — convenience targets
# Usage:
#   make build          Build the Docker image
#   make ingest         Ingest the bundled sample documents
#   make query Q="..."  Ask a question  (default question if Q is omitted)
#   make clean          Remove the local vector store
#   make help           Show this message

.PHONY: build ingest query query-verbose clean ui help

## Build the Docker image
build:
	docker compose build

## Ingest the bundled sample_docs/ into the vector store
ingest:
	docker compose run --rm rag \
		python -m rag_assistant ingest /app/sample_docs

## Ask a question — set Q="your question" or rely on the default
##   make query Q="What is RAG?"
query:
	docker compose run --rm \
		-e QUERY="$(or $(Q),What is retrieval-augmented generation?)" rag

## Ask a question and print full evaluation scores + retrieval metadata
query-verbose:
	docker compose run --rm \
		-e QUERY="$(or $(Q),What is retrieval-augmented generation?)" rag \
		python -m rag_assistant query "$(or $(Q),What is retrieval-augmented generation?)" --verbose

## Start the Gradio web UI on http://localhost:7860
ui:
	docker compose up ui

## Remove the persisted vector store (forces a fresh ingest next time)
clean:
	rm -rf chroma_db/

help:
	@echo ""
	@echo "  make build            Build the Docker image"
	@echo "  make ingest           Ingest sample_docs/ into the vector store"
	@echo "  make query Q=\"...\"    Ask a question"
	@echo "  make query-verbose Q=\"...\"  Query + print full evaluation JSON"
	@echo "  make clean            Delete the local chroma_db/ vector store"
	@echo "  make ui               Start the Gradio web UI on http://localhost:7860"
	@echo ""
	@echo "  Requires: ANTHROPIC_API_KEY set in .env"
	@echo ""
