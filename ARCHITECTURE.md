# RAG Assistant — Architecture

## 1. Overview

RAG Assistant is a production-quality Retrieval-Augmented Generation (RAG) system
built in Python. It ingests arbitrary documents (text, PDF, Markdown, HTML), stores
their embeddings in a local ChromaDB vector store, and answers natural-language
questions by retrieving the most relevant context and generating grounded answers
with Anthropic Claude models.

Target use cases include internal knowledge-base Q&A, document-focused chatbots, and
research assistants where answers must be traceable to source material.

---

## 2. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                            RAGPipeline                              │
│                                                                     │
│  User Query                                                         │
│      │                                                              │
│      ▼                                                              │
│  ┌────────────┐   OOS?   ┌──────────────────┐                      │
│  │ Fallback   │◄─────────│  AnswerGenerator  │ (OOS detect)         │
│  │ Handler    │          └──────────────────┘                      │
│  └─────┬──────┘                                                     │
│  no OOS│                                                            │
│        ▼                                                            │
│  ┌─────────────────┐                                                │
│  │ ContextRetriever│──► ChromaDB ──► HuggingFace Embeddings        │
│  └────────┬────────┘                                                │
│  no ctx?  │ chunks + scores                                         │
│  ◄────────┘                                                         │
│        ▼                                                            │
│  ┌──────────────────┐                                               │
│  │  AnswerGenerator │──► Anthropic Claude (primary / fallback)     │
│  └────────┬─────────┘                                               │
│           │ answer                                                  │
│           ▼                                                         │
│  ┌──────────────────┐                                               │
│  │   RAGEvaluator   │──► Claude Haiku (4 parallel judge calls)     │
│  └────────┬─────────┘                                               │
│           │ EvaluationResult                                        │
│           ▼                                                         │
│  ┌──────────────────┐                                               │
│  │  FallbackHandler │  (low-score → re-query once)                 │
│  └────────┬─────────┘                                               │
│           ▼                                                         │
│       RAGResponse                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Responsibilities

### `config.py`
Houses all configuration as typed Python `dataclass` instances. Every tunable
knob — chunk size, embedding model, similarity thresholds, Claude model names,
evaluation weights — is defined here. No other module contains hard-coded values.

### `ingestion/loader.py` — `DocumentLoader`
Dispatches file loading to the appropriate LangChain community loader based on
file extension (`.txt` → `TextLoader`, `.pdf` → `PyPDFLoader`, `.md` →
`UnstructuredMarkdownLoader`, `.html` → `BSHTMLLoader`). Supports single-file
and whole-directory ingestion.

### `ingestion/chunker.py` — `DocumentChunker`
Wraps LangChain's `RecursiveCharacterTextSplitter` to break long documents into
overlapping fixed-size chunks, preserving source metadata on every chunk.

### `ingestion/embedder.py` — `EmbeddingManager`
Owns the `HuggingFaceEmbeddings` model and the `Chroma` vector store client.
Exposes `add_documents()` (embed + persist) and `get_vectorstore()` (used by
the retriever).

### `retrieval/retriever.py` — `ContextRetriever`
Performs cosine-similarity search via `similarity_search_with_relevance_scores`.
Returns `(Document, score)` pairs sorted by descending score and provides
`has_relevant_context()` to gate the no-context fallback.

### `generation/prompts.py`
Defines the three prompt families as plain Python string templates: RAG QA,
OOS classification, and per-metric LLM-as-judge evaluation. All prompt
construction is isolated here.

### `generation/generator.py` — `AnswerGenerator`
Wraps `anthropic.Anthropic()` for answer generation and OOS detection. Implements
exponential-backoff retry on `RateLimitError` / `APIStatusError`, then degrades
to the configured fallback model before raising `GenerationError`.

### `evaluation/scorer.py` — `RAGEvaluator`
Scores answers on four metrics (relevance, faithfulness, retrieval precision,
completeness) using parallel Claude Haiku calls via `ThreadPoolExecutor`. Parses
the JSON judge responses, normalises scores to [0, 1], computes a weighted
aggregate, and marks the result as `passed` or not.

### `fallback/handler.py` — `FallbackHandler`
Encapsulates the four fallback scenarios as discrete, testable methods: OOS
detection, no-context check, low-score re-query, and API-error surfacing. Returns
a `FallbackDecision` dataclass so the pipeline can log, branch, and respond
consistently.

### `pipeline/orchestrator.py` — `RAGPipeline`
The single entry point for application code. `ingest()` chains loader →
chunker → embedder. `query()` executes the full six-step pipeline (OOS check →
retrieval → generation → evaluation → optional re-query → response assembly)
and returns a `RAGResponse` dataclass.

---

## 4. Data Flow — Step-by-Step Query Walk-through

1. **User calls** `pipeline.query("What is X?")`
2. **OOS check**: If a `domain_description` is configured, `AnswerGenerator.detect_oos()`
   classifies the query. An out-of-scope verdict short-circuits the pipeline and returns
   the abstain message immediately.
3. **Retrieval**: `ContextRetriever.retrieve()` queries ChromaDB for the top-k most
   similar chunks. `has_relevant_context()` checks whether any chunk exceeds
   `similarity_threshold`. If not, the pipeline returns the abstain message.
4. **Generation**: `AnswerGenerator.generate()` formats the RAG prompt, calls the
   primary Claude model, and retries with exponential backoff on transient errors.
   After exhausting retries it falls back to the cheaper haiku model. If both fail,
   `GenerationError` is raised and caught by the pipeline.
5. **Evaluation**: `RAGEvaluator.evaluate()` scores the answer in four parallel
   Claude Haiku calls. A weighted average determines whether the answer `passed`.
6. **Low-score fallback**: If `passed == False` on the first attempt, the query is
   expanded (e.g., "Please explain in detail: …"), retrieval and generation are re-run
   once, and the better of the two answers is kept. If the retry still fails evaluation,
   the pipeline abstains.
7. **Response**: A `RAGResponse` is returned containing the answer, chunks, scores,
   evaluation metrics, fallback metadata, and the model name that produced the answer.

---

## 5. Evaluation Design

### Why LLM-as-Judge?
Reference-free LLM evaluation removes the need for human-annotated ground-truth
answers. Claude Haiku is used as the judge for cost efficiency — it is fast, cheap,
and sufficiently capable for meta-evaluation tasks.

### Metric Definitions

| Metric | Question answered |
|---|---|
| **Answer Relevance** | Does the answer address the question? |
| **Context Faithfulness** | Is every claim grounded in the retrieved context? |
| **Retrieval Precision** | Are the retrieved chunks actually relevant? |
| **Answer Completeness** | Does the answer cover all aspects of the question? |

### Parallelism
All four judge calls are submitted to a `ThreadPoolExecutor` simultaneously,
cutting evaluation latency to roughly that of the slowest single call (~1–2 s)
rather than the sum of four sequential calls (~4–8 s).

### Score Aggregation
Each metric is scored 0–10 by the judge and normalised to [0, 1]. The weighted
average uses the `EvaluationConfig.weights` mapping
(`relevance=0.3, faithfulness=0.3, completeness=0.2, retrieval_precision=0.2`).
A `min_acceptable_score` threshold (default 0.6) gates the low-score fallback.

---

## 6. Fallback Design

Fallbacks are checked in priority order inside `RAGPipeline.query()`:

| Priority | Trigger | Action | Reason code |
|---|---|---|---|
| 1 | Query classified as out-of-scope | Return abstain message | `oos` |
| 2 | All retrieved chunks below similarity threshold | Return abstain message | `no_context` |
| 3 | API errors after retries exhausted | Downgrade to haiku model; abstain on total failure | `api_error` |
| 4 | Weighted evaluation score below threshold | Re-query once with expanded query; abstain if still low | `low_score` |

All decisions are returned as `FallbackDecision` dataclasses and surfaced in
`RAGResponse.fallback_reason` for observability.

---

## 7. Configuration Reference

All settings live in `rag_assistant/config.py`.

### `IngestionConfig`
| Field | Default | Description |
|---|---|---|
| `chunk_size` | `512` | Maximum characters per chunk |
| `chunk_overlap` | `64` | Overlap between consecutive chunks |
| `supported_extensions` | `[".txt",".pdf",".md",".html"]` | Accepted file types |

### `EmbeddingConfig`
| Field | Default | Description |
|---|---|---|
| `model_name` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model ID |
| `device` | `cpu` | `"cpu"` or `"cuda"` |
| `normalize_embeddings` | `True` | Enables cosine similarity |

### `VectorStoreConfig`
| Field | Default | Description |
|---|---|---|
| `persist_directory` | `./chroma_db` | ChromaDB storage path |
| `collection_name` | `rag_documents` | ChromaDB collection |

### `RetrievalConfig`
| Field | Default | Description |
|---|---|---|
| `top_k` | `5` | Number of chunks to retrieve |
| `similarity_threshold` | `0.35` | Minimum score for "relevant context" |

### `GenerationConfig`
| Field | Default | Description |
|---|---|---|
| `primary_model` | `claude-sonnet-4-6` | First-choice generation model |
| `fallback_model` | `claude-haiku-4-5-20251001` | Used after retry exhaustion |
| `max_tokens` | `1024` | Maximum tokens in generated answer |
| `max_retries` | `3` | Retry attempts before model downgrade |
| `retry_base_delay` | `1.0` | Base seconds for exponential backoff |

### `EvaluationConfig`
| Field | Default | Description |
|---|---|---|
| `judge_model` | `claude-haiku-4-5-20251001` | Model used for evaluation |
| `min_acceptable_score` | `0.6` | Threshold below which low-score fallback fires |
| `weights` | `{relevance:0.3, faithfulness:0.3, completeness:0.2, retrieval_precision:0.2}` | Metric weights |

### `FallbackConfig`
| Field | Default | Description |
|---|---|---|
| `domain_description` | `""` | Describe the domain; empty = skip OOS check |
| `oos_similarity_threshold` | `0.25` | Reserved for embedding-based OOS (future) |
| `abstain_message` | `"I don't have enough information…"` | Returned on abstain |

---

## 8. Extending the System

### Swap the Vector Store
Replace `EmbeddingManager` with a class that wraps your preferred store (Pinecone,
Weaviate, pgvector, etc.) and exposes the same `add_documents()` / `get_vectorstore()`
interface. Update `RAGPipeline.__init__()` to instantiate the new class.

### Add an Evaluation Metric
1. Add a new template string to `generation/prompts.py` `_METRIC_USER_TEMPLATES`.
2. Add the metric name to `_METRICS` in `evaluation/scorer.py`.
3. Add the metric attribute to `EvaluationResult` and the weight key to
   `EvaluationConfig.weights`.
4. Update `RAGEvaluator.evaluate()` to read and include the new score.

### Change the LLM Provider
Replace `AnswerGenerator._call_model()` with a call to your provider's SDK.
The method signature (`model, system, user_message, max_tokens → str`) is the
only contract that the rest of the system depends on.

### Add a New File Type
1. Import the appropriate LangChain loader in `ingestion/loader.py`.
2. Add the extension → loader mapping to `_LOADER_MAP`.
3. Add the extension to `IngestionConfig.supported_extensions`.
