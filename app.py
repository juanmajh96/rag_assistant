"""Gradio web UI for the RAG Assistant.

Run directly:
    python app.py

Or via Docker Compose:
    docker compose up ui
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

try:
    import gradio as gr
except ImportError:
    raise SystemExit(
        "gradio is not installed.\n"
        "Install it with:  pip install 'gradio>=4.0.0'"
    )

try:
    from rag_assistant.config import RAGConfig
    from rag_assistant.pipeline.orchestrator import RAGPipeline
except ImportError as exc:
    raise SystemExit(
        f"rag_assistant package not found: {exc}\n"
        "Ensure you are running inside the Docker container or have installed the package."
    )

# Initialise once at startup — loads the embedding model and connects to Chroma.
_pipeline = RAGPipeline(RAGConfig())

_FALLBACK_REASON_MAP: dict[str, str] = {
    "oos": "Warning: Question is outside the scope of the knowledge base.",
    "no_context": "Warning: No relevant context was found in the vector store.",
    "low_score": "Warning: Answer quality fell below the acceptable threshold after retry.",
    "api_error": "Warning: An API error occurred during answer generation.",
}


# ---------------------------------------------------------------------------
# Handler functions
# ---------------------------------------------------------------------------


def handle_query(
    question: str,
) -> tuple[str, str, list[list[str]], list[list[str]]]:
    """Run the full RAG pipeline and return UI-ready outputs.

    Returns:
        answer       — the generated answer text
        status       — "Model: <name>" on success, or a Warning string
        eval_rows    — list of [Metric, Score, Pass/Fail, Explanation] rows
        source_rows  — list of [Source, Similarity Score, Snippet] rows
    """
    if not question or not question.strip():
        return "", "Please enter a question.", [], []

    response = _pipeline.query(question.strip())

    # --- Status line -------------------------------------------------------
    if response.fallback_triggered:
        reason = response.fallback_reason
        status = _FALLBACK_REASON_MAP.get(
            reason, f"Warning: Fallback triggered ({reason})."
        )
    else:
        status = f"Model: {response.model_used}"

    # --- Evaluation rows ---------------------------------------------------
    eval_rows: list[list[str]] = []
    if response.evaluation is not None:
        threshold = _pipeline._config.evaluation.min_acceptable_score
        ev = response.evaluation
        metrics = [
            ("Relevance", ev.answer_relevance, ev.explanations.get("relevance", "")),
            (
                "Faithfulness",
                ev.context_faithfulness,
                ev.explanations.get("faithfulness", ""),
            ),
            (
                "Completeness",
                ev.answer_completeness,
                ev.explanations.get("completeness", ""),
            ),
            (
                "Retrieval Precision",
                ev.retrieval_precision,
                ev.explanations.get("retrieval_precision", ""),
            ),
        ]
        for metric, score, explanation in metrics:
            pass_fail = "Pass" if score >= threshold else "Fail"
            eval_rows.append([metric, f"{score:.2f}", pass_fail, explanation])

    # --- Source rows -------------------------------------------------------
    source_rows: list[list[str]] = []
    for doc, score in zip(response.retrieved_chunks, response.retrieval_scores):
        source = doc.metadata.get("source", "unknown")
        snippet = doc.page_content[:200].replace("\n", " ")
        source_rows.append([source, f"{score:.4f}", snippet])

    return response.answer, status, eval_rows, source_rows


def handle_ingest(uploaded_files: list | None, dir_path: str) -> str:
    """Ingest uploaded files and/or a server-side directory path.

    Uploaded files are copied to a temp directory (preserving their original
    filename so the loader can determine document type by extension), ingested,
    and then cleaned up.
    """
    total = 0
    errors: list[str] = []

    if uploaded_files:
        tmp_dir = tempfile.mkdtemp()
        try:
            for f in uploaded_files:
                dest = Path(tmp_dir) / Path(f.name).name
                shutil.copy(f.name, dest)
            count = _pipeline.ingest(tmp_dir)
            total += count
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Upload ingest error: {exc}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if dir_path and dir_path.strip():
        try:
            count = _pipeline.ingest(dir_path.strip())
            total += count
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Directory ingest error: {exc}")

    if not uploaded_files and not (dir_path and dir_path.strip()):
        return "Please upload files or enter a server directory path."

    parts: list[str] = []
    if total:
        parts.append(f"Ingested {total} chunk(s) successfully.")
    parts.extend(errors)
    return " ".join(parts) if parts else "Nothing ingested."


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="RAG Assistant") as ui:
        gr.Markdown("# RAG Assistant")

        with gr.Tab("Query"):
            question_box = gr.Textbox(
                label="Question",
                placeholder="Ask something about your documents…",
                lines=2,
            )
            submit_btn = gr.Button("Submit", variant="primary")
            answer_box = gr.Textbox(label="Answer", lines=8, interactive=False)
            status_box = gr.Textbox(label="Status", interactive=False)

            with gr.Accordion("Evaluation Scores", open=False):
                eval_df = gr.Dataframe(
                    headers=["Metric", "Score", "Pass/Fail", "Explanation"],
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                )

            with gr.Accordion("Retrieved Sources", open=False):
                sources_df = gr.Dataframe(
                    headers=["Source", "Similarity Score", "Snippet"],
                    datatype=["str", "str", "str"],
                    interactive=False,
                )

            submit_btn.click(
                fn=handle_query,
                inputs=[question_box],
                outputs=[answer_box, status_box, eval_df, sources_df],
            )
            # Also submit on Enter inside the textbox
            question_box.submit(
                fn=handle_query,
                inputs=[question_box],
                outputs=[answer_box, status_box, eval_df, sources_df],
            )

        with gr.Tab("Ingest"):
            upload_box = gr.File(
                label="Upload files",
                file_count="multiple",
                file_types=[".txt", ".pdf", ".md", ".html", ".docx"],
            )
            dir_box = gr.Textbox(
                label="Server directory path (optional)",
                placeholder="/app/docs",
            )
            ingest_btn = gr.Button("Ingest", variant="primary")
            ingest_status = gr.Textbox(label="Status", interactive=False)

            ingest_btn.click(
                fn=handle_ingest,
                inputs=[upload_box, dir_box],
                outputs=[ingest_status],
            )

    return ui


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
