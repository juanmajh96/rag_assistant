"""Prompt templates for RAG QA, OOS detection, and LLM-as-judge evaluation."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. RAG question-answering prompt
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """\
You are a knowledgeable assistant. Answer the user's question using ONLY the \
information provided in the context below. If the context does not contain \
enough information to answer the question, say so clearly rather than \
speculating or hallucinating facts.

Keep your answer concise, accurate, and grounded in the context.\
"""

RAG_USER_TEMPLATE = """\
Context:
{context}

Question: {question}

Answer:\
"""


def build_rag_prompt(question: str, context: str) -> tuple[str, str]:
    """Return (system_prompt, user_message) for the RAG QA call."""
    return RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE.format(
        context=context, question=question
    )


# ---------------------------------------------------------------------------
# 2. Out-of-scope detection prompt
# ---------------------------------------------------------------------------

OOS_SYSTEM_PROMPT = """\
You are a classifier. Your only job is to determine whether a user question \
is within the scope of a given domain.

Respond with ONLY one word: "IN_SCOPE" or "OUT_OF_SCOPE". \
Do not include any explanation.\
"""

OOS_USER_TEMPLATE = """\
Domain description: {domain_description}

User question: {question}

Classification:\
"""


def build_oos_prompt(question: str, domain_description: str) -> tuple[str, str]:
    """Return (system_prompt, user_message) for the OOS detection call."""
    return OOS_SYSTEM_PROMPT, OOS_USER_TEMPLATE.format(
        domain_description=domain_description, question=question
    )


# ---------------------------------------------------------------------------
# 3. LLM-as-judge evaluation prompts (one per metric)
# ---------------------------------------------------------------------------

_EVALUATION_SYSTEM = """\
You are an expert evaluator of AI-generated answers. Score the provided \
answer on a scale of 0 to 10, where 0 is completely wrong/irrelevant and \
10 is perfect. Respond with JSON in the following format:

{"score": <integer 0-10>, "explanation": "<one or two sentences>"}\
"""

_METRIC_USER_TEMPLATES: dict[str, str] = {
    "relevance": """\
Question: {question}

Answer: {answer}

Task: Does this answer directly address the question? \
Score how relevant the answer is to the question asked (0–10).\
""",
    "faithfulness": """\
Context:
{context}

Answer: {answer}

Task: Is every factual claim in the answer supported by the context? \
Score faithfulness (0–10), deducting for any unsupported or contradicted claims.\
""",
    "retrieval_precision": """\
Question: {question}

Retrieved context chunks:
{context}

Task: For each chunk, assess whether it is relevant to answering the question. \
Score the overall precision of the retrieved set (0–10).\
""",
    "completeness": """\
Question: {question}

Answer: {answer}

Task: Does the answer cover all aspects of the question? \
Score completeness (0–10), deducting for missing key points.\
""",
}


def build_evaluation_prompt(
    metric: str,
    question: str,
    answer: str,
    context: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_message) for the given *metric* evaluation.

    Args:
        metric: One of "relevance", "faithfulness", "retrieval_precision",
                "completeness".
        question: The user's original question.
        answer: The generated answer to evaluate.
        context: The retrieved context (joined chunk texts).
    """
    if metric not in _METRIC_USER_TEMPLATES:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Valid options: {list(_METRIC_USER_TEMPLATES)}"
        )
    user_msg = _METRIC_USER_TEMPLATES[metric].format(
        question=question, answer=answer, context=context
    )
    return _EVALUATION_SYSTEM, user_msg
