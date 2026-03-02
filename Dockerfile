FROM python:3.11-slim

# System dependencies needed by `unstructured` (Markdown) and PyPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY rag_assistant/ ./rag_assistant/
COPY app.py .

# Persistent volumes (declared for documentation; mounted via compose)
VOLUME ["/app/chroma_db", "/app/docs"]

# Allow the API key to be injected at runtime
ENV ANTHROPIC_API_KEY=""

EXPOSE 7860

# Default command — override via `docker compose run` or CLI flags
CMD ["python", "-m", "rag_assistant", "query", "Hello"]
