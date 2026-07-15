# ROScribe v2 — multi-stage, non-root, reproducible (installs from the lockfile).
FROM python:3.12-slim AS builder

# Build tools only exist in this stage; the runtime image ships without them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.lock


FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ROSCRIBE_HOST=0.0.0.0 \
    ROSCRIBE_PORT=8080

# Runtime system deps: Tesseract OCR (driven through Docling) with the three
# corpus languages. No compilers in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-sin tesseract-ocr-tam \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

COPY --from=builder /install /usr/local
WORKDIR /app
COPY app/ ./app/
COPY src/ ./src/
COPY prompts/ ./prompts/
COPY taxonomy/ ./taxonomy/
COPY scripts/ ./scripts/

RUN mkdir -p /app/data /app/models && chown -R app:app /app
USER app

EXPOSE 8080
CMD ["python", "app/workspace.py"]
