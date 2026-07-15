# ROScribe Structured Metadata Ingestion Engine

A decoupled, lightweight metadata extraction subsystem designed to ingest raw text judgements (or PDFs) from the Court of Appeal of Sri Lanka, extract deterministic metadata fields, and output a structured JSON registry.

Designed to process large corpora (e.g. 18,000+ files / 10 GB) concurrently and safely with automatic crash resumability.

---

## 🛠️ Features

* **High-Accuracy Structured Schemas**: Extracts Case Number, Judges (Panel), Date, Parties (Appellants vs. Respondents), Legislation Cited, and Dynamic Keywords.
* **Pluggable Orchestration**: Supports `openai` (including Groq, OpenRouter, and Ollama compatibility), `anthropic` (Claude), and `llamacpp` (direct local GGUF execution).
* **Multi-Threaded Concurrency**: Implements `ThreadPoolExecutor` to execute multiple LLM queries in parallel.
* **Crash-Resilient Resumability**: Saves progress incrementally to a JSON Lines (`.jsonl`) tracker file. If the process is killed, it skips already-analyzed files when resumed.
* **Context Truncation**: Automatically trims long legal texts to keep first and last page blocks, protecting context budgets and reducing tokens.
* **PDF Extraction Fallback**: If a folder contains `.pdf` files, it uses PyMuPDF (`fitz`) to extract text before processing.

---

## 🚀 Setup & Installation

Ensure you are inside the virtual environment:

```bash
cd /Users/laksh/Desktop/ROScribe
source .venv/bin/activate
```

Install any missing requirements (if not already met by main project):

```bash
pip install pydantic openai anthropic python-dotenv pymupdf
```

---

## ⚙️ Configuration

Ensure you have your keys added to a `.env` file in the root or export them directly in the terminal:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

For **Groq** (low-latency, lightweight Llama-3 extraction) or other custom endpoints, configure `--base-url`:

```bash
export OPENAI_API_KEY="your-groq-api-key"
```

---

## 📖 Command Line Interface (CLI)

Run `metadata_extractor/main.py` directly from the project root:

```bash
python -m metadata_extractor.main --help
```

### Options

| Option | Shorthand | Description |
| :--- | :--- | :--- |
| `--input` | `-i` | **Required**. Path to a `.txt` or `.pdf` file, or a directory of judgments. |
| `--output` | `-o` | Output file path (default: `metadata_registry.json`). |
| `--provider` | `-p` | LLM API provider (`openai`, `anthropic`, `llamacpp`). |
| `--model-or-path`| `-m` | **Required**. LLM model identifier (e.g., `gpt-4o-mini`, `claude-3-5-haiku-20241022`) or GGUF path. |
| `--workers` | `-w` | Number of concurrent threads to process files (default: `4`). |
| `--limit` | `-l` | Limit the number of files to process (ideal for quick dry runs). |
| `--base-url` | | Custom endpoint URL for Groq, OpenRouter, or local servers. |
| `--compile-only` | | Skip extraction and compile the progress log into the registry JSON. |

---

## 💡 Usage Examples

### 1. Simple Test Run (OpenAI gpt-4o-mini)
Run a limit of 5 text files using standard OpenAI:

```bash
python -m metadata_extractor.main \
  -i /path/to/judgments/ \
  -p openai \
  -m gpt-4o-mini \
  -l 5
```

### 2. High-Speed Batching using Groq (Llama-3-8b)
Process a directory of text judgments concurrently with 20 threads via Groq endpoint:

```bash
python -m metadata_extractor.main \
  -i /path/to/judgments/ \
  -p openai \
  -m llama3-8b-8192 \
  --base-url https://api.groq.com/openai/v1 \
  -w 20
```

### 3. Local direct GGUF Execution (Llama.cpp CPU/GPU)
Analyze text judgments using the local Qwen GGUF model path:

```bash
python -m metadata_extractor.main \
  -i /path/to/judgments/ \
  -p llamacpp \
  -m /Users/laksh/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff \
  -w 1
```

### 4. Compile Progress Only
If you interrupted the process and want to compile the results collected in `metadata_progress.jsonl` into `metadata_registry.json` without running further LLM requests:

```bash
python -m metadata_extractor.main --compile-only
```
