"""Central configuration. Reads from environment / .env (see .env.example).

v2 introduces hardware profiles (ROSCRIBE_PROFILE):
  lite    (default) — 8 GB / CPU-only laptops: multilingual-e5-small embeddings,
                      12k LLM context, FTS+dense RRF retrieval (no neural reranker)
  quality           — 16 GB+ machines: BAAI/bge-m3 embeddings, 16k context

Any individual setting still overrides its profile default via env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]

PROFILES = {
    "lite": {"embedding_model": "intfloat/multilingual-e5-small", "num_ctx": 12288},
    "quality": {"embedding_model": "BAAI/bge-m3", "num_ctx": 16384},
}

PROFILE = os.getenv("ROSCRIBE_PROFILE", "lite").lower()
if PROFILE not in PROFILES:
    print(f"⚠️  Unknown ROSCRIBE_PROFILE={PROFILE!r} — using 'lite'.")
    PROFILE = "lite"
_P = PROFILES[PROFILE]

# Ingest parser (v2): "docling" — the layout-aware DocumentConverter (reading
# order, provenance page anchors, built-in OCR) — or "legacy", the v1
# PyMuPDF-text + Tesseract-OCR path, kept as a fallback. Docling is the default.
PARSERS = ("docling", "legacy")
PARSER = os.getenv("ROSCRIBE_PARSER", "docling").lower()
if PARSER not in PARSERS:
    print(f"⚠️  Unknown ROSCRIBE_PARSER={PARSER!r} — using 'docling'.")
    PARSER = "docling"

# The default GGUF (scripts/get_model.py drops it here).
_DEFAULT_GGUF = REPO_ROOT / "models" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"


@dataclass(frozen=True)
class Settings:
    profile: str = PROFILE

    # --- LLM provider: "llamacpp" (local GGUF, default), "ollama", "anthropic", "openai" ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "llamacpp")
    llm_model: str = os.getenv("LLM_MODEL", "qwen3-4b-instruct")  # Ollama/OpenAI model tag
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    # Context window — must hold a (trimmed) judgment + the breakdown output.
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", str(_P["num_ctx"])))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    # llamacpp: run a GGUF directly (no server)
    llamacpp_model_path: str = os.getenv(
        "LLAMACPP_MODEL_PATH", str(_DEFAULT_GGUF) if _DEFAULT_GGUF.exists() else ""
    )
    llamacpp_gpu_layers: int = int(os.getenv("LLAMACPP_GPU_LAYERS", "-1"))
    # "q8_0" halves KV-cache RAM (≈1.1 GB at 12k ctx for a 4B model); "" = fp16.
    llamacpp_kv_type: str = os.getenv("LLAMACPP_KV_TYPE", "")
    # Anthropic (optional — set LLM_PROVIDER=anthropic to use)
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")

    # --- Embeddings (local, multilingual EN / Sinhala / Tamil) ---
    embedding_model: str = os.getenv("EMBEDDING_MODEL", _P["embedding_model"])
    # "auto" tries the ONNX backend (lower RAM) and falls back to torch.
    embed_backend: str = os.getenv("ROSCRIBE_EMBED_BACKEND", "auto")
    # Optional neural reranker (quality profile; needs `pip install FlagEmbedding`).
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    use_reranker: bool = os.getenv("USE_RERANKER", "false").lower() in ("1", "true", "yes")

    # --- Document parsing (Docling) ---
    # "docling" (v2 default, layout-aware) or "legacy" (v1 PyMuPDF + Tesseract).
    parser: str = PARSER
    parse_threads: int = int(os.getenv("ROSCRIBE_PARSE_THREADS", str(min(8, os.cpu_count() or 4))))
    # "auto" = OCR pages without a text layer (needs system tesseract); "off" = never.
    ocr_mode: str = os.getenv("ROSCRIBE_OCR", "auto").lower()
    tesseract_langs: str = os.getenv("TESSERACT_LANGS", "eng+sin+tam")

    # --- Local storage ---
    sqlite_path: str = os.getenv("SQLITE_PATH", str(REPO_ROOT / "data" / "roscribe.db"))
    chroma_dir: str = os.getenv("CHROMA_DIR", str(REPO_ROOT / "data" / "chroma"))
    parsed_dir: str = os.getenv("ROSCRIBE_PARSED_DIR", str(REPO_ROOT / "data" / "parsed"))

    # --- Personal repository (your law-school notes) ---
    personal_repo_dir: str = os.getenv("PERSONAL_REPO_DIR", "")


settings = Settings()
