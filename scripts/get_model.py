"""Fetch the default local LLM (GGUF) into models/.

Usage:
  .venv/bin/python scripts/get_model.py            # Qwen3-4B-Instruct Q4_K_M (default)
  .venv/bin/python scripts/get_model.py --tier min # Qwen2.5-1.5B for 4-6 GB machines

After download, point .env at it:
  LLM_PROVIDER=llamacpp
  LLAMACPP_MODEL_PATH=models/<file>.gguf
"""

from __future__ import annotations

import argparse
from pathlib import Path

# (repo_id, filename) per tier — all open weights, Q4_K_M quantization.
TIERS = {
    "default": ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
    "fallback": ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
    "min": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
}

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download a local GGUF model.")
    ap.add_argument("--tier", choices=sorted(TIERS), default="default")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    repo, fname = TIERS[args.tier]
    MODELS_DIR.mkdir(exist_ok=True)
    path = hf_hub_download(repo, fname, local_dir=str(MODELS_DIR))
    print(f"\nModel ready: {path}\nSet in .env:\n  LLM_PROVIDER=llamacpp\n  LLAMACPP_MODEL_PATH={path}")


if __name__ == "__main__":
    main()
