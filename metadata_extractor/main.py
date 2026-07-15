import os
import sys
import json
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("metadata_extractor")

# Add parent directory to path so relative imports work when executed from workspace root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metadata_extractor.extractor import run_extraction
from metadata_extractor.models import JudgmentMetadata

PROGRESS_FILE = "metadata_progress.jsonl"

def extract_pdf_text(pdf_path: str) -> str:
    """Fallback utility to extract text from PDFs if encountered in the folder."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text = ""
        for i, page in enumerate(doc):
            text += f"\n===== Page {i+1} =====\n" + page.get_text()
        return text
    except ImportError:
        logger.error("PyMuPDF (fitz) is not installed. Cannot parse PDF files directly.")
        raise
    except Exception as e:
        logger.error(f"Error parsing PDF {pdf_path}: {e}")
        raise

def read_file_content(file_path: str) -> str:
    """Read contents of a text file or extract text from a PDF."""
    _, ext = os.path.splitext(file_path.lower())
    if ext == ".pdf":
        return extract_pdf_text(file_path)
    
    # Otherwise read as plain text
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Unable to decode file: {file_path}")

def load_progress() -> Dict[str, Dict[str, Any]]:
    """Loads already processed files from the progress.jsonl file to support resumable runs."""
    processed = {}
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    processed[record["filename"]] = record["metadata"]
            logger.info(f"Loaded {len(processed)} already processed files from progress log.")
        except Exception as e:
            logger.error(f"Error loading progress file: {e}")
    return processed

def write_progress(filename: str, metadata: Dict[str, Any]):
    """Appends a single processed file's metadata to the progress JSONL file."""
    record = {
        "filename": filename,
        "metadata": metadata,
        "timestamp": time.time()
    }
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def compile_registry(output_file: str):
    """Compiles all records from the progress log into a formatted JSON array registry."""
    processed = load_progress()
    registry = []
    for filepath, meta in processed.items():
        # Clean relative path for storage
        registry.append({
            "filepath": filepath,
            "metadata": meta
        })
        
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    logger.info(f"Successfully compiled registry: Saved {len(registry)} records to {output_file}")

def process_single_file(filepath: str, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Reads, trims, and sends a single judgment file to the LLM extractor."""
    try:
        content = read_file_content(filepath)
        if not content.strip():
            logger.warning(f"Skipping empty file: {filepath}")
            return None
            
        # Call LLM with simple rate limit retry loop
        max_retries = 3
        backoff = 2
        for attempt in range(max_retries):
            try:
                metadata = run_extraction(
                    text=content,
                    provider=args.provider,
                    model_or_path=args.model_or_path,
                    api_key=args.api_key,
                    base_url=args.base_url
                )
                return metadata.model_dump(mode="json")
            except Exception as e:
                # Handle rate limits or temporary connection issues
                if "rate_limit" in str(e).lower() or "too many requests" in str(e).lower() or attempt < max_retries - 1:
                    logger.warning(f"Rate limit or API error on attempt {attempt+1} for {filepath}: {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise
                    
    except Exception as e:
        logger.error(f"Failed to process {filepath}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="ROScribe Standalone Judicial Metadata Ingestion Engine (resumable & concurrent)"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a text/PDF file, or a directory containing judgments."
    )
    parser.add_argument(
        "--output", "-o",
        default="metadata_registry.json",
        help="Consolidated output JSON file path (default: metadata_registry.json)"
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["openai", "anthropic", "llamacpp"],
        default="openai",
        help="LLM orchestration provider (default: openai)"
    )
    parser.add_argument(
        "--model-or-path", "-m",
        required=True,
        help="Model identifier (e.g. gpt-4o-mini, claude-3-5-haiku-20241022) or local GGUF path"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of concurrent file processing threads (default: 4)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Limit number of files to process (for dry runs)"
    )
    parser.add_argument(
        "--api-key",
        help="Explicit API Key (overrides env values)"
    )
    parser.add_argument(
        "--base-url",
        help="Custom base URL for OpenAI-compatible endpoints (e.g. Groq, OpenRouter, local server)"
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Skip extraction and simply compile progress JSONL into the registry output file."
    )

    args = parser.parse_args()

    # Compile-only mode
    if args.compile_only:
        compile_registry(args.output)
        return

    # Scan for files
    target_files = []
    if os.path.isfile(args.input):
        target_files.append(args.input)
    elif os.path.isdir(args.input):
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.lower().endswith((".txt", ".pdf")):
                    target_files.append(os.path.join(root, file))
    else:
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)

    if not target_files:
        logger.warning("No .txt or .pdf files found in target input path.")
        return

    # Handle limit
    if args.limit:
        target_files = target_files[:args.limit]

    logger.info(f"Discovered {len(target_files)} target files for metadata extraction.")

    # Deduplicate against progress log
    processed = load_progress()
    files_to_process = [f for f in target_files if f not in processed]
    
    logger.info(f"Skipping {len(target_files) - len(files_to_process)} already processed files.")
    logger.info(f"Starting batch extraction on {len(files_to_process)} files using {args.workers} threads.")

    if not files_to_process:
        logger.info("All files are already processed. Compiling final registry...")
        compile_registry(args.output)
        return

    # Run execution loop
    success_count = 0
    failure_count = 0
    start_time = time.time()
    
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Map futures to filepaths
            future_to_file = {
                executor.submit(process_single_file, filepath, args): filepath
                for filepath in files_to_process
            }
            
            for index, future in enumerate(as_completed(future_to_file), 1):
                filepath = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        write_progress(filepath, result)
                        success_count += 1
                        logger.info(f"[{index}/{len(files_to_process)}] SUCCESS: {filepath} -> Case {result.get('case_number')}")
                    else:
                        failure_count += 1
                        logger.warning(f"[{index}/{len(files_to_process)}] FAILED: {filepath}")
                except Exception as exc:
                    failure_count += 1
                    logger.error(f"[{index}/{len(files_to_process)}] EXCEPTION for {filepath}: {exc}")
                
                # Print real-time stats
                elapsed = time.time() - start_time
                avg_speed = index / elapsed
                eta = (len(files_to_process) - index) / avg_speed if avg_speed > 0 else 0
                logger.info(
                    f"Stats: Speed: {avg_speed:.2f} files/sec | Success: {success_count} | Failures: {failure_count} | ETA: {eta:.0f}s"
                )
                
    except KeyboardInterrupt:
        logger.warning("\nExecution interrupted by user. Cleaning up and compiling registry...")
    finally:
        # Always compile output on finish/crash
        compile_registry(args.output)
        total_time = time.time() - start_time
        logger.info(f"Batch execution complete in {total_time:.1f}s. Success: {success_count}, Failures: {failure_count}")

if __name__ == "__main__":
    main()
