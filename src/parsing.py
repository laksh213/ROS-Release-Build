"""v2 ingestion engine — Docling parses every document ONCE, everything reuses it.

`parse_document` converts a PDF/DOCX/HTML/MD file into a `DoclingDocument`
(layout-aware reading order, per-item page provenance, bold/italic formatting
where the backend provides it) and persists the lossless JSON under
`data/parsed/<stem>.docling.json`. Subsequent calls — chunking, bench
extraction, citation extraction, LLM context building — load the cached JSON
instead of re-parsing, so OCR and layout analysis run once per document, ever.

CPU posture for low-end hardware: table-structure recognition is OFF (judgments
are prose; TableFormer dominates per-page cost), OCR runs only on pages without
a text layer (Tesseract, eng+sin+tam — EasyOCR has no Sinhala), and thread
count is capped via ROSCRIBE_PARSE_THREADS.

CLI:
  python -m src.parsing data/sc_judgements/<file>.pdf   # parse + show stats
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from .config import settings

if TYPE_CHECKING:  # docling imports are heavy — only at type-check time here
    from docling_core.types.doc.document import DoclingDocument

PARSED_DIR = Path(settings.parsed_dir)
PARSE_FORMATS = {".pdf", ".docx", ".html", ".htm", ".md"}

_CONVERTER = None
_CONVERTER_LOCK = threading.Lock()
_CHUNKER = None


def _tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def get_converter():
    """Singleton DocumentConverter tuned for CPU-bound, prose-only judgments."""
    global _CONVERTER
    with _CONVERTER_LOCK:
        if _CONVERTER is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                AcceleratorOptions,
                PdfPipelineOptions,
                TesseractCliOcrOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption

            opts = PdfPipelineOptions()
            opts.do_table_structure = False
            opts.generate_page_images = False
            opts.accelerator_options = AcceleratorOptions(num_threads=settings.parse_threads)
            use_ocr = settings.ocr_mode != "off" and _tesseract_available()
            opts.do_ocr = use_ocr
            if use_ocr:
                langs = [lang for lang in settings.tesseract_langs.split("+") if lang]
                opts.ocr_options = TesseractCliOcrOptions(lang=langs)
            elif settings.ocr_mode != "off":
                print("[parsing] tesseract binary not found — scanned pages will be empty "
                      "(brew install tesseract tesseract-lang / apt install tesseract-ocr-*)")
            _CONVERTER = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            )
    return _CONVERTER


def parsed_path(src: str | Path) -> Path:
    return PARSED_DIR / f"{Path(src).stem}.docling.json"


def parse_document(path: str | Path, use_cache: bool = True) -> "DoclingDocument":
    """DoclingDocument for `path` — from the JSON cache when present."""
    from docling_core.types.doc.document import DoclingDocument

    pp = parsed_path(path)
    if use_cache and pp.exists():
        try:
            return DoclingDocument.load_from_json(pp)
        except Exception:
            pass  # cache written by an older docling-core — fall through and re-parse
    doc = get_converter().convert(str(path)).document
    if use_cache:
        PARSED_DIR.mkdir(parents=True, exist_ok=True)
        try:
            doc.save_as_json(pp)
        except Exception as e:  # cache failure must never fail the parse
            print(f"[parsing] could not cache {pp.name}: {e}")
    return doc


def is_parsed(path: str | Path) -> bool:
    return parsed_path(path).exists()


def iter_text_items(doc: "DoclingDocument") -> Iterator[tuple[str, int, bool | None]]:
    """Yield (text, page_no, italic) per text item in reading order.

    `italic` is True/False when the backend recorded formatting, None when
    unknown (typical for OCR'd pages, which carry no font information).
    """
    from docling_core.types.doc.document import TextItem

    for item, _level in doc.iterate_items():
        if not isinstance(item, TextItem):
            continue
        text = (item.text or "").strip()
        if not text:
            continue
        page = item.prov[0].page_no if item.prov else 1
        italic = item.formatting.italic if item.formatting is not None else None
        yield text, page, italic


def page_count(doc: "DoclingDocument") -> int:
    pages = set(getattr(doc, "pages", {}) or {})
    for _t, page, _i in iter_text_items(doc):
        pages.add(page)
    return max(pages, default=1)


def pages_text(doc: "DoclingDocument") -> list[str]:
    """Per-page text in reading order — v1's `extract_pages` shape."""
    n = page_count(doc)
    buckets: dict[int, list[str]] = {}
    for text, page, _italic in iter_text_items(doc):
        buckets.setdefault(page, []).append(text)
    return ["\n".join(buckets.get(p, [])) for p in range(1, n + 1)]


def extract_pages(path: str | Path, ocr_langs: str | None = None,
                  ocr_threshold: int | None = None) -> list[str]:
    """v1-compatible entry point. Honours ``ROSCRIBE_PARSER``: "docling" (default)
    returns Docling per-page text in reading order (cached — repeat calls don't
    re-parse); "legacy" delegates to the v1 PyMuPDF-text + Tesseract-OCR
    extractor. The return shape (one string per page) is identical either way, so
    every caller (analyze, dedupe, bench, UI) honours the flag transparently."""
    if settings.parser == "legacy":
        from .ingest import extract_pages as _legacy_extract_pages

        return _legacy_extract_pages(
            str(path),
            ocr_langs or settings.tesseract_langs,
            200 if ocr_threshold is None else ocr_threshold,
        )
    return pages_text(parse_document(path))


def pages_from_bytes(filename: str, data: bytes) -> list[str]:
    """Parse an in-memory upload (extractor page). No cache — not a corpus file."""
    suffix = Path(filename).suffix.lower() or ".pdf"
    if suffix not in PARSE_FORMATS:
        return [data.decode("utf-8", errors="ignore")]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        return pages_text(parse_document(tmp_path, use_cache=False))
    finally:
        tmp_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Chunking                                                                     #
# --------------------------------------------------------------------------- #
def _get_chunker():
    """HybridChunker aligned with the active embedder's tokenizer (≤480 tokens,
    leaving room for the e5 'passage: ' prefix within its 512 limit)."""
    global _CHUNKER
    if _CHUNKER is None:
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

        try:
            from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
            from transformers import AutoTokenizer

            tok = HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(settings.embedding_model),
                max_tokens=480,
            )
            _CHUNKER = HybridChunker(tokenizer=tok)
        except Exception:
            _CHUNKER = HybridChunker()  # default tokenizer — still structure-aware
    return _CHUNKER


def chunk_document(doc: "DoclingDocument", case_no: str, source: str = "judgment") -> list:
    """Structure-aware chunks carrying the v1 page/para anchor contract."""
    from .ingest import Chunk, chunk_pages

    try:
        chunker = _get_chunker()
        out: list[Chunk] = []
        per_page: dict[int, int] = {}
        for ch in chunker.chunk(dl_doc=doc):
            page = 1
            try:
                page = ch.meta.doc_items[0].prov[0].page_no
            except Exception:
                pass
            per_page[page] = per_page.get(page, 0) + 1
            text = chunker.contextualize(chunk=ch)  # heading-path prefix aids retrieval
            out.append(Chunk(text=text, case_no=case_no, page=page,
                             para=str(per_page[page]), source=source))
        if out:
            return out
    except Exception as e:
        print(f"[parsing] HybridChunker failed ({e}) — falling back to paragraph chunking")
    return chunk_pages(pages_text(doc), case_no, source)


def parse_and_chunk(path: str | Path, case_no: str, source: str = "judgment") -> list:
    """Parse + chunk one document with the active parser — the single ingest
    entry point used by ``src.index``.

    "docling" (``ROSCRIBE_PARSER`` default): Docling layout-aware parse →
    structure-aware chunks. "legacy": v1 PyMuPDF-text + Tesseract-OCR → paragraph
    chunks. Both return page/para-anchored ``Chunk``s with the identical schema,
    so the store / retrieve / citation-anchor contract (``[Case No | Page:Para]``)
    is unchanged whichever parser runs."""
    if settings.parser == "legacy":
        from .ingest import chunk_pages
        from .ingest import extract_pages as _legacy_extract_pages

        pages = _legacy_extract_pages(str(path), settings.tesseract_langs)
        return chunk_pages(pages, case_no, source)
    return chunk_document(parse_document(path), case_no, source)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Parse a document with Docling (cached).")
    ap.add_argument("path")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    doc = parse_document(args.path, use_cache=not args.no_cache)
    pages = pages_text(doc)
    items = list(iter_text_items(doc))
    n_italic = sum(1 for _t, _p, i in items if i)
    n_known = sum(1 for _t, _p, i in items if i is not None)
    print(f"{Path(args.path).name}: {len(pages)} pages, {len(items)} text items "
          f"({n_known} with formatting info, {n_italic} italic)")
    chunks = chunk_document(doc, case_no=Path(args.path).stem.upper())
    print(f"chunks: {len(chunks)}; first anchor {chunks[0].anchor() if chunks else '—'}")


if __name__ == "__main__":
    main()
