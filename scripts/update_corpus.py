#!/usr/bin/env python3
"""Incremental corpus update — fetch new Supreme Court judgements and fold them
into the live index, without re-doing the whole 3,800-case corpus.

What it does, in order:
  1. Scrape the two SC listings (archive table + directory index) — one GET each.
  2. Diff against what's already on disk → the set of *new* PDFs.
  3. Download only those (polite delay, resumable, skips files already present).
  4. Refresh data/manifest.json (the audit trail + metadata the app reads).
  5. Index the new PDFs into SQLite + Chroma (PyMuPDF/OCR → chunks → embeddings).
     The indexer is resumable, so a previously interrupted run self-heals here.
  6. Incrementally add the new cases to the FTS keyword index.
  7. Print a summary (new cases, new chunks, date span, failures).

Run it monthly (see `roscribe.sh update` / the launchd plist in docs/) or any
time with a custom window:

  .venv/bin/python -m scripts.update_corpus                 # everything new
  .venv/bin/python -m scripts.update_corpus --since 2026    # only 2026 onward
  .venv/bin/python -m scripts.update_corpus --limit 20      # cap new downloads
  .venv/bin/python -m scripts.update_corpus --dry-run       # show plan only
  .venv/bin/python -m scripts.update_corpus --metadata-only # refresh manifest

IMPORTANT: run with the SAME embedder as the live corpus (bge-m3). Do NOT set
ROSCRIBE_EMBEDDER=default here — that writes to a different Chroma collection
than the app reads. The script prints the target collection so you can confirm.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import scrape, store  # noqa: E402
from src.index import build_judgements  # noqa: E402
from src.ingest import case_no_from_filename  # noqa: E402


def _disk_names(data_dir: Path) -> set[str]:
    """The safe on-disk filenames already downloaded (download() rewrites names)."""
    return {p.name for p in data_dir.glob("*.pdf")}


def _matches_since(rec: scrape.JudgementRecord, since: str | None) -> bool:
    """True if a record is on/after the `since` window. Dated records compare by
    ISO date; undated (directory-index) records fall back to the latest 4-digit
    year in the filename (the case-number year) — never a raw string compare,
    which would wrongly admit every letter-prefixed filename (`'S' > '2'`)."""
    if not since:
        return True
    if rec.date:
        return rec.date >= since
    since_yr = since[:4]
    if not since_yr.isdigit():
        return False
    yrs = [int(y) for y in re.findall(r"(?:19|20)\d{2}", rec.filename or "")]
    return bool(yrs) and max(yrs) >= int(since_yr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Incrementally update the ROScribe judgement corpus.")
    ap.add_argument("--since", type=str, default=None,
                    help="only consider judgements on/after this date or year (e.g. 2026 or 2026-01)")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of NEW downloads")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between downloads (politeness)")
    ap.add_argument("--metadata-only", action="store_true", help="refresh the manifest only; no downloads/index")
    ap.add_argument("--no-index", action="store_true", help="download only; skip indexing + FTS")
    ap.add_argument("--rebuild-fts", action="store_true", help="rebuild the whole FTS index instead of incremental")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and change nothing")
    ap.add_argument("--data-dir", type=Path, default=scrape.DATA_DIR)
    ap.add_argument("--manifest", type=Path, default=scrape.MANIFEST_PATH)
    args = ap.parse_args(argv)

    print(f"Target Chroma collection: {store.COLLECTION}")
    if store.COLLECTION.endswith("_default"):
        print("  WARNING: this is the DEFAULT (English) collection, not the app's bge-m3 one.\n"
              "  Unset ROSCRIBE_EMBEDDER so the update lands in the collection the app reads.")

    session = scrape.make_session()
    print("Scraping current listings …", flush=True)
    try:
        records = scrape.collect(session)
    except Exception as e:  # noqa: BLE001 — network/site issues shouldn't traceback
        print(f"  Could not reach the Supreme Court site: {e}")
        return 1
    print(f"  {len(records)} judgements listed online.")

    existing = _disk_names(args.data_dir)
    candidates = [r for r in records if _matches_since(r, args.since)]
    new_records = [r for r in candidates if scrape._safe_name(r.filename) not in existing]
    if args.limit is not None:
        new_records = new_records[: args.limit]

    span = ""
    dated = sorted(r.date for r in new_records if r.date)
    if dated:
        span = f"  ({dated[0]} … {dated[-1]})"
    print(f"  {len(existing)} already on disk · {len(new_records)} new to fetch{span}")

    if args.dry_run:
        for r in new_records[:30]:
            print(f"   + {r.date or '????-??-??'}  {r.case_no or r.filename}")
        if len(new_records) > 30:
            print(f"   … and {len(new_records) - 30} more")
        print("Dry run — nothing written.")
        return 0

    # Always refresh the manifest so metadata for existing rows stays current.
    scrape.save_manifest(records, args.manifest)
    print(f"Manifest refreshed: {args.manifest}")
    if args.metadata_only:
        return 0

    args.data_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[scrape.JudgementRecord] = []
    for i, rec in enumerate(new_records, 1):
        try:
            scrape.download(session, rec, args.data_dir, args.delay)
            downloaded.append(rec)
            print(f"  [{i}/{len(new_records)}] ↓ {rec.filename}", flush=True)
        except Exception as e:  # noqa: BLE001 — one bad download must not abort the run
            print(f"  [{i}/{len(new_records)}] FAILED {rec.filename}: {e}", flush=True)
    # Persist downloaded/local_path back into the manifest.
    scrape.save_manifest(records, args.manifest)

    if not downloaded:
        print("\nNo new judgements downloaded — corpus already up to date.")
        return 0
    if args.no_index:
        print(f"\nDownloaded {len(downloaded)} new PDFs (indexing skipped via --no-index).")
        return 0

    # Index: resumable, so this embeds the new PDFs (and any earlier unindexed ones).
    print(f"\nIndexing new judgements into {store.COLLECTION} …", flush=True)
    build_judgements()

    # Make the new cases keyword-searchable (incremental FTS, no full rebuild).
    new_case_nos = sorted({(r.case_no or case_no_from_filename(r.filename)).strip() for r in downloaded})
    if args.rebuild_fts:
        print("Rebuilding the full FTS index …", flush=True)
        store.build_fts(store.COLLECTION, rebuild=True)
        n_chunks = "all"
    else:
        n_chunks = store.fts_index_cases(new_case_nos)
        print(f"FTS: indexed {n_chunks} chunks across {len(new_case_nos)} new cases.")

    print("\n── Update complete ─────────────────────────────")
    print(f"  New judgements:  {len(downloaded)}")
    print(f"  New FTS chunks:  {n_chunks}")
    if span:
        print(f"  Date span:      {span.strip()}")
    print(f"  Failed:          {len(new_records) - len(downloaded)}")
    print("  Restart the app (./scripts/roscribe.sh restart) to serve them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
