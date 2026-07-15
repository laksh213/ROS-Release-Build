"""Clean, resumable Docling reparse of the corpus (v2).

`python -m src.index --force` re-embeds via the Docling path now, but it would
leave ORPHAN chunks behind: Docling produces different chunk ids than the old
PyMuPDF path, so the stale vectors are never overwritten. This script replaces
each judgment's chunks atomically instead:

  1. delete the case's existing vectors from Chroma (where case_no == cn),
  2. parse with Docling (cached under data/parsed/),
  3. add the structure-aware Docling chunks,
  4. refresh that case's FTS rows,
  5. mark it done on the 'judgment_docling' source so a re-run resumes.

Resumable + idempotent: safe to interrupt and restart, safe to run overnight.
Per-case commit means the live collection is always consistent (no half state),
so search keeps working throughout the reparse.

  python scripts/reparse_docling.py                 # reparse all (resume)
  python scripts/reparse_docling.py --limit 50      # one batch
  python scripts/reparse_docling.py --case SC/APPEAL/103/2021   # specific case(s)
  python scripts/reparse_docling.py --status        # how many done / remaining
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import parsing, store  # noqa: E402

DOCLING_SOURCE = "judgment_docling"  # resume marker namespace in the `indexed` table


def _rows(con, only_cases: list[str] | None):
    if only_cases:
        out = []
        for cn in only_cases:
            r = con.execute(
                "SELECT case_no, filename, local_path, date FROM judgements "
                "WHERE case_no=? OR filename=? LIMIT 1", (cn, cn)).fetchone()
            if r:
                out.append(r)
        return out
    return con.execute(
        "SELECT case_no, filename, local_path, date FROM judgements "
        "WHERE local_path IS NOT NULL AND local_path!='' ORDER BY case_no").fetchall()


def reparse(limit: int | None, only_cases: list[str] | None, force: bool) -> None:
    con = store.init_db()
    col = store.get_collection()
    rows = _rows(con, only_cases)
    done = skipped = failed = 0
    todo = [r for r in rows if force or not store.is_indexed(con, r[1], DOCLING_SOURCE)]
    if limit:
        todo = todo[:limit]
    print(f"Reparsing {len(todo)} judgement(s) with Docling into {store.COLLECTION} "
          f"(resume={not force}) …", flush=True)
    for case_no, filename, local_path, date in todo:
        pdf = Path(local_path)
        if not pdf.exists():
            print(f"  SKIP {case_no}: file missing ({local_path})", flush=True)
            skipped += 1
            continue
        try:
            t = time.time()
            # 1. drop the old chunks for this case (PyMuPDF or a prior Docling run)
            try:
                col.delete(where={"case_no": case_no})
            except Exception:
                pass
            # 2-3. Docling parse (cached) -> structure-aware chunks -> embed
            doc = parsing.parse_document(pdf)
            chunks = parsing.chunk_document(doc, case_no)
            store.add_chunks(chunks, extra_meta={"date": date or "", "filename": filename})
            # 4. refresh this case's keyword (FTS) rows
            store.fts_index_cases([case_no])
            # 5. mark done so a re-run resumes past it
            store.mark_indexed(con, filename, DOCLING_SOURCE, len(chunks))
            done += 1
            print(f"  [{done}] {case_no}: {len(chunks)} Docling chunks ({time.time()-t:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001 — one bad PDF must not kill an overnight run
            failed += 1
            print(f"  FAIL {case_no}: {type(e).__name__}: {e}", flush=True)
    print(f"\nDone: {done} reparsed, {skipped} skipped, {failed} failed.", flush=True)


def status() -> None:
    con = store.init_db()
    total = con.execute(
        "SELECT count(*) FROM judgements WHERE local_path IS NOT NULL AND local_path!=''").fetchone()[0]
    try:
        done = con.execute(
            "SELECT count(*) FROM indexed WHERE collection=? AND source=?",
            (store.COLLECTION, DOCLING_SOURCE)).fetchone()[0]
    except Exception:
        done = 0
    print(f"Docling reparse progress: {done}/{total} judgements "
          f"({(done/total*100 if total else 0):.1f}%) — {total-done} remaining.")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Clean, resumable Docling reparse of the corpus.")
    ap.add_argument("--limit", type=int, default=None, help="reparse at most N cases this run")
    ap.add_argument("--case", action="append", default=None, help="specific case_no (repeatable)")
    ap.add_argument("--force", action="store_true", help="reparse even cases already on Docling")
    ap.add_argument("--status", action="store_true", help="show progress and exit")
    args = ap.parse_args(argv)
    if args.status:
        status()
        return
    reparse(args.limit, args.case, args.force)


if __name__ == "__main__":
    main()
