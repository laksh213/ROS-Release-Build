#!/usr/bin/env python3
"""Collapse duplicate judgement rows (same case under two filename variants).

Dry-run by default — prints every planned merge and writes
data/dedupe_report.json for review. Nothing on disk changes until --apply.

  .venv/bin/python -m scripts.dedupe                # plan + report only
  .venv/bin/python -m scripts.dedupe --apply        # back up DB, then merge
  .venv/bin/python -m scripts.dedupe --apply --keep-vectors   # skip Chroma cleanup

PDFs are never deleted; `indexed` markers are kept so dropped files are not
re-ingested by later index runs. See src/dedupe.py for the merge rules.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import settings  # noqa: E402
from src.dedupe import apply_clusters, plan_dedupe, write_report  # noqa: E402

REPORT_PATH = REPO_ROOT / "data" / "dedupe_report.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Merge duplicate judgement rows (dry-run by default).")
    ap.add_argument("--apply", action="store_true", help="actually merge (default: plan only)")
    ap.add_argument("--keep-vectors", action="store_true",
                    help="leave duplicate Chroma vectors in place (SQLite/FTS still cleaned)")
    ap.add_argument("--quiet", action="store_true", help="suppress per-cluster lines")
    args = ap.parse_args(argv)

    con = sqlite3.connect(settings.sqlite_path)
    total = con.execute("SELECT count(*) FROM judgements").fetchone()[0]
    print(f"Scanning {total} judgement rows for duplicates …", flush=True)
    clusters = plan_dedupe(con, quiet=args.quiet)
    n_drop = sum(len(c.dropped) for c in clusters)
    by_reason = {r: sum(1 for c in clusters if c.reason == r) for r in ("md5", "text")}
    print(f"\nFound {len(clusters)} duplicate clusters → {n_drop} rows to merge away "
          f"({by_reason['md5']} byte-identical, {by_reason['text']} same-text re-uploads).")
    write_report(clusters, REPORT_PATH)
    print(f"Full plan written to {REPORT_PATH}")

    if not clusters:
        con.close()
        return 0
    if not args.apply:
        print("\nDry run — re-run with --apply to merge. (The DB is backed up first.)")
        con.close()
        return 0

    backup = Path(settings.sqlite_path).with_suffix(
        f".pre-dedupe-{datetime.now():%Y%m%d-%H%M%S}.db")
    con.close()  # back up a quiescent file, then reopen
    shutil.copy2(settings.sqlite_path, backup)
    print(f"\nBacked up SQLite DB → {backup}")

    con = sqlite3.connect(settings.sqlite_path)
    stats = apply_clusters(con, clusters, clean_vectors=not args.keep_vectors,
                           quiet=args.quiet)
    remaining = con.execute("SELECT count(*) FROM judgements").fetchone()[0]
    con.close()
    print("\n── Dedupe complete ─────────────────────────────")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  judgements rows: {total} → {remaining}")
    print("  Restart the app (./scripts/roscribe.sh restart) to serve the clean library.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
