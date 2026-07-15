#!/usr/bin/env python3
"""Reindex derived metadata over the whole corpus — independent of the vector
index, so it's cheap to re-run as judgments are added.

Two products, both stored in SQLite (`data/roscribe.db`):

  * **parties_json** (a column on `judgements`) — the clumped scrape `parties`
    string parsed into an ordered list of ``{name, role, side}`` so the UI can
    show parties as a list instead of one run-on paragraph. Fast (pure regex).

  * **case_judges** (a new table) — the FULL coram per case, parsed from the
    judgment (front-matter panel + end-of-judgment signatures, merged with the
    scrape authoring judge), canonicalised so "J.A.N. de Silva CJ" collapses with
    "Hon. J.A.N. De Silva, C.J.". This powers an instant bench (no per-open PDF
    parse) and a By-Justice filter that matches every justice who SAT, not just
    the author. Slow (reads/OCRs each PDF) but resumable — re-runs skip cases
    already present unless --force.

CLI:
  .venv/bin/python -m scripts.reindex                 # parties + benches (resume)
  .venv/bin/python -m scripts.reindex --parties-only  # just reparse parties (seconds)
  .venv/bin/python -m scripts.reindex --benches-only
  .venv/bin/python -m scripts.reindex --force         # redo everything
  .venv/bin/python -m scripts.reindex --limit 100     # first N (testing)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import settings  # noqa: E402
from src.ingest import extract_bench, merge_benches, parse_parties  # noqa: E402
from src.store import _canonical_justice  # noqa: E402

JUDGE_DIR = REPO_ROOT / "data" / "sc_judgements"


def _column_exists(con: sqlite3.Connection, table: str, col: str) -> bool:
    return any(r[1] == col for r in con.execute(f"PRAGMA table_info({table})"))


def reindex_parties(con: sqlite3.Connection, limit: int | None, force: bool) -> None:
    if not _column_exists(con, "judgements", "parties_json"):
        con.execute("ALTER TABLE judgements ADD COLUMN parties_json TEXT")
        con.commit()
    # Retry rows never parsed AND rows that parsed to nothing ('[]') — so improving
    # the parser and re-running picks up previous failures automatically.
    where = "" if force else "WHERE parties_json IS NULL OR parties_json IN ('', '[]')"
    rows = con.execute(
        f"SELECT case_no, parties FROM judgements {where} {'LIMIT ' + str(limit) if limit else ''}"
    ).fetchall()
    print(f"Parties: parsing {len(rows)} rows …", flush=True)
    done = empty = 0
    for cn, parties in rows:
        parsed = parse_parties(parties or "")
        con.execute("UPDATE judgements SET parties_json=? WHERE case_no=?",
                    (json.dumps(parsed, ensure_ascii=False), cn))
        done += 1
        empty += not parsed
        if done % 500 == 0:
            con.commit()
            print(f"  …{done}", flush=True)
    con.commit()
    print(f"Parties done: {done} rows ({empty} unparseable → kept as raw text).", flush=True)


def reindex_benches(con: sqlite3.Connection, limit: int | None, force: bool) -> None:
    con.execute(
        """CREATE TABLE IF NOT EXISTS case_judges (
            case_no TEXT, canonical TEXT, display TEXT, seat INTEGER,
            PRIMARY KEY (case_no, seat))"""
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_case_judges_canon ON case_judges(canonical)")
    con.commit()

    done_set = {cn for (cn,) in con.execute("SELECT DISTINCT case_no FROM case_judges")}
    rows = con.execute("SELECT case_no, filename, judges FROM judgements").fetchall()
    todo = [r for r in rows if force or r[0] not in done_set]
    if limit:
        todo = todo[:limit]
    print(f"Benches: {len(todo)} cases to parse "
          f"({len(done_set)} already done) …", flush=True)

    done = empty = 0
    for cn, filename, judges_raw in todo:
        try:
            meta_judges = [j for j in (json.loads(judges_raw) if judges_raw else []) if str(j).strip()]
        except Exception:  # noqa: BLE001
            meta_judges = []
        parsed: list[str] = []
        pdf = JUDGE_DIR / filename if filename else None
        if pdf and pdf.exists():
            try:
                parsed = extract_bench(str(pdf))
            except Exception:  # noqa: BLE001 — one bad PDF can't stop the run
                parsed = []
        bench = merge_benches(parsed, meta_judges) if parsed else meta_judges
        con.execute("DELETE FROM case_judges WHERE case_no=?", (cn,))
        seen: set[str] = set()
        seat = 0
        for name in bench:
            canon = _canonical_justice(name)
            if not canon or canon in seen:
                continue
            seen.add(canon)
            con.execute(
                "INSERT OR REPLACE INTO case_judges (case_no, canonical, display, seat) VALUES (?,?,?,?)",
                (cn, canon, name, seat),
            )
            seat += 1
        done += 1
        empty += seat == 0
        if done % 50 == 0:
            con.commit()
            print(f"  …{done}/{len(todo)} ({empty} with no bench)", flush=True)
    con.commit()
    n_rows = con.execute("SELECT count(*) FROM case_judges").fetchone()[0]
    n_just = con.execute("SELECT count(DISTINCT canonical) FROM case_judges").fetchone()[0]
    print(f"Benches done: {done} cases parsed; case_judges now holds {n_rows} seats "
          f"across {n_just} distinct justices ({empty} cases had no detectable bench).", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reindex parties + full benches over the corpus.")
    ap.add_argument("--parties-only", action="store_true")
    ap.add_argument("--benches-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo rows even if already populated")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    con = sqlite3.connect(settings.sqlite_path)
    if not args.benches_only:
        reindex_parties(con, args.limit, args.force)
    if not args.parties_only:
        reindex_benches(con, args.limit, args.force)
    con.close()
    print("\nReindex complete. Restart the app to pick up the new tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
