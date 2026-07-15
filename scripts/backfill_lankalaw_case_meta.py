"""One-time migration: give already-ingested lankalaw.net law-report cases their
short display names and decision years.

  'A. A. GUNAWARDENA Appellant and Mrs. R. K. D. GUNAWARDENA Respondent'
      -> case_no 'Gunawardena v. Gunawardena (76 NLR)', date from the judgment
         text (NLR) or the report year (SLR), parties = the short name.

Updates every layer keyed by case_no: judgements, chunks_fts (by rowid — one
scan up front, no per-case FTS scans), Chroma chunk metadata, and the
bookmarks / annotations / case_judges / analyses side tables. Idempotent:
already-migrated rows are skipped. Run only while the ingest is NOT running
(single Chroma writer).

  .venv/bin/python -u scripts/backfill_lankalaw_case_meta.py [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings                      # noqa: E402
from src.index import case_year, short_case_name    # noqa: E402

# An already-migrated case_no: '<short> (76 NLR)' / '<short> (SLR 1982)' + optional [tag]
_CITE_SUFFIX = re.compile(r"\s*\((?:\d{1,3} NLR|SLR \d{4})\)(?:\s*\[[^\]]+\])?$")

from src.ingest import extract_opinion_author as author_from_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print planned changes, write nothing")
    ap.add_argument("--meta-only", action="store_true",
                    help="dates/authors/parties only (SQLite; safe while the ingest runs) — "
                         "no case_no renames, no FTS/Chroma writes")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    con = sqlite3.connect(settings.sqlite_path)
    rows = con.execute(
        """SELECT filename, case_no, COALESCE(report_cite,''), COALESCE(date,''),
                  COALESCE(judges,'[]')
           FROM judgements WHERE local_path LIKE '%lankalaw_cases%' ORDER BY filename"""
    ).fetchall()
    if args.limit:
        rows = rows[: args.limit]
    used = {cn for (cn,) in con.execute("SELECT case_no FROM judgements")}

    print(f"Scanning chunks_fts rowids for {len(rows)} lankalaw cases …", flush=True)
    fts_rowids: dict[str, list[int]] = {}
    lanka_cns = {cn for _, cn, _, _, _ in rows}
    for rowid, cn in con.execute("SELECT rowid, case_no FROM chunks_fts"):
        if cn in lanka_cns:
            fts_rowids.setdefault(cn, []).append(rowid)

    col = None
    if not args.dry_run and not args.meta_only:
        from src.store import get_collection
        col = get_collection()

    done = skipped = renamed = dated = authored = 0
    for filename, old_cn, cite, old_date, old_judges in rows:
        base = _CITE_SUFFIX.sub("", old_cn)
        short = short_case_name(base)
        # Only real law-report cites belong in the display name (not 'Digest').
        show_cite = bool(re.fullmatch(r"\d{1,3} NLR|SLR \d{4}", cite or ""))
        new_cn = f"{short} ({cite})" if show_cite else short
        if new_cn != old_cn and new_cn in used:
            new_cn = f"{new_cn} [{Path(filename).stem.rsplit('-', 1)[-1][:6]}]"
        if args.meta_only:
            new_cn = old_cn  # renames need the FTS/Chroma sync — full run only

        need_author = old_judges in ("", "[]", "null")
        texts = ""
        if not old_date or need_author:
            texts = " ".join(
                t for (t,) in con.execute(
                    "SELECT text FROM chunks_fts WHERE rowid IN (%s) LIMIT 3"
                    % ",".join(map(str, fts_rowids.get(old_cn, [0])[:3]))))
        year = old_date or case_year(cite, texts)
        author = author_from_text(texts) if need_author else ""

        if new_cn == old_cn and year == old_date and not author:
            skipped += 1
            continue
        if args.dry_run:
            print(f"  {year or '----'} | by {author or '?':<18} | {old_cn[:58]!r}\n"
                  f"         -> {new_cn!r}")
            done += 1
            continue

        judges_val = json.dumps([author]) if author else old_judges
        if author:
            authored += 1
        con.execute("UPDATE judgements SET case_no=?, date=?, parties=?, judges=? WHERE filename=?",
                    (new_cn, year, short, judges_val, filename))
        if new_cn != old_cn:
            for t in ("bookmarks", "annotations", "case_judges", "analyses"):
                con.execute(f"UPDATE {t} SET case_no=? WHERE case_no=?", (new_cn, old_cn))
            for rid in fts_rowids.get(old_cn, []):
                con.execute("UPDATE chunks_fts SET case_no=? WHERE rowid=?", (new_cn, rid))
            renamed += 1
        if year != old_date:
            dated += 1
        # Chroma chunk metadata (case_no + date) — what search/filters/goto read.
        if col is not None:
            got = col.get(where={"case_no": old_cn}, include=["metadatas"])
            if got["ids"]:
                metas = [{**m, "case_no": new_cn, "date": year} for m in got["metadatas"]]
                col.update(ids=got["ids"], metadatas=metas)

        used.discard(old_cn)
        used.add(new_cn)
        done += 1
        if done % 200 == 0:
            con.commit()
            print(f"  …{done} migrated ({renamed} renamed, {dated} newly dated)", flush=True)

    con.commit()
    print(f"Backfill done: {done} migrated ({renamed} renamed, {dated} newly dated, "
          f"{authored} authoring judges found), {skipped} already correct.", flush=True)


if __name__ == "__main__":
    main()
