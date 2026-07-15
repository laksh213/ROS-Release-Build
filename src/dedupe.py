"""Detect and collapse duplicate judgement rows — same case listed twice.

The two scrape sources (archive table + directory index) are merged by PDF
*filename*, so one judgment uploaded under two filename variants becomes two
`judgements` rows with differently formatted case numbers, e.g.
``SC/APPEAL/243/2014`` (archive, rich metadata) and ``SC APPEAL 243 14``
(directory index, bare). Both then show up in every search.

Two rows are merged only when BOTH tests pass:

  1. **Same case identity** — same court-prefix letters and the same set of
     (number, 4-digit year) signatures, via the same normalization that powers
     `store.resolve_citation`.
  2. **Same content** — identical PDF bytes (md5), or extracted-text similarity
     ≥ SIM_THRESHOLD. Measured on the real corpus, true re-uploads score
     ≥ 0.91 and *different documents of the same case* (leave order vs final
     judgment — common, and NOT duplicates) score ≤ 0.60, so the gate is wide.

The richest-metadata row survives; missing fields are backfilled from the
dropped twins. Dropped rows' FTS chunks, Chroma vectors, bench rows, analyses
and bookmarks are deleted or re-pointed to the keeper. PDFs on disk and their
`indexed` markers are never touched — the markers are what stop a later
`src.index` run from re-ingesting the dropped files.

Verdicts (same/different per md5 pair) are cached in `dedupe_verdicts`, and
every dropped row is recorded in `dedupe_log` (audit trail + lets the Chroma
vector cleanup be re-run if it is ever interrupted).

Library use:  `dedupe_pass(con, apply=True)` — called at the end of
`src.index.build_judgements`, so plain index runs and the monthly
`scripts.update_corpus` self-heal. CLI: `python -m scripts.dedupe`.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from .config import settings
from .store import _case_signatures

SIM_THRESHOLD = 0.90      # text similarity at/above which two PDFs are one doc
MIN_COMPARABLE = 200      # fewer normalized chars than this → scanned/empty,
                          # text not comparable → never merge without same md5
_TEXT_CAP = 30_000        # chars of normalized text compared per document


def case_group_key(case_no: str) -> tuple[str, tuple] | None:
    """Identity key for grouping case-number variants: (court-prefix letters,
    sorted set of (number, year4) signatures). 'SC/APPEAL/243/2014' and
    'SC APPEAL 243 14' both map to ('scappeal', (('243','2014'),))."""
    letters = re.sub(r"[^a-z]", "", (case_no or "").lower())
    sigs = tuple(sorted(set(_case_signatures(case_no or ""))))
    if not sigs:
        return None
    return (letters, sigs)


# --------------------------------------------------------------------------- #
# Content identity                                                             #
# --------------------------------------------------------------------------- #
def _md5(path: str) -> str | None:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def _norm_text(path: str) -> str:
    """Normalized extracted text from the Docling parse cache (dedupe runs after
    indexing, so the parse exists; an unparseable file → not comparable)."""
    try:
        from .parsing import extract_pages

        text = "".join(extract_pages(path))
    except Exception:  # noqa: BLE001 — unreadable file → not comparable
        return ""
    return re.sub(r"[^a-z0-9]", "", text.lower())[:_TEXT_CAP]


def _ensure_dedupe_tables(con: sqlite3.Connection) -> None:
    con.execute(
        """CREATE TABLE IF NOT EXISTS dedupe_verdicts (
            md5_pair TEXT PRIMARY KEY, same INTEGER, sim REAL)"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS dedupe_log (
            filename TEXT PRIMARY KEY, case_no TEXT,
            keeper_filename TEXT, keeper_case_no TEXT,
            dropped_at TEXT, vectors_cleaned INTEGER DEFAULT 0)"""
    )
    con.commit()


def _same_content(con: sqlite3.Connection, a: dict, b: dict,
                  md5s: dict[str, str | None], texts: dict[str, str]) -> bool:
    ma, mb = md5s.get(a["filename"]), md5s.get(b["filename"])
    if ma and mb and ma == mb:
        return True
    if not ma or not mb:
        return False
    pair = "|".join(sorted((ma, mb)))
    row = con.execute("SELECT same FROM dedupe_verdicts WHERE md5_pair=?", (pair,)).fetchone()
    if row is not None:
        return bool(row[0])
    for r in (a, b):
        if r["filename"] not in texts:
            texts[r["filename"]] = _norm_text(r["local_path"])
    ta, tb = texts[a["filename"]], texts[b["filename"]]
    if len(ta) < MIN_COMPARABLE or len(tb) < MIN_COMPARABLE:
        sim, same = 0.0, False  # scanned/empty → cannot prove sameness
    else:
        sim = SequenceMatcher(None, ta, tb).ratio()
        same = sim >= SIM_THRESHOLD
    con.execute("INSERT OR REPLACE INTO dedupe_verdicts (md5_pair, same, sim) VALUES (?,?,?)",
                (pair, int(same), sim))
    con.commit()
    return same


# --------------------------------------------------------------------------- #
# Planning                                                                     #
# --------------------------------------------------------------------------- #
_META_FIELDS = ("date", "parties", "judges", "keywords", "legislation",
                "pdf_url", "parties_json")


def _empty(v) -> bool:
    return v is None or v == "" or v == "[]"


def _keeper_score(r: dict) -> tuple:
    richness = sum((not _empty(r.get("date")), not _empty(r.get("parties")),
                    not _empty(r.get("judges")),
                    not (_empty(r.get("keywords")) and _empty(r.get("legislation")))))
    return (richness, (r.get("n_chunks") or 0) > 0, "/" in (r.get("case_no") or ""),
            len(r.get("parties") or ""), r.get("n_chunks") or 0,
            -len(r.get("filename") or ""))


@dataclass
class MergeCluster:
    keeper: dict
    dropped: list[dict] = field(default_factory=list)
    reason: str = ""          # "md5" or "text~0.97"
    backfill: dict = field(default_factory=dict)


def _load_rows(con: sqlite3.Connection) -> list[dict]:
    cols = [r[1] for r in con.execute("PRAGMA table_info(judgements)")]
    return [dict(zip(cols, row)) for row in con.execute("SELECT * FROM judgements")]


def plan_dedupe(con: sqlite3.Connection, quiet: bool = True) -> list[MergeCluster]:
    """Group rows by case identity, sub-cluster by content, pick keepers.
    Pure planning — no writes beyond the verdict cache."""
    _ensure_dedupe_tables(con)
    groups: dict[tuple, list[dict]] = {}
    for r in _load_rows(con):
        key = case_group_key(r.get("case_no") or "")
        if key:
            groups.setdefault(key, []).append(r)

    clusters: list[MergeCluster] = []
    for key, rows in groups.items():
        if len(rows) < 2:
            continue
        md5s = {r["filename"]: _md5(r["local_path"] or "") for r in rows}
        texts: dict[str, str] = {}
        # Union-find over content equality: twins of twins land in one cluster.
        parent = list(range(len(rows)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if find(i) != find(j) and _same_content(con, rows[i], rows[j], md5s, texts):
                    parent[find(j)] = find(i)

        buckets: dict[int, list[dict]] = {}
        for i, r in enumerate(rows):
            buckets.setdefault(find(i), []).append(r)
        for members in buckets.values():
            if len(members) < 2:
                continue
            members.sort(key=_keeper_score, reverse=True)
            keeper, dropped = members[0], members[1:]
            uniq_md5 = {md5s.get(m["filename"]) for m in members}
            reason = "md5" if len(uniq_md5) == 1 and None not in uniq_md5 else "text"
            backfill = {}
            for f in _META_FIELDS:
                if _empty(keeper.get(f)):
                    donor = next((d[f] for d in dropped if not _empty(d.get(f))), None)
                    if donor is not None:
                        backfill[f] = donor
            clusters.append(MergeCluster(keeper, dropped, reason, backfill))
            if not quiet:
                drops = ", ".join(repr(d["case_no"]) for d in dropped)
                print(f"  keep {keeper['case_no']!r:42} drop {drops}  [{reason}]")
    return clusters


# --------------------------------------------------------------------------- #
# Applying                                                                     #
# --------------------------------------------------------------------------- #
def _repoint_or_delete(con: sqlite3.Connection, table: str, keeper_cn: str,
                       dropped_cn: str) -> None:
    """Move `table` rows from the dropped case_no to the keeper when the keeper
    has none (so nothing is lost), else drop them as redundant twins."""
    has_keeper = con.execute(
        f"SELECT 1 FROM {table} WHERE case_no=? LIMIT 1", (keeper_cn,)
    ).fetchone()
    if has_keeper:
        con.execute(f"DELETE FROM {table} WHERE case_no=?", (dropped_cn,))
    else:
        con.execute(f"UPDATE {table} SET case_no=? WHERE case_no=?", (keeper_cn, dropped_cn))


def apply_clusters(con: sqlite3.Connection, clusters: list[MergeCluster],
                   clean_vectors: bool = True, quiet: bool = True) -> dict:
    """Execute the plan: SQLite first (one transaction per cluster, logged in
    dedupe_log), then Chroma vector cleanup driven off the log — so an
    interrupted run resumes correctly."""
    _ensure_dedupe_tables(con)
    stats = {"clusters": 0, "rows_dropped": 0, "fts_deleted": 0, "fts_repointed": 0,
             "backfilled": 0, "same_case_no_twins": 0}
    now = datetime.now().isoformat(timespec="seconds")

    for cl in clusters:
        k = cl.keeper
        for f, v in cl.backfill.items():
            con.execute(f"UPDATE judgements SET {f}=? WHERE filename=?", (v, k["filename"]))
            stats["backfilled"] += 1
        for d in cl.dropped:
            # If ANY still-surviving row carries this case_no (the keeper, or an
            # unmerged different-document row of the same case), the FTS/bench/
            # analyses rows under it are shared — touch nothing but the extra
            # judgements row, and never delete its vectors.
            same_cn = con.execute(
                "SELECT 1 FROM judgements WHERE case_no=? AND filename!=? LIMIT 1",
                (d["case_no"], d["filename"]),
            ).fetchone() is not None
            if same_cn:
                stats["same_case_no_twins"] += 1
            else:
                keeper_has_chunks = (k.get("n_chunks") or 0) > 0 or con.execute(
                    "SELECT 1 FROM chunks_fts WHERE case_no=? LIMIT 1", (k["case_no"],)
                ).fetchone()
                if keeper_has_chunks:
                    cur = con.execute("DELETE FROM chunks_fts WHERE case_no=?", (d["case_no"],))
                    stats["fts_deleted"] += cur.rowcount
                else:
                    cur = con.execute("UPDATE chunks_fts SET case_no=? WHERE case_no=?",
                                      (k["case_no"], d["case_no"]))
                    stats["fts_repointed"] += cur.rowcount
                    con.execute("UPDATE judgements SET n_chunks=? WHERE filename=?",
                                (d.get("n_chunks") or 0, k["filename"]))
                try:
                    _repoint_or_delete(con, "case_judges", k["case_no"], d["case_no"])
                except sqlite3.Error:
                    con.execute("DELETE FROM case_judges WHERE case_no=?", (d["case_no"],))
                _repoint_or_delete(con, "analyses", k["case_no"], d["case_no"])
                for t in ("bookmarks", "annotations"):
                    try:
                        con.execute(f"UPDATE OR IGNORE {t} SET case_no=? WHERE case_no=?",
                                    (k["case_no"], d["case_no"]))
                        con.execute(f"DELETE FROM {t} WHERE case_no=?", (d["case_no"],))
                    except sqlite3.OperationalError:
                        pass  # table not created yet on this install
            con.execute("DELETE FROM judgements WHERE filename=?", (d["filename"],))
            con.execute(
                "INSERT OR REPLACE INTO dedupe_log "
                "(filename, case_no, keeper_filename, keeper_case_no, dropped_at, vectors_cleaned) "
                "VALUES (?,?,?,?,?,?)",
                (d["filename"], d["case_no"], k["filename"], k["case_no"], now,
                 1 if same_cn else 0),  # shared-case_no vectors must NOT be cleaned
            )
            stats["rows_dropped"] += 1
        con.commit()
        stats["clusters"] += 1

    if clean_vectors:
        stats["vectors_cleaned_cases"] = clean_dropped_vectors(con, quiet=quiet)
    return stats


def clean_dropped_vectors(con: sqlite3.Connection, quiet: bool = True) -> int:
    """Delete Chroma vectors for case_nos recorded in dedupe_log and not yet
    cleaned. Identical content survives under the keeper, so this only removes
    redundant copies. Safe to re-run; returns the number of case_nos cleaned."""
    _ensure_dedupe_tables(con)
    # A logged case_no that still exists on a surviving judgements row shares
    # its vectors with that row — mark it cleaned without touching Chroma.
    con.execute(
        "UPDATE dedupe_log SET vectors_cleaned=1 WHERE vectors_cleaned=0 AND "
        "EXISTS (SELECT 1 FROM judgements j WHERE j.case_no = dedupe_log.case_no)"
    )
    con.commit()
    pending = [r[0] for r in con.execute(
        "SELECT DISTINCT case_no FROM dedupe_log WHERE vectors_cleaned=0"
    )]
    if not pending:
        return 0
    try:
        import chromadb

        from . import store

        client = chromadb.PersistentClient(path=settings.chroma_dir)
        names = {c.name for c in client.list_collections()}
        name = store.COLLECTION if store.COLLECTION in names else next(
            (n for n in names if n.startswith("judgements_")), None)
        if name is None:
            print("[dedupe] no judgements collection found — vector cleanup skipped")
            return 0
        col = client.get_collection(name)
    except Exception as e:  # noqa: BLE001 — vectors are an optimization, not integrity
        print(f"[dedupe] Chroma unavailable, vector cleanup deferred: {e}")
        return 0

    cleaned = 0
    for i in range(0, len(pending), 50):
        batch = pending[i:i + 50]
        try:
            col.delete(where={"case_no": {"$in": batch}})
        except Exception as e:  # noqa: BLE001
            print(f"[dedupe] vector delete failed for {len(batch)} cases: {e}")
            continue
        con.executemany("UPDATE dedupe_log SET vectors_cleaned=1 WHERE case_no=?",
                        [(cn,) for cn in batch])
        con.commit()
        cleaned += len(batch)
    if not quiet and cleaned:
        print(f"[dedupe] removed vectors for {cleaned} duplicate case_nos from {name}")
    return cleaned


def dedupe_pass(con: sqlite3.Connection | None = None, apply: bool = False,
                clean_vectors: bool = True, quiet: bool = True) -> dict:
    """One full detect→(optionally) merge pass. Returns stats. Cheap when the
    table is already clean: only multi-row identity groups are examined, and
    content verdicts are cached by md5 pair."""
    own = con is None
    if own:
        con = sqlite3.connect(settings.sqlite_path)
    try:
        clusters = plan_dedupe(con, quiet=quiet)
        report = {
            "duplicate_clusters": len(clusters),
            "rows_to_drop": sum(len(c.dropped) for c in clusters),
        }
        if not clusters:
            return {**report, "applied": False}
        if apply:
            report.update(apply_clusters(con, clusters, clean_vectors=clean_vectors, quiet=quiet))
            report["applied"] = True
            if not quiet:
                print(f"[dedupe] merged {report['rows_to_drop']} duplicate rows "
                      f"into {report['duplicate_clusters']} cases")
        else:
            report["applied"] = False
        return report
    finally:
        if own:
            con.close()


def write_report(clusters: list[MergeCluster], path: str | Path) -> None:
    """Human-auditable JSON of every planned merge (written by the CLI)."""
    payload = [
        {
            "keeper": {"case_no": c.keeper["case_no"], "filename": c.keeper["filename"]},
            "dropped": [{"case_no": d["case_no"], "filename": d["filename"]} for d in c.dropped],
            "reason": c.reason,
            "backfilled_fields": sorted(c.backfill),
        }
        for c in clusters
    ]
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
