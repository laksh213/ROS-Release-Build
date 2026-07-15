"""Build the index — resumable, idempotent, error-tolerant (built for the full corpus).

Judgements: extract + chunk every downloaded PDF, join with manifest metadata,
store in SQLite + Chroma. Personal repo: walk the notes folder, tag by
Subject/Category, store under source="personal_repo".

Re-runs skip files already embedded into the *current* collection (so switching
embedders re-indexes correctly, and a crashed run resumes where it stopped).
A single broken/huge/scanned file is logged and skipped, never fatal.

CLI:
  python -m src.index                          # index downloaded judgements (resume)
  python -m src.index --force                  # re-index everything
  python -m src.index --notes "/path/to/notes" # index your law notes (resume)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from . import store
from .config import REPO_ROOT, settings
from .ingest import (
    case_no_from_filename,
    chunk_pages,
    chunks_for_note,
    list_note_files,
)
from .parsing import parse_and_chunk

JUDGEMENTS_DIR = REPO_ROOT / "data" / "sc_judgements"
MANIFEST = REPO_ROOT / "data" / "manifest.json"
STATUTES_DIR = REPO_ROOT / "data" / "statutes"
LANKALAW_CASES_DIR = REPO_ROOT / "data" / "lankalaw_cases"
LANKALAW_MANIFEST = REPO_ROOT / "data" / "lankalaw_manifest.json"


def load_manifest() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    out: dict[str, dict] = {}
    for r in json.loads(MANIFEST.read_text()):
        lp = r.get("local_path")
        if lp:
            out[Path(lp).name] = r
    return out


def load_lankalaw_manifest() -> dict[str, dict]:
    """lankalaw.net manifest keyed by local filename (statutes + cases)."""
    if not LANKALAW_MANIFEST.exists():
        return {}
    out: dict[str, dict] = {}
    for r in json.loads(LANKALAW_MANIFEST.read_text()):
        fn = r.get("filename") or (Path(r["local_path"]).name if r.get("local_path") else "")
        if fn:
            out[fn] = r
    return out


def build_judgements(limit: int | None = None, force: bool = False) -> None:
    con = store.init_db()
    meta_by_file = load_manifest()
    pdfs = sorted(JUDGEMENTS_DIR.glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]

    done = skipped = failed = 0
    print(f"Indexing {len(pdfs)} judgements into {store.COLLECTION} (resume={not force}) …", flush=True)
    for p in pdfs:
        if not force and store.is_indexed(con, p.name, "judgment"):
            skipped += 1
            continue
        try:
            meta = dict(meta_by_file.get(p.name, {}))
            case_no = meta.get("case_no") or case_no_from_filename(p.name)
            chunks = parse_and_chunk(p, case_no)
            store.add_chunks(chunks, extra_meta={"date": meta.get("date", ""), "filename": p.name})
            meta.update({"filename": p.name, "case_no": case_no, "local_path": str(p)})
            store.upsert_judgement(con, meta, len(chunks))
            store.mark_indexed(con, p.name, "judgment", len(chunks))
            done += 1
            if done % 25 == 0:
                print(f"  …{done} indexed ({skipped} skipped, {failed} failed)", flush=True)
        except Exception as e:  # noqa: BLE001 — one bad PDF must not kill the run
            failed += 1
            print(f"  SKIP {p.name}: {type(e).__name__}: {e}", flush=True)
    print(f"Judgements done: {done} indexed, {skipped} already-done, {failed} failed.", flush=True)

    if done:
        # The two scrape sources can list one judgment under two filename
        # variants; merge any twins this run just (re-)introduced.
        from .dedupe import dedupe_pass

        rep = dedupe_pass(con, apply=True, quiet=True)
        if rep.get("rows_to_drop"):
            print(f"Dedupe: merged {rep['rows_to_drop']} duplicate rows into "
                  f"{rep['duplicate_clusters']} cases.", flush=True)


def build_notes(directory: str, limit: int | None = None, force: bool = False) -> None:
    con = store.init_db()
    files = list_note_files(directory)
    if limit:
        files = files[:limit]

    done = skipped = failed = total_chunks = 0
    print(f"Indexing {len(files)} note files into {store.COLLECTION} (resume={not force}) …", flush=True)
    for p, subject, category in files:
        if not force and store.is_indexed(con, str(p), "personal_repo"):
            skipped += 1
            continue
        try:
            chunks = chunks_for_note(p, subject, category)
            store.add_chunks(chunks)
            store.mark_indexed(con, str(p), "personal_repo", len(chunks))
            total_chunks += len(chunks)
            done += 1
            print(f"  [{done}] {subject}/{category}/{p.name}: {len(chunks)} chunks", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  SKIP {p.name}: {type(e).__name__}: {e}", flush=True)
    print(f"Notes done: {done} files ({total_chunks} chunks), {skipped} already-done, {failed} failed.", flush=True)


# Anchor text that names a language/edition, not the Act — fall back to category.
# Includes the native-script language labels used on the provincial-statute pages
# (a few Acts are linked once per language: English / සිංහල / தமிழ்).
_JUNK_TITLE = {"english", "sinhala", "tamil", "sinhalese", "click to view",
               "download", "view", "pdf", "here", "",
               "සිංහල", "தமிழ்", "ඉංග්‍රීසි", "ஆங்கிலம்"}


def _statute_title(meta: dict, p: Path) -> str:
    """Best display title for a statute PDF: the scraped Act name, else the source
    category (e.g. 'Provincial Council And Local Authority Statutes'), else the
    filename — never a bare 'English'/'Click to view' language link."""
    title = (meta.get("title") or "").strip()
    if title.lower() not in _JUNK_TITLE:
        return title
    cat = (meta.get("category") or "").replace("-", " ").strip()
    return cat.title() or p.stem.rsplit("-", 1)[0].replace("-", " ").title()


def _unique_statute_id(con, base: str, filename: str) -> str:
    """A statute_id unique per PDF (the chunk case_no, so chunks never collide):
    the readable label when free, else the same label + the filename's hash tag.
    DB-backed so it stays unique across resumable runs too."""
    tag = Path(filename).stem.rsplit("-", 1)[-1]
    cand, i = base, 1
    while True:
        row = con.execute("SELECT filename FROM statutes WHERE statute_id=?", (cand,)).fetchone()
        if row is None or row[0] == filename:
            return cand
        i += 1
        cand = f"{base} [{tag}]" if i == 2 else f"{base} [{tag}-{i}]"


def _pdf_queue(directory: Path, max_mb: float | None) -> list[Path]:
    """Ingest queue: smallest first, so the bulk of a corpus lands quickly
    instead of waiting behind a handful of 100-700 MB scanned volumes whose
    page-by-page OCR takes hours each. `max_mb` defers the giants entirely —
    re-run without it (e.g. overnight) to sweep them up."""
    pdfs = sorted(directory.glob("*.pdf"), key=lambda p: p.stat().st_size)
    if max_mb:
        big = [p for p in pdfs if p.stat().st_size > max_mb * 1e6]
        if big:
            pdfs = pdfs[: len(pdfs) - len(big)]
            print(f"  deferring {len(big)} PDFs larger than {max_mb:g} MB", flush=True)
    return pdfs


def build_statutes(limit: int | None = None, force: bool = False,
                   max_mb: float | None = None) -> None:
    """Index downloaded lankalaw.net Acts/statutes (source='statute'). Each gets a
    canonical label (the chunk case_no) so the breakdown's "Legislation Cited"
    links can resolve to the actual statute text via store.resolve_statute()."""
    con = store.init_db()
    meta_by_file = load_lankalaw_manifest()
    pdfs = _pdf_queue(STATUTES_DIR, max_mb)
    if limit:
        pdfs = pdfs[:limit]

    done = skipped = failed = 0
    labels: list[str] = []
    print(f"Indexing {len(pdfs)} statutes into {store.COLLECTION} (resume={not force}) …", flush=True)
    for p in pdfs:
        if not force and store.is_indexed(con, p.name, "statute"):
            skipped += 1
            continue
        try:
            meta = dict(meta_by_file.get(p.name, {}))
            title = _statute_title(meta, p)
            label = _unique_statute_id(
                con, store.statute_label(title, meta.get("act_no", ""), meta.get("year", "")), p.name)
            chunks = parse_and_chunk(p, label, source="statute")
            store.add_chunks(chunks, extra_meta={
                "year": meta.get("year", ""), "act_no": meta.get("act_no", ""),
                "filename": p.name, "kind": "statute",
            })
            store.upsert_statute(con, {
                "statute_id": label, "title": title, "act_no": meta.get("act_no", ""),
                "year": meta.get("year", ""), "kind": meta.get("kind", "statute"),
                "filename": p.name, "local_path": str(p), "pdf_url": meta.get("pdf_url", ""),
                "source_url": meta.get("source_page", ""),
            }, len(chunks))
            store.mark_indexed(con, p.name, "statute", len(chunks))
            labels.append(label)
            done += 1
            if done % 25 == 0:
                print(f"  …{done} indexed ({skipped} skipped, {failed} failed)", flush=True)
        except Exception as e:  # noqa: BLE001 — one bad PDF must not kill the run
            failed += 1
            print(f"  SKIP {p.name}: {type(e).__name__}: {e}", flush=True)
    # Make the new statutes keyword-searchable (incremental FTS, like update_corpus).
    if labels:
        try:
            added = store.fts_index_cases(labels)
            print(f"  FTS: indexed {added} statute chunks.", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  FTS index skipped: {e}", flush=True)
    print(f"Statutes done: {done} indexed, {skipped} already-done, {failed} failed.", flush=True)


# A few PDFs land on case pages but are really Acts/bills ("34-2024_E"),
# gazettes ("No. 2091/03 dated 01.10.2018") or Act write-ups — not judgments.
_NON_CASE_TITLE = re.compile(
    r"^\d{1,3}-\d{4}_[A-Z]$"
    r"|^No\.\s*\d{3,4}/\d+"
    r"|\bAct\b.*No\.\s*\d+\s+of\s+(?:18|19|20)\d{2}"
)


def clean_case_title(title: str, filename: str) -> str:
    """lankalaw NLR titles are often hyphen-packed filenames
    ('MAYOR-OF-GALLE-v.-THE-ESTATE….pdf') — turn them back into case names."""
    t = (title or "").strip()
    if t.lower().endswith(".pdf"):
        t = t[:-4]
    if "-" in t and " " not in t:
        t = t.replace("-", " ")
    t = re.sub(r"\s+", " ", t.replace("_", " ")).strip(" .,-")
    return t or case_no_from_filename(filename)


# --- law-report case display names + decision years ------------------------- #
_HON = r"(?:mrs?|miss|ms|dr|rev|hon|sir|lady|col|major|capt|lt|inspector|sergeant)\.?"
_PARTY_TRAIL = re.compile(
    r"[\s,]+(?:et\s+al\.?|and\s+(?:an)?others?|and\s+(?:two|three|four|\d+)\s+others?\b.*)\s*$", re.I)
# 'petitoner' = a recurring OCR/site typo on the NLR listing pages.
_ROLES_A = r"(?:appellants?|applicants?|petitione?rs?|petitoners?|plaintiffs?|complainants?)"
_ROLES_B = r"(?:respondents?|defendants?|accused)"


def _party_short(seg: str) -> str:
    """'Mrs. R. K. D. GUNAWARDENA' -> 'Gunawardena'; drops honorifics, initials,
    role words, and 'et al'/'and others'/'and another' trails. Deliberately does
    NOT split on a bare ' and ' — institutional names contain it."""
    seg = (seg or "").strip(" .,;")
    seg = re.sub(rf"[\s,]+(?:{_ROLES_A}|{_ROLES_B})\.?\s*$", "", seg, flags=re.I)
    seg = _PARTY_TRAIL.sub("", seg).strip(" .,;")
    toks = [t for t in seg.split()
            if not re.fullmatch(_HON, t, re.I) and not re.fullmatch(r"[A-Za-z]\.?", t)]
    out = " ".join(toks).strip(" .,;") or seg
    return out.title() if out.isupper() else out


def short_case_name(title: str) -> str:
    """Law-report style short name: 'A. A. GUNAWARDENA Appellant and Mrs. R. K. D.
    GUNAWARDENA Respondent' -> 'Gunawardena v. Gunawardena'. Titles that match
    neither the role form nor an existing 'X v. Y' are returned (title-cased)."""
    t = re.sub(r"\s+", " ", (title or "")).strip()
    m = re.search(rf"^(?P<a>.+?)[,.]?\s+{_ROLES_A}\b.*?\band\b\s+(?P<b>.+?)[,.]?\s+{_ROLES_B}\b",
                  t, re.I)
    if not m:  # role named but the respondent role got truncated off the listing
        m = re.search(rf"^(?P<a>.+?)[,.]?\s+{_ROLES_A}\b[,.]?\s+and\s+(?P<b>.+)$", t, re.I)
    if not m:
        m = re.search(r"^(?P<a>.+?)\s+v[s]?[.,]?\s*(?P<b>.+)$", t, re.I)
    if not m:
        return t.title() if t.isupper() else t
    return f"{_party_short(m.group('a'))} v. {_party_short(m.group('b'))}"


def case_year(report_cite: str, text: str) -> str:
    """Decision year for a law-report case. SLR cites carry the report year; NLR
    years come from the judgment text (headnotes open with e.g. '1969 Present:'),
    sanity-bounded to the volume's era (vol->year is near-linear, ±8) so a cited
    older authority's year can't win. Falls back to the era estimate itself."""
    if (report_cite or "").startswith("SLR "):
        return report_cite[4:]
    est = None
    m = re.match(r"(\d{1,3}) NLR$", report_cite or "")
    if m:
        est = 1893 + round(int(m.group(1)) * 1.07)
    yrs = [int(y) for y in re.findall(r"\b(1[89]\d\d|20[0-2]\d)\b", (text or "")[:4000])
           if 1850 <= int(y) <= 2026 and (est is None or abs(int(y) - est) <= 8)]
    if not yrs:
        return str(est) if est else ""
    from collections import Counter
    best = max(Counter(yrs).items(), key=lambda kv: (kv[1], -yrs.index(kv[0])))
    return str(best[0])


def _unique_case_no(con, base: str, filename: str) -> str:
    """Chunks tie to a judgement by case_no, so two 'Fernando v. Fernando (45 NLR)'
    files must not share one — DB-backed, like _unique_statute_id."""
    tag = Path(filename).stem.rsplit("-", 1)[-1][:6]
    cand, i = base, 1
    while True:
        row = con.execute("SELECT filename FROM judgements WHERE case_no=?", (cand,)).fetchone()
        if row is None or row[0] == filename:
            return cand
        i += 1
        cand = f"{base} [{tag}]" if i == 2 else f"{base} [{tag}-{i}]"


def report_cite_from_category(category: str) -> str:
    """Canonical law-report citation from the source listing page:
    'new-law-reports-volume-68' -> '68 NLR'; 'sri-lanka-law-reports-1982' -> 'SLR 1982'."""
    m = re.match(r"new-law-reports-volume-(\d+)", category or "")
    if m:
        return f"{int(m.group(1))} NLR"
    m = re.match(r"sri-lanka-law-reports-(\d{4})", category or "")
    if m:
        return f"SLR {m.group(1)}"
    return ""


def build_lankalaw_cases(limit: int | None = None, force: bool = False,
                         max_mb: float | None = None) -> None:
    """Index downloaded lankalaw.net judgement PDFs as ordinary judgments
    (source='judgment'); the content dedupe then merges any that overlap the
    supremecourt.lk corpus (some reuse the same PDF / case). Law-report cases
    (NLR/SLR) also get a report_cite so precedent citations can resolve to them."""
    con = store.init_db()
    meta_by_file = load_lankalaw_manifest()
    pdfs = _pdf_queue(LANKALAW_CASES_DIR, max_mb)
    if limit:
        pdfs = pdfs[:limit]

    done = skipped = failed = non_case = 0
    labels: list[str] = []
    print(f"Indexing {len(pdfs)} lankalaw cases into {store.COLLECTION} (resume={not force}) …", flush=True)
    for p in pdfs:
        if not force and store.is_indexed(con, p.name, "judgment"):
            skipped += 1
            continue
        try:
            meta = dict(meta_by_file.get(p.name, {}))
            if _NON_CASE_TITLE.search(meta.get("title") or ""):
                non_case += 1
                continue
            title = clean_case_title(meta.get("title", ""), p.name)
            report_cite = report_cite_from_category(meta.get("category", ""))
            short = short_case_name(title)
            case_no = _unique_case_no(
                con, f"{short} ({report_cite})" if report_cite else short, p.name)
            chunks = parse_and_chunk(p, case_no)
            year = (meta.get("year") or "").strip()
            if not (year.isdigit() and 1800 <= int(year) <= 2030):
                year = case_year(report_cite, " ".join(c.text for c in chunks[:3]))
            store.add_chunks(chunks, extra_meta={"date": year, "filename": p.name})
            store.upsert_judgement(con, {
                "filename": p.name, "case_no": case_no, "local_path": str(p),
                "date": year, "pdf_url": meta.get("pdf_url", ""),
                "report_cite": report_cite, "parties": short,
            }, len(chunks))
            store.mark_indexed(con, p.name, "judgment", len(chunks))
            labels.append(case_no)
            done += 1
            if done % 25 == 0:
                print(f"  …{done} indexed ({skipped} skipped, {failed} failed)", flush=True)
            if len(labels) >= 500:  # keep FTS current across long resumable runs
                try:
                    store.fts_index_cases(labels)
                finally:
                    labels = []
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  SKIP {p.name}: {type(e).__name__}: {e}", flush=True)
    if labels:
        try:
            added = store.fts_index_cases(labels)
            print(f"  FTS: indexed {added} case chunks.", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  FTS index skipped: {e}", flush=True)
    print(f"Lankalaw cases done: {done} indexed, {skipped} already-done, "
          f"{failed} failed, {non_case} non-case PDFs left out.", flush=True)
    if done:
        from .dedupe import dedupe_pass

        rep = dedupe_pass(con, apply=True, quiet=True)
        if rep.get("rows_to_drop"):
            print(f"Dedupe: merged {rep['rows_to_drop']} duplicate rows into "
                  f"{rep['duplicate_clusters']} cases.", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the ROScribe index (resumable).")
    ap.add_argument("--notes", type=str, default=None, help="index a personal-notes folder")
    ap.add_argument("--statutes", action="store_true", help="index downloaded lankalaw statutes")
    ap.add_argument("--lankalaw-cases", action="store_true", help="index downloaded lankalaw cases")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-index even if already done")
    ap.add_argument("--max-mb", type=float, default=None,
                    help="defer PDFs larger than this (sweep them up in a later run)")
    args = ap.parse_args(argv)
    if args.notes:
        build_notes(args.notes, args.limit, args.force)
    elif args.statutes:
        build_statutes(args.limit, args.force, args.max_mb)
    elif args.lankalaw_cases:
        build_lankalaw_cases(args.limit, args.force, args.max_mb)
    else:
        build_judgements(args.limit, args.force)


if __name__ == "__main__":
    main()
