#!/usr/bin/env python3
"""Continuous QA — verify cached AI breakdowns against the source judgments.

For every row in the SQLite ``analyses`` table this script loads the cached
``CaseAnalysis`` together with the judgment's extracted text and looks for clear,
checkable discrepancies:

  * **Bench**   — the panel recorded in ``metadata.judges`` vs. the panel parsed
                  from the judgment front matter (`ingest.extract_bench`).
  * **Parties** — present in the analysis, and not blanked when the text clearly
                  names the parties.
  * **Final order** — non-trivial, and not blanked when the text clearly carries
                  an operative order ("appeal is allowed/dismissed", …).
  * **Precedents** — every ``precedent_index.cited_case`` should actually appear
                  in the judgment text (catches invented / hallucinated cites).
  * **Placeholder misuse** — "Information not available in source text." sitting
                  on a field the source text plainly provides.

It is read-only and re-runnable: run it any time the cache changes.

    ROSCRIBE_EMBEDDER=default .venv/bin/python -m scripts.verify_breakdowns
    ROSCRIBE_EMBEDDER=default .venv/bin/python scripts/verify_breakdowns.py --case "SC/APPEAL/175/2017"

Exit code is 0 when every breakdown passes, 1 when any case has an ISSUE (so it
can gate CI), regardless of WARN-level notes.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import settings  # noqa: E402
from src.ingest import extract_bench  # noqa: E402
from src.parsing import extract_pages  # noqa: E402
from src.schema import NOT_AVAILABLE, CaseAnalysis  # noqa: E402

JUDGE_DIR = REPO_ROOT / "data" / "sc_judgements"


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for fuzzy matching."""
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _is_blank(s: str) -> bool:
    s = (s or "").strip()
    return not s or s == NOT_AVAILABLE or s in {"—", "-", "N/A", "n/a"}


def _judge_key(name: str) -> str:
    """Surname-ish key so 'Hon. Justice Menaka Wijesundera' ~= 'Menaka Wijesundera, J.'."""
    n = _norm(name)
    for stop in ("hon", "honble", "honourable", "honorable", "justice", "judge",
                 "mr", "mrs", "ms", "dr", "the"):
        n = re.sub(rf"\b{stop}\b", " ", n)
    n = re.sub(r"\b(c j|a c j|d c j|j|pc|qc)\b", " ", n)  # drop suffixes
    toks = [t for t in n.split() if len(t) > 1]
    return " ".join(sorted(toks))


# Operative-order language: if the text has this but final_order is blank, flag.
_ORDER_SIGNALS = re.compile(
    r"\b(?:appeal|application|petition)\s+is\s+(?:hereby\s+)?(?:allowed|dismissed|"
    r"refused|rejected|granted)\b"
    r"|\bwe\s+(?:allow|dismiss|set\s+aside|affirm|hold|order|direct)\b"
    r"|\bjudgment\s+(?:is\s+)?(?:affirmed|set\s+aside|reversed|varied)\b"
    r"|\b(?:conviction|sentence)\s+is\s+(?:affirmed|set\s+aside|quashed)\b"
    r"|\bproxy\s+revoked\b|\bno\s+order\s+as\s+to\s+costs\b",
    re.IGNORECASE,
)

# Party-naming signals in the text.
_PARTY_SIGNALS = re.compile(
    r"\b(?:vs?\.?|versus|petitioner|respondent|appellant|plaintiff|defendant|accused)\b",
    re.IGNORECASE,
)


def _cited_in_text(cited: str, norm_text: str) -> bool:
    """Is this cited case plausibly present in the judgment text?

    Tolerant of citation-format drift: we accept a match on the whole normalised
    cite, on its 'Party v Party' core, or on a distinctive 4+-word run of it.
    """
    nc = _norm(cited).strip()
    if not nc:
        return True  # nothing to check
    if nc in norm_text:
        return True

    # Core before any reporter citation / parentheses, e.g.
    # "Silva v Perera (1998) 1 SLR 23" -> "silva v perera"
    core = re.split(r"\b(?:19|20)\d{2}\b|\(", nc, maxsplit=1)[0].strip()
    if len(core) >= 6 and core in norm_text:
        return True

    # Either party name appearing on its own (>= 5 chars to avoid noise).
    for side in re.split(r"\bv(?:s|ersus)?\b", core):
        side = side.strip()
        if len(side) >= 5 and side in norm_text:
            return True

    # A distinctive consecutive run of words from the cite.
    toks = nc.split()
    for n in (6, 5, 4):
        if len(toks) >= n:
            for i in range(len(toks) - n + 1):
                run = " ".join(toks[i:i + n])
                if len(run) >= 12 and run in norm_text:
                    return True
            break
    return False


# --------------------------------------------------------------------------- #
# per-case check                                                               #
# --------------------------------------------------------------------------- #
def verify_case(case_no: str, raw_json: str, con: sqlite3.Connection) -> tuple[list[str], list[str]]:
    """Return (issues, warns) for one cached breakdown."""
    issues: list[str] = []
    warns: list[str] = []

    # --- load the analysis -------------------------------------------------- #
    try:
        data = json.loads(raw_json)
        ca = CaseAnalysis.model_validate(data)
    except Exception as e:  # noqa: BLE001
        return [f"cached JSON does not validate against CaseAnalysis: {e}"], warns

    # --- locate + read the judgment ---------------------------------------- #
    row = con.execute(
        "SELECT filename, local_path, judges, parties FROM judgements "
        "WHERE case_no=? OR filename=? LIMIT 1",
        (case_no, case_no),
    ).fetchone()
    if not row:
        return [f"no judgements row for {case_no!r}; cannot verify against source"], warns
    filename, local_path, scrape_judges_raw, scrape_parties = row

    pdf = None
    for cand in (local_path, str(JUDGE_DIR / filename) if filename else None):
        if cand and Path(cand).exists():
            pdf = cand
            break
    if not pdf:
        return [f"source PDF not found (filename={filename!r}); cannot verify"], warns

    try:
        pages = extract_pages(pdf)
    except Exception as e:  # noqa: BLE001
        return [f"failed to extract text from {pdf}: {e}"], warns
    full_text = "\n".join(pages)
    norm_text = _norm_ws(_norm(full_text))

    try:
        scrape_judges = json.loads(scrape_judges_raw) if scrape_judges_raw else []
    except Exception:
        scrape_judges = []

    # --- 1. bench ----------------------------------------------------------- #
    text_bench = extract_bench(pages)
    ana_judges = ca.metadata.judges or []
    if text_bench:
        text_keys = {_judge_key(j) for j in text_bench}
        ana_keys = {_judge_key(j) for j in ana_judges}
        missing = text_keys - ana_keys
        if not ana_judges:
            issues.append(
                f"bench empty in analysis but the judgment names {len(text_bench)}: "
                f"{text_bench}"
            )
        elif len(ana_judges) < len(text_bench) or missing:
            issues.append(
                f"bench incomplete: analysis has {ana_judges} but the judgment's "
                f"panel is {text_bench}"
            )
        # Cross-check the scrape metadata too (informational).
        if len(scrape_judges) < len(text_bench):
            warns.append(
                f"scrape metadata lists {len(scrape_judges)} judge(s) "
                f"{scrape_judges} vs. {len(text_bench)} on the bench {text_bench} "
                f"— UI should use the text-extracted panel"
            )
    else:
        warns.append("could not parse a bench from the text; relying on metadata")

    # --- 2. parties --------------------------------------------------------- #
    parties = ca.metadata.parties
    if _is_blank(parties):
        if _PARTY_SIGNALS.search(full_text):
            issues.append("parties blank in analysis but the text clearly names parties")
        else:
            warns.append("parties missing from analysis metadata")

    # --- 3. final order ----------------------------------------------------- #
    fo = ca.final_order
    if _is_blank(fo):
        if _ORDER_SIGNALS.search(full_text):
            issues.append(
                "final_order blank but the judgment contains an operative order "
                "(e.g. 'appeal is allowed/dismissed')"
            )
        else:
            warns.append("final_order is blank / not available")
    elif len(_norm_ws(fo)) < 15:
        warns.append(f"final_order looks trivially short: {fo!r}")

    # --- 4. precedents present in the text ---------------------------------- #
    for p in ca.precedent_index:
        cited = (p.cited_case or "").strip()
        if not cited or cited == NOT_AVAILABLE:
            continue
        if not _cited_in_text(cited, norm_text):
            issues.append(f"cited precedent not found in judgment text: {cited!r}")

    # --- 5. ratio / factual placeholders where text clearly has content ----- #
    if _is_blank(ca.ratio_decidendi) and len(full_text) > 4000:
        warns.append("ratio_decidendi blank on a substantial judgment — review")
    if _is_blank(ca.factual_matrix) and len(full_text) > 4000:
        warns.append("factual_matrix blank on a substantial judgment — review")

    return issues, warns


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def purge_hollow(con: sqlite3.Connection) -> int:
    """Delete cached breakdowns that are hollow (mostly placeholders) or no
    longer parse against CaseAnalysis, so the quality gate regenerates them on
    next view. Re-runnable maintenance — run after model/prompt changes or a
    corpus update. Returns the number of rows purged."""
    rows = con.execute("SELECT case_no, json FROM analyses ORDER BY case_no").fetchall()
    doomed: list[tuple[str, str]] = []
    for case_no, raw_json in rows:
        try:
            ca = CaseAnalysis.model_validate(json.loads(raw_json))
        except Exception as e:  # noqa: BLE001 — unparseable cache is junk too
            doomed.append((case_no, f"does not validate: {e}"))
            continue
        q = ca.quality()
        if q["hollow"]:
            doomed.append((case_no, f"hollow ({q['filled']}/{q['total']} core fields filled)"))
    print(f"Scanned {len(rows)} cached breakdown(s) — {len(doomed)} to purge.")
    for case_no, why in doomed:
        print(f"  purge {case_no}: {why}")
    if doomed:
        con.executemany("DELETE FROM analyses WHERE case_no=?", [(cn,) for cn, _ in doomed])
        con.commit()
        print(f"Purged {len(doomed)} — they will regenerate (quality-gated) on next view.")
    return len(doomed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify cached AI breakdowns vs. source judgments.")
    ap.add_argument("--case", help="verify only this case_no (default: all rows in analyses)")
    ap.add_argument("--db", default=settings.sqlite_path, help="path to roscribe.db")
    ap.add_argument("--purge-hollow", action="store_true",
                    help="delete hollow/unparseable cached breakdowns (they regenerate on next view) and exit")
    args = ap.parse_args(argv)

    con = sqlite3.connect(args.db)
    if args.purge_hollow:
        purge_hollow(con)
        con.close()
        return 0
    if args.case:
        rows = con.execute(
            "SELECT case_no, json FROM analyses WHERE case_no=?", (args.case,)
        ).fetchall()
    else:
        rows = con.execute("SELECT case_no, json FROM analyses ORDER BY case_no").fetchall()

    if not rows:
        print("No cached breakdowns found in 'analyses'. Nothing to verify.")
        con.close()
        return 0

    n_pass = n_issue = 0
    total_issues = total_warns = 0
    print(f"Verifying {len(rows)} cached breakdown(s) against source judgments\n" + "=" * 64)
    for case_no, raw_json in rows:
        issues, warns = verify_case(case_no, raw_json, con)
        total_issues += len(issues)
        total_warns += len(warns)
        if issues:
            n_issue += 1
            print(f"\n[ISSUE] {case_no}")
        else:
            n_pass += 1
            print(f"\n[PASS]  {case_no}")
        for msg in issues:
            print(f"    ✗ {msg}")
        for msg in warns:
            print(f"    ! {msg}")

    con.close()
    print("\n" + "=" * 64)
    print(
        f"Summary: {n_pass} passed, {n_issue} with issues "
        f"({total_issues} issue(s), {total_warns} warning(s)) "
        f"across {len(rows)} breakdown(s)."
    )
    return 1 if n_issue else 0


if __name__ == "__main__":
    raise SystemExit(main())
