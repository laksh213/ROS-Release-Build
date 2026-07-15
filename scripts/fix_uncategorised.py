"""One-time repair of uncategorised/undated corpus rows.

1. Undated SC cases: decision date from the judgment text — regex battery
   first, local GGUF model fallback for stragglers.
2. Junk case names ('695CCDE778E92 9R3'): rebuild 'X v. Y' from the parties
   field or caption text; local model fallback. Renames sync chunks_fts.
3. Missing authoring judges (SLR opener style) via the improved extractor.

Run:  .venv/bin/python -u scripts/fix_uncategorised.py
"""
from __future__ import annotations
import json, re, sqlite3, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import settings                                   # noqa: E402
from src.ingest import extract_opinion_author                     # noqa: E402
from src.index import short_case_name                             # noqa: E402

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_D1 = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTHS})[,.]?\s+(\d{{4}})", re.I)
_D2 = re.compile(rf"({MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?[,.]?\s+(\d{{4}})", re.I)
_D3 = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")
_MN = {m.lower(): i + 1 for i, m in enumerate(MONTHS.split("|"))}


def date_from_text(t: str) -> str:
    t = t[:6000]
    best = []
    for m in _D1.finditer(t):
        d, mo, y = int(m.group(1)), _MN[m.group(2).lower()], int(m.group(3))
        best.append((y, mo, d, "decided" in t[max(0, m.start() - 60):m.start()].lower()))
    for m in _D2.finditer(t):
        mo, d, y = _MN[m.group(1).lower()], int(m.group(2)), int(m.group(3))
        best.append((y, mo, d, "decided" in t[max(0, m.start() - 60):m.start()].lower()))
    for m in _D3.finditer(t):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            best.append((y, mo, d, False))
    best = [b for b in best if 1900 <= b[0] <= 2026 and 1 <= b[1] <= 12 and 1 <= b[2] <= 31]
    if not best:
        return ""
    pick = next((b for b in best if b[3]), None) or max(best)  # 'Decided on' wins, else latest date
    return f"{pick[0]:04d}-{pick[1]:02d}-{pick[2]:02d}"


_LLM = None


def llm_ask(prompt: str) -> str:
    global _LLM
    if _LLM is None:
        from llama_cpp import Llama
        _LLM = Llama(model_path=settings.llamacpp_model_path, n_ctx=4096,
                     n_gpu_layers=settings.llamacpp_gpu_layers, verbose=False)
    r = _LLM.create_chat_completion(messages=[{"role": "user", "content": prompt}],
                                    max_tokens=40, temperature=0.0)
    return r["choices"][0]["message"]["content"].strip()


def main() -> None:
    con = sqlite3.connect(settings.sqlite_path)

    def txt(cn, n=4):
        return " ".join(t for (t,) in con.execute(
            "SELECT text FROM chunks_fts WHERE case_no=? LIMIT ?", (cn, n)))

    # --- 1. dates ---------------------------------------------------------
    rows = con.execute("SELECT case_no FROM judgements WHERE COALESCE(report_cite,'')=''"
                       " AND COALESCE(date,'')=''").fetchall()
    dated = llm_dated = 0
    for (cn,) in rows:
        t = txt(cn)
        d = date_from_text(t)
        if not d and t:
            a = llm_ask("Text of a court judgment follows. Reply with ONLY the decision "
                        "date as YYYY-MM-DD, or NONE if absent.\n\n" + t[:3000])
            if re.fullmatch(r"(19|20)\d{2}-\d{2}-\d{2}", a):
                d, llm_dated = a, llm_dated + 1
        if d:
            con.execute("UPDATE judgements SET date=? WHERE case_no=?", (d, cn))
            dated += 1
    con.commit()
    print(f"dates: {dated}/{len(rows)} fixed ({llm_dated} via local model)", flush=True)

    # --- 2. junk case names -------------------------------------------------
    rows = con.execute("SELECT case_no, COALESCE(parties,'') FROM judgements "
                       "WHERE COALESCE(report_cite,'')='' AND case_no NOT LIKE '%v%' "
                       "AND case_no NOT LIKE 'SC%' AND case_no NOT LIKE '%Appeal%'").fetchall()
    used = {c for (c,) in con.execute("SELECT case_no FROM judgements")}
    junk = {cn for cn, _ in rows}
    fts_map: dict[str, list[int]] = {}
    for rowid, cn in con.execute("SELECT rowid, case_no FROM chunks_fts"):
        if cn in junk:
            fts_map.setdefault(cn, []).append(rowid)
    renamed = llm_named = 0
    for cn, parties in rows:
        new = ""
        if parties and " v" in parties.lower():
            new = short_case_name(parties)
        if not new or " v. " not in new:
            t = txt(cn)
            m = re.search(r"([A-Z][A-Za-z .'’&()-]{2,50}?)\s+[vV][s]?\.?\s+([A-Z][A-Za-z .'’&()-]{2,60})", t)
            if m:
                new = short_case_name(f"{m.group(1)} v. {m.group(2)}")
            elif t:
                a = llm_ask("From this judgment text, reply ONLY with the case name as "
                            "'X v. Y' (party surnames), or NONE.\n\n" + t[:2500])
                if re.fullmatch(r"[A-Z][\w .'’&()-]{1,60} v\.? [A-Z][\w .'’&()-]{1,60}", a):
                    new, llm_named = short_case_name(a), llm_named + 1
        new = (new or "").strip()
        if not new or len(new) < 8 or " v. " not in new or new == cn:
            continue
        if new in used:
            new = f"{new} [{abs(hash(cn)) % 100000}]"
        con.execute("UPDATE judgements SET case_no=? WHERE case_no=?", (new, cn))
        for t_ in ("bookmarks", "annotations", "case_judges", "analyses"):
            con.execute(f"UPDATE {t_} SET case_no=? WHERE case_no=?", (new, cn))
        for rid in fts_map.get(cn, []):
            con.execute("UPDATE chunks_fts SET case_no=? WHERE rowid=?", (new, rid))
        used.discard(cn)
        used.add(new)
        renamed += 1
    con.commit()
    print(f"junk names: {renamed}/{len(rows)} renamed ({llm_named} via local model)", flush=True)

    # --- 3. missing authors (SLR opener style) ------------------------------
    rows = con.execute("SELECT case_no FROM judgements WHERE report_cite LIKE 'SLR%' "
                       "AND (judges IS NULL OR judges IN ('','[]'))").fetchall()
    authored = 0
    for (cn,) in rows:
        a = extract_opinion_author("\n".join(t for (t,) in con.execute(
            "SELECT text FROM chunks_fts WHERE case_no=? LIMIT 2", (cn,))))
        if a:
            con.execute("UPDATE judgements SET judges=? WHERE case_no=?", (json.dumps([a]), cn))
            authored += 1
    con.commit()
    print(f"authors: {authored}/{len(rows)} SLR rows filled", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
