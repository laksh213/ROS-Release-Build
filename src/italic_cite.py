"""Typography-scoped citation extraction (v2).

Law reports set case-name citations in *italics* — "*Faris v The Officer-In-Charge,
Police Station … [1992] 1 SLR 167*" — and italicise Latin terms (*inter alia*,
*ratio decidendi*). Isolating citations by **typography** is far higher-precision
than scanning prose for the word "v" (which also appears in ordinary sentences).

Why PyMuPDF and not Docling: on the Supreme Court corpus, Docling's `docling-parse`
backend returns an empty `formatting` field (italic = None) for every document, so
its italic channel yields nothing here. PyMuPDF exposes per-span font **flags** and
**font names** (e.g. ``BookmanOldStyle-Italic``), which DO carry the italic styling
of these judgments — so this module reads italics straight from the font layer.

The extracted citations feed `store.resolve_citation` to link each one to a corpus
case (internal-first; web-search fallback in the UI for out-of-corpus citations).

CLI:
  python -m src.italic_cite data/sc_judgements/<file>.pdf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import fitz  # PyMuPDF

# PyMuPDF span "flags" is a bitfield; bit 1 (value 2) = italic, bit 4 (16) = bold.
_ITALIC_FLAG = 2

# A law-report / neutral citation closes a case citation: "[1992] 1 SLR 167",
# "(2001) ...", "73 NLR 121", SC/CA neutral numbers. Used to recognise and to
# stop greedy line-stitching once a citation is complete.
_REPORTER_RE = re.compile(
    r"\[(?:19|20)\d{2}\]"                                  # [1992]
    r"|\((?:19|20)\d{2}\)"                                 # (1992)
    r"|\b\d{1,3}\s+(?:SLR|N\.?L\.?R|C\.?L\.?W|Bar\.?\s*R|SCC|App\.?\s*Cas)\b"  # 73 NLR 121
    r"|\b(?:SC|CA|HC)[/\s.]",                               # neutral SC/CA refs
    re.IGNORECASE,
)

# The adversarial party separator, as a standalone token ("Silva v Perera",
# "A vs B", "X versus Y") — never inside a word.
_VS_INLINE = re.compile(r"(?<!\w)[Vv][Ss]?\.?(?!\w)|(?<!\w)versus(?!\w)")

# Italic runs that are pure Latin/legal terms, not citations — excluded from the
# citation list (still available via extract_italic_runs if a caller wants them).
_LATIN_TERMS = {
    "inter alia", "in toto", "ratio decidendi", "obiter dicta", "obiter dictum",
    "prima facie", "ultra vires", "mutatis mutandis", "ex parte", "suo motu",
    "bona fide", "mala fide", "stare decisis", "res judicata", "locus standi",
    "audi alteram partem", "ipso facto", "in limine", "per se", "de novo",
}


def _span_is_italic(span: dict) -> bool:
    """True if a PyMuPDF span is italic — by flag bit OR by font name, since some
    PDFs encode the slant only in the font name (``...,Italic`` / ``-Oblique``)."""
    if span.get("flags", 0) & _ITALIC_FLAG:
        return True
    font = (span.get("font") or "").lower()
    return "italic" in font or "oblique" in font


def extract_italic_runs(pdf_path: str | Path, max_pages: int | None = None) -> list[tuple[str, int]]:
    """Contiguous italic text runs (adjacent italic spans on a line merged), each
    with its 1-based page number. The raw typographic signal other extractors build on."""
    runs: list[tuple[str, int]] = []
    with fitz.open(str(pdf_path)) as doc:
        for pno, page in enumerate(doc, start=1):
            if max_pages and pno > max_pages:
                break
            for blk in page.get_text("dict").get("blocks", []):
                for line in blk.get("lines", []):
                    buf: list[str] = []
                    for sp in line.get("spans", []):
                        if _span_is_italic(sp):
                            buf.append(sp.get("text", ""))
                        elif buf:
                            runs.append(("".join(buf).strip(), pno))
                            buf = []
                    if buf:
                        runs.append(("".join(buf).strip(), pno))
    return [(re.sub(r"\s+", " ", t).strip(), p) for t, p in runs if len(t.strip()) >= 2]


def extract_italic_citations(pdf_path: str | Path) -> list[dict]:
    """Italic runs that look like **case citations** — i.e. carry a ``v``/``vs``
    party separator and/or a law-report reference. Consecutive italic runs are
    stitched (citations wrap across lines), Latin terms are excluded, and duplicates
    are merged. Returns ``[{text, page, kind}]`` where kind is "case" (has v/vs) or
    "reporter" (report ref only). Typography-scoped, so the prose word "versus" in a
    sentence is NOT mistaken for a citation."""
    runs = extract_italic_runs(pdf_path)
    out: list[dict] = []
    seen: set[str] = set()
    i = 0
    while i < len(runs):
        text, page = runs[i]
        j = i + 1
        # Stitch following italic runs on the same page until the citation looks
        # complete (a reporter ref appeared) or it gets implausibly long.
        while (j < len(runs) and runs[j][1] == page
               and not _REPORTER_RE.search(text) and len(text) < 180):
            text = f"{text} {runs[j][0]}".strip()
            j += 1
            if _REPORTER_RE.search(runs[j - 1][0]):
                break
        text = re.sub(r"\s+", " ", text).strip(" .,;:")
        low = text.lower()
        has_vs = bool(_VS_INLINE.search(text))
        has_rep = bool(_REPORTER_RE.search(text))
        is_latin = low in _LATIN_TERMS or (len(low.split()) <= 3 and not has_vs and not has_rep)
        if (has_vs or has_rep) and not is_latin and len(text) >= 6:
            key = re.sub(r"[^a-z0-9]", "", low)[:80]
            if key and key not in seen:
                seen.add(key)
                out.append({"text": text, "page": page, "kind": "case" if has_vs else "reporter"})
        i = j if j > i + 1 else i + 1
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Extract italic-scoped citations from a judgment PDF.")
    ap.add_argument("pdf")
    ap.add_argument("--runs", action="store_true", help="also dump every italic run")
    args = ap.parse_args(argv)

    cites = extract_italic_citations(args.pdf)
    print(f"{Path(args.pdf).name}: {len(cites)} italic citation(s)")
    for c in cites:
        print(f"  [p{c['page']} · {c['kind']:8}] {c['text']}")
    if args.runs:
        print("\n-- all italic runs --")
        for t, p in extract_italic_runs(args.pdf):
            print(f"  p{p}: {t}")


if __name__ == "__main__":
    main()
