"""Phase 2 — Text extraction & chunking (open source).

Judgements: PyMuPDF (fitz) extracts the text layer fast; if a page has too little
text (scanned), fall back to Tesseract OCR (`eng+sin+tam`). Chunks keep page +
paragraph anchors so citations `[Case No | Page:Para]` stay verifiable.

Personal repository: `load_personal_repo` walks your notes folder (PDF / docx /
txt / md / html), tagging each chunk with Subject and Category derived from the
`NN - Subject / Category / file` folder layout.

CLI:
  python -m src.ingest data/sc_judgements/<file>.pdf --out data/extracted/<file>.md
"""

from __future__ import annotations

import argparse
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF

Source = Literal["judgment", "personal_repo", "statute"]

NOTE_EXTS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}


@dataclass
class Chunk:
    text: str
    case_no: str
    page: int
    para: str | None = None
    source: Source = "judgment"
    metadata: dict = field(default_factory=dict)  # e.g. {"subject", "category"}

    def anchor(self) -> str:
        if self.para is None:
            return f"[{self.case_no} | p{self.page}]"
        return f"[{self.case_no} | {self.page}:{self.para}]"


# --------------------------------------------------------------------------- #
# Extraction                                                                  #
# --------------------------------------------------------------------------- #
def _ocr_page(page: "fitz.Page", langs: str) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    try:
        pix = page.get_pixmap(dpi=300)
        return pytesseract.image_to_string(Image.open(io.BytesIO(pix.tobytes("png"))), lang=langs)
    except Exception:
        return ""  # tesseract binary missing or OCR failed — keep the text layer


def extract_pages(pdf_path: str, ocr_langs: str = "eng+sin+tam", ocr_threshold: int = 200) -> list[str]:
    """Per-page text via PyMuPDF; OCR pages below `ocr_threshold` chars (0 = never)."""
    pages: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text("text")
            if len(text.strip()) < ocr_threshold:
                text = _ocr_page(page, ocr_langs) or text
            pages.append(text)
    return pages


def _read_docx(path: str) -> str:
    from docx import Document

    return "\n".join(p.text for p in Document(path).paragraphs if p.text.strip())


def _read_html(path: str) -> str:
    from bs4 import BeautifulSoup

    return BeautifulSoup(Path(path).read_text(errors="ignore"), "html.parser").get_text(" ", strip=True)


def extract_document(path: str) -> list[str]:
    """Return text 'pages' for any supported note format (no OCR — speed)."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_pages(path, ocr_threshold=0)
    if ext == ".docx":
        return [_read_docx(path)]
    if ext in {".txt", ".md"}:
        return [Path(path).read_text(errors="ignore")]
    if ext in {".html", ".htm"}:
        return [_read_html(path)]
    return []


# --------------------------------------------------------------------------- #
# Chunking                                                                     #
# --------------------------------------------------------------------------- #
def _split_paragraphs(text: str, max_chars: int = 1200) -> list[str]:
    pieces: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if len(block) <= max_chars:
            pieces.append(block)
        else:
            pieces.extend(line.strip() for line in block.split("\n") if line.strip())

    out: list[str] = []
    buf = ""
    for p in pieces:
        cand = f"{buf} {p}".strip() if buf else p
        if len(cand) <= max_chars:
            buf = cand
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def chunk_pages(pages: list[str], case_no: str, source: Source = "judgment") -> list[Chunk]:
    chunks: list[Chunk] = []
    for pno, text in enumerate(pages, start=1):
        for i, para in enumerate(_split_paragraphs(text), start=1):
            chunks.append(Chunk(text=para, case_no=case_no, page=pno, para=str(i), source=source))
    return chunks


def case_no_from_filename(name: str) -> str:
    return Path(name).stem.replace("_", " ").upper()


# --------------------------------------------------------------------------- #
# Bench (the coram / panel of judges)                                          #
# --------------------------------------------------------------------------- #
# The web scrape only records the *authoring* judge. The full panel that heard
# the appeal is printed in the judgment's front matter, introduced by a marker
# like "Before :", "BEFORE", "Coram", or "Present :", followed by 1–3 names that
# each end in a judicial suffix (J. / C.J. / J., PC / etc.). `extract_bench`
# parses that panel straight from the text so the UI can show every judge.

# Markers that introduce the panel. Matched only at the start of a line so a
# stray "before the court" mid-sentence never triggers extraction.
_BENCH_MARKER = re.compile(
    r"^[\s>*•\-]*"
    r"(?:(?:1[89]|20)\d\d\s+)?"   # NLR/SLR headnote form: '1978 Present: X, J., …'
    r"(?:BEFORE|CORAM|PRESENT|BENCH|QUORUM)"
    r"\s*[:\-–.]*\s*",
    re.IGNORECASE,
)

# A line clearly belonging to a *different* labelled section — stop collecting
# the panel once we hit one of these.
_BENCH_STOP = re.compile(
    r"^[\s>*•\-]*"
    r"(?:COUNSEL|ARGUED|DECIDED|DELIVERED|JUDG(?:E)?MENT|JUDGEMENT|ORDER|FOR\s+THE|"
    r"PETITIONER|RESPONDENT|APPELLANT|PLAINTIFF|DEFENDANT|ON\s+BEHALF|"
    r"WRITTEN\s+SUBMISSION|DATE\s+OF|S\.?C\.?\s|C\.?A\.?\s|IN\s+THE\s+MATTER)",
    re.IGNORECASE,
)

# A judicial suffix at the end of a candidate name: J. / J / C.J. / CJ /
# J., PC / PC, J. / ACJ / DCJ … (tolerant of spaces, optional trailing dot).
_JUDGE_SUFFIX = re.compile(
    r"(?:,?\s*(?:P\.?C\.?|Q\.?C\.?|PC|QC))?"      # optional silk: PC / QC
    r"\s*,?\s*"
    r"(?:C\.?\s*J\.?|A\.?\s*C\.?\s*J\.?|D\.?\s*C\.?\s*J\.?|"  # CJ / ACJ / DCJ
    r"CHIEF\s+JUSTICE|JUSTICE|J\.?)"                          # 'Chief Justice'/'Justice' spelled out, or J
    r"\s*$",
    re.IGNORECASE,
)

# Honorifics / titles to strip from the front of a name (keep the suffix).
_HONORIFIC_PREFIX = re.compile(
    r"^(?:the\s+)?"
    r"(?:hon(?:'?ble|ourable|orable)?\.?\s*)?"
    r"(?:(?:mr|mrs|ms|dr)\.?\s*)?"
    r"(?:justice\s+|judge\s+)?",
    re.IGNORECASE,
)


# A concurrence phrase fused into a candidate line ("… , J. I agree") — Docling's
# reading order can merge a judge's name with their sign-off into ONE text item.
# Such a line is a signature, not a coram name (the judge already appears in the
# front-matter panel / their own role line), so it must never be read as a judge.
_CONCURRENCE_IN_NAME = re.compile(
    r"\bI\s+(?:respectfully\s+)?(?:agree|concur|have\s+read)\b", re.IGNORECASE
)


# OCR of old law reports letter-spaces names ('V Y T H IA L IN G A M, J.').
# A run of 5+ short, dot-less tokens is letter-spacing, never real initials
# (those are dotted and shorter, e.g. 'A. H. M. D. Nawaz').
_SPACED_NAME_RUN = re.compile(r"\b(?:[A-Za-z]{1,2} +){4,}[A-Za-z]{1,2}\b")


def _collapse_spaced_name(s: str) -> str:
    def _fuse(m):
        w = m.group(0).replace(" ", "")
        return w.title() if sum(c.isupper() for c in w) >= len(w) * 0.6 else w
    return _SPACED_NAME_RUN.sub(_fuse, s or "")


def _clean_judge(raw: str) -> str | None:
    """Normalise one candidate into 'Name, J.'-style, or None if it isn't a judge.

    Accepts BOTH the suffix form ("Mahinda Samayawardhena, J.") and the modern
    prefix form ("Hon. Justice Mahinda Samayawardhena" — title up front, no
    trailing suffix), which recent SC judgments use for the coram."""
    name = raw.strip(" \t\r\n.,;:·•*->–—")
    name = _collapse_spaced_name(re.sub(r"\s+", " ", name)).strip()
    if not name:
        return None
    # "Wanasundera, J. I agree" (name + sign-off fused by layout analysis) is a
    # signature line, not a coram entry — reject it rather than emit a mangled name.
    if _CONCURRENCE_IN_NAME.search(name):
        return None
    # Drop a leading numbering like "1." or "(i)".
    name = re.sub(r"^\(?\s*[0-9ivx]+\s*[.)]\s*", "", name, flags=re.IGNORECASE)
    # A leading "Justice"/"Judge" title marks a judge even without a trailing suffix.
    had_title = bool(re.match(
        r"^(?:the\s+)?(?:hon(?:'?ble|ourable|orable)?\.?\s*)?(?:mr|mrs|ms|dr)?\.?\s*(?:justice|judge)\b",
        name, re.IGNORECASE))
    name = _HONORIFIC_PREFIX.sub("", name).strip(" .,")
    has_suffix = bool(_JUDGE_SUFFIX.search(name))
    if not (has_suffix or had_title):
        return None
    # Reject lines that are obviously not a person (too long / sentence-like).
    if len(name) > 70 or name.count(" ") > 8:
        return None
    head = _JUDGE_SUFFIX.sub("", name).strip(" ,.") if has_suffix else name
    if not re.search(r"[A-Za-z]{2,}", head):
        return None
    if not has_suffix:
        # prefix-only judge: require a plausible multi-token name, then normalise.
        if head.count(" ") < 1:
            return None
        name = f"{name}, J."
    return name


def _split_candidates(block: str) -> list[str]:
    """Break a bench block into individual name candidates.

    Names can be newline-separated, joined by 'and' / '&', or comma-separated.
    We split on newlines and 'and'/'&' always; commas only when the following
    fragment still looks like it carries a judicial suffix, so 'Wijeratne, J.'
    (a name + its own suffix) is NOT split apart.
    """
    parts: list[str] = []
    for line in re.split(r"[\n\r]+|\s+&\s+|\s+\band\b\s+", block, flags=re.IGNORECASE):
        line = line.strip()
        if not line:
            continue
        # A line may pack several judges, e.g. "A, J., B, J." (comma-separated)
        # or "A, J.  B, J." (space-separated). Insert a split point right after
        # each judicial suffix that is followed by another (capitalised) name,
        # then split on those points. Matching '...J.' / '...J' / '...C.J.'.
        marked = re.sub(
            r"((?:C\.?\s*J|A\.?\s*C\.?\s*J|D\.?\s*C\.?\s*J|J)\.?)"  # a suffix
            r"\s*[,;]?\s+"                                          # gap to next
            r"(?=(?:(?:the\s+)?(?:hon(?:'?ble|ourable|orable)?\.?\s*)?"
            r"(?:justice\s+)?)?[A-Z])",                            # next name starts
            lambda mm: mm.group(1) + "\x00",
            line,
            flags=re.IGNORECASE,
        )
        for seg in marked.split("\x00"):
            seg = seg.strip(" \t,;")
            if seg:
                parts.append(seg)
    return parts


# A panel may share ONE plural suffix — "A, B and C, JJ." — where only the last
# name carries "JJ." and the earlier names have no suffix of their own.
_PLURAL_SUFFIX = re.compile(r",?\s*(?:P\.?C\.?\s*,?\s*)?J\s*\.?\s*J\s*\.?\s*$", re.IGNORECASE)


def _expand_plural_bench(block: str) -> list[str]:
    """Expand the shared-'JJ.' coram form into individual judges, or [] if absent."""
    text = re.sub(r"\s+", " ", block.replace("\n", " ")).strip().rstrip(".,") + "."
    if not _PLURAL_SUFFIX.search(text):
        return []
    names_part = _PLURAL_SUFFIX.sub("", text).strip(" ,.")
    out: list[str] = []
    for nm in re.split(r"\s*(?:,|&|\band\b)\s*", names_part, flags=re.IGNORECASE):
        nm = re.sub(r"^\(?\s*[0-9ivx]+\s*[.)]\s*", "", nm.strip(), flags=re.IGNORECASE)
        nm = _HONORIFIC_PREFIX.sub("", nm).strip(" .,")
        if re.search(r"[A-Za-z]{2,}", nm) and 2 <= len(nm) <= 50 and nm.count(" ") <= 6:
            out.append(f"{nm}, J.")
    return out if len(out) >= 2 else []


def _judges_from_block(block: str) -> list[str]:
    """Every judge in a coram window: shared-'JJ.' layout first, then
    individual-suffix names; capped at a realistic 7-judge bench."""
    plural = _expand_plural_bench(block)
    if len(plural) >= 2:
        return plural[:7]
    out: list[str] = []
    for cand in _split_candidates(block):
        j = _clean_judge(cand)
        if j and j not in out:
            out.append(j)
            if len(out) >= 7:
                break
    return out


def _scan_judge_clusters(lines: list[str]) -> list[str]:
    """Marker-less fallback: the largest run of adjacent judge-pattern lines (blank
    lines ignored; a real non-judge line or a stop section ends the run). Catches
    OCR'd captions whose 'Before:' marker was garbled. Needs >= 2 judges to count."""
    best: list[str] = []
    cur: list[str] = []
    for ln in lines:
        if _BENCH_STOP.match(ln):
            if len(cur) > len(best):
                best = cur
            cur = []
            continue
        s = ln.strip()
        if not s:
            continue
        if _clean_judge(s):
            cur.append(_clean_judge(s))
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    out: list[str] = []
    for j in best:
        if j not in out:
            out.append(j)
    return out[:7] if len(out) >= 2 else []


# --------------------------------------------------------------------------- #
# Signature-block fallback (coram from the END of the judgment)                #
# --------------------------------------------------------------------------- #
# Many judgments (esp. FR cases) print NO "Before:" marker in the front matter —
# the only complete record of the panel is the signature block at the very end,
# where each judge signs above/below a role line and the concurring judges add
# "I agree." We scan the tail for judge-pattern lines anchored to those cues.

# A bare judicial-office line ("JUDGE OF THE SUPREME COURT", "CHIEF JUSTICE",
# "PRESIDENT OF THE COURT OF APPEAL"). These sit next to a signature, but are
# NOT themselves names — they anchor the scan and must be excluded as judges.
_SIGN_ROLE = re.compile(
    r"^[\s>*•\-]*"
    r"(?:(?:THE\s+)?(?:ACTING\s+)?CHIEF\s+JUSTICE"
    r"|(?:JUDGE|JUSTICE)\s+OF\s+THE\s+(?:SUPREME\s+COURT|COURT\s+OF\s+APPEAL|HIGH\s+COURT)"
    r"|PRESIDENT\s+OF\s+THE\s+COURT\s+OF\s+APPEAL)"
    r"\s*$",
    re.IGNORECASE,
)
# A concurrence line — "I agree." / "I respectfully agree" / "I have read…".
_SIGN_AGREE = re.compile(r"^[\s>*•\-]*I\s+(?:respectfully\s+)?(?:agree|concur|have\s+read)\b", re.IGNORECASE)


def _scan_signature_block(lines: list[str]) -> list[str]:
    """Coram from the end-of-judgment signatures: every judge-pattern line that
    sits within a few lines of a signature cue (a role line or "I agree."),
    deduped, capped at 7. Returns [] when the tail carries no signature cues, so
    a judgment without a clear sign-off never produces spurious names."""
    cue_idx = [i for i, ln in enumerate(lines) if _SIGN_ROLE.match(ln) or _SIGN_AGREE.match(ln)]
    if not cue_idx:
        return []
    cues = set(cue_idx)
    out: list[str] = []
    for idx, ln in enumerate(lines):
        s = ln.strip()
        if not s or _SIGN_ROLE.match(ln):       # never treat a role line as a name
            continue
        # A signature cue within ±3 lines marks this as a signing judge (vs. an
        # incidental "per Fernando, J." somewhere in the body).
        if not any(0 < abs(c - idx) <= 3 for c in cues):
            continue
        j = _clean_judge(s)
        if j and j not in out:
            out.append(j)
            if len(out) >= 7:
                break
    return out


def _surname(name: str) -> str:
    """Last alphabetic token of a judge name, lowercased — for surname-dedup."""
    head = _JUDGE_SUFFIX.sub("", name).strip(" ,.")
    head = re.sub(r"\b(?:PC|QC|P\.C\.|Q\.C\.)\b", "", head, flags=re.IGNORECASE)
    toks = re.findall(r"[A-Za-z]{2,}", head)
    return toks[-1].lower() if toks else name.strip().lower()


def merge_benches(*benches: list[str]) -> list[str]:
    """Union several judge lists, keeping the first occurrence and dropping later
    entries that repeat a surname already seen (so "A.H.M.D. Nawaz" and
    "Nawaz, J." collapse to one). Order: earlier lists first. Capped at 7."""
    out: list[str] = []
    seen: set[str] = set()
    for bench in benches:
        for nm in bench:
            nm = (nm or "").strip()
            if not nm:
                continue
            sn = _surname(nm)
            if sn in seen:
                continue
            seen.add(sn)
            out.append(nm)
            if len(out) >= 7:
                return out
    return out


def _docling_bench_pages(path: str) -> list[str] | None:
    """Per-page text for the coram scan when Docling is the active parser
    (``ROSCRIBE_PARSER=docling``). Returns Docling's cached reading-order pages —
    cheap, since indexing already parsed the document — or ``None`` in legacy mode
    or when Docling is unavailable, so the caller falls back to the fitz front +
    signature read. Docling preserves the caption's reading order, so the regex
    panel-parser below sees a far cleaner "Before:/Coram:" block."""
    from .config import settings

    if settings.parser != "docling":
        return None
    try:
        from .parsing import extract_pages as _docling_extract_pages

        pages = _docling_extract_pages(path)
        return pages or None
    except Exception:
        return None


def extract_bench(pages_or_path) -> list[str]:
    """Parse the full panel of judges (the coram) from a judgment.

    Accepts a PDF path (str / Path), a single text string, or a list of page
    strings (as returned by `extract_pages`). Returns the judges in the order
    printed, each normalised like ``"Mahinda Samayawardhena, J."``. Returns
    ``[]`` when no panel can be confidently identified — callers should then
    fall back to the scrape metadata.

    Resolution order: (1) the front-matter "Before:/Coram:" panel, (2) the
    end-of-judgment signature block, (3) a marker-less cluster of judge lines in
    the front matter. The front and signature panels are merged (surname-deduped)
    so a 3- or 5-judge coram is recovered even when one source is incomplete.
    """
    # --- normalise input to front pages + tail pages -------------------------
    tail_pages: list[str] = []
    if isinstance(pages_or_path, (list, tuple)):
        pages = list(pages_or_path)
        if len(pages) > 3:
            tail_pages = pages[-2:]
    elif isinstance(pages_or_path, Path) or (
        isinstance(pages_or_path, str)
        and pages_or_path.lower().endswith(".pdf")
        and Path(pages_or_path).exists()
    ):
        # The coram lives in the front matter (pages 1-3) OR the signature block
        # (last 1-2 pages). v2: prefer Docling's cached reading-order pages — only
        # [:3] (front) and [-2:] (signatures) are used below. Else read/OCR only
        # those pages directly with fitz — never the whole long judgment.
        docling_pages = _docling_bench_pages(str(pages_or_path))
        if docling_pages is not None:
            pages = docling_pages
            tail_pages = docling_pages[-2:] if len(docling_pages) > 3 else []
        else:
            try:
                pages = []
                with fitz.open(str(pages_or_path)) as doc:
                    n = doc.page_count

                    def _page_text(i: int) -> str:
                        page = doc[i]
                        t = page.get_text("text")
                        if len(t.strip()) < 200:
                            t = _ocr_page(page, "eng+sin+tam") or t
                        return t

                    for i in range(min(3, n)):
                        pages.append(_page_text(i))
                    # Tail: the last two pages, skipping any already read as front matter.
                    tail_idx = [i for i in (n - 2, n - 1) if i >= 3]
                    tail_pages = [_page_text(i) for i in tail_idx]
            except Exception:
                return []
    else:
        pages = [str(pages_or_path)]

    # The coram is in the front matter; a long multi-party caption can push it onto
    # the 3rd page, so search the first three.
    text = "\n".join(str(p) for p in pages[:3])
    front_lines = text.splitlines() if text.strip() else []

    front_panel: list[str] = []
    for idx, line in enumerate(front_lines):
        m = _BENCH_MARKER.match(line)
        if not m:
            continue
        # Take a WINDOW from the marker up to the next labelled section (Counsel,
        # For the …, the parties, etc.) or ~15 lines. SC captions spread the panel
        # across many blank lines, so we gather the whole window THEN extract — far
        # more robust than the old line-by-line scan that stopped at the first gap.
        window: list[str] = []
        remainder = line[m.end():].strip()
        if remainder:
            window.append(remainder)
        nonblank = 1 if remainder else 0
        for nxt in front_lines[idx + 1: idx + 41]:   # span blank-heavy captions (judges sit ~6 blanks apart)
            if _BENCH_STOP.match(nxt):
                break
            window.append(nxt)
            if nxt.strip():
                nonblank += 1
                if nonblank >= 9:              # enough for a 7-judge panel; stop before over-reaching
                    break

        front_panel = _judges_from_block("\n".join(window))
        if front_panel:
            break  # first valid panel wins; ignore later "before" mentions

    # The signature block (tail, with front matter as a fallback location for
    # short judgments that signed on page 1-3).
    sig_lines = "\n".join(str(p) for p in tail_pages).splitlines()
    sig_panel = _scan_signature_block(sig_lines) if sig_lines else []
    if not sig_panel and front_lines:
        sig_panel = _scan_signature_block(front_lines)

    # Merge the two authoritative sources (surname-deduped). The signature block
    # lists every concurring judge; the "Before:" line gives presiding order.
    merged = merge_benches(front_panel, sig_panel)
    if merged:
        return merged

    # No marker and no signatures (e.g. OCR garbled both) — fall back to the
    # largest run of adjacent judge-pattern lines in the front matter.
    return _scan_judge_clusters(front_lines)


# Authoring judge of a law-report opinion: 'PATHIRANA, J.— The …' opener or a
# 'Per Pathirana, J.' attribution after the headnote's HELD block.
_OPINION_AUTHOR = re.compile(r"\b([A-Z][A-Za-z'’.\- ]{2,30}?),?\s*(A\.?C\.?J|C\.?J|J)\.?\s*[—–―-]")
_PER_AUTHOR = re.compile(r"\bPer\s+([A-Z][A-Za-z'’.\- ]{2,30}?),?\s*(A\.?C\.?J|C\.?J|J)\b")
_AUTHOR_SFX = {"J": "J.", "CJ": "C.J.", "ACJ": "A.C.J."}


def extract_opinion_author(text: str) -> str:
    """The judge who delivered a law-report judgment, e.g. 'Pathirana, J.' —
    or '' when the report doesn't attribute one (per-curiam minutes etc.)."""
    t = (text or "")[:4000]
    m = _OPINION_AUTHOR.search(t) or _PER_AUTHOR.search(t)
    if not m:
        # SLR style: the opinion opens with a standalone 'NAME, J.' line. Bench
        # listings are ALSO standalone lines, but only the opener is followed by
        # prose — prefer that; else fall back to the last standalone line.
        ms = list(re.finditer(
            r"(?m)^\s*([A-Z][A-Za-z'’.\- ]{2,30}?),?\s*(A\.?C\.?J|C\.?J|J)\.?\s*$", t))
        prose = [x for x in ms if re.match(r"\s*(?:The|This|In|On|By|It|An?|Learned|Heard)\b",
                                           t[x.end():x.end() + 40].lstrip("\n\r "))]
        m = prose[0] if prose else None
        if not m:
            # Chunked text flattens newlines — inline: 'ALUWIHARE, J. The Petitioner…'
            m = re.search(r"([A-Z][A-Za-z'’.\- ]{2,28}?),\s*(A\.?C\.?J|C\.?J|J)\.\s+"
                          r"(?=(?:The|This|In|On|By|It|Learned|Having|Heard)\b)", t)
        if not m and ms:
            m = ms[-1]  # last resort: final standalone judge line
    if not m:
        return ""
    raw = m.group(1)
    # Headnote words fused before the name ('Permissibility. Vythialingam') —
    # strip leading LONG dotted words; dotted single-letter initials survive.
    raw = re.sub(r"^(?:[A-Za-z'’\-]{3,}\.\s+)+", "", raw)
    raw = _collapse_spaced_name(raw)          # 'V Y T H IA L IN G A M' -> one word
    toks = raw.split()
    # Shorter all-single-letter runs ('W I T H E R S') the run-collapse missed.
    if len(toks) >= 3 and all(len(x.strip(".'’")) == 1 for x in toks):
        raw = "".join(x.strip(".'’") for x in toks)
    name = re.sub(r"\s+", " ", raw).strip(" .'’-").title()
    if not (2 < len(name) <= 30):
        return ""
    return f"{name}, {_AUTHOR_SFX.get(m.group(2).replace('.', '').upper(), 'J.')}"


# --------------------------------------------------------------------------- #
# Parties (split the clumped scrape string into a structured list)             #
# --------------------------------------------------------------------------- #
# The archive table records parties as one run-on string, e.g.
#   "Lalitha Weerasinghe … PETITIONER Vs. 1. Ranmal Kodithuwakku … 2. Prof …
#    RESPONDENTS AND … Petitioner-Appellant Vs …".
# We split it into individual {name, role, side} entries for a readable list.

# Procedural roles (incl. compound forms like "Plaintiff-Respondent-Appellant").
_ROLE_WORD = r"(?:Plaintiff|Defendant|Petitioner|Respondent|Appellant|Applicant|Complainant|Accused|Intervenient|Claimant|Defaulter)"
_PARTY_ROLE = re.compile(
    r"(?:Substituted[-\s]+|Added[-\s]+|Added\s+)?"
    rf"{_ROLE_WORD}(?:s)?"
    rf"(?:\s*[-–]\s*(?:Substituted\s+|Added\s+)?{_ROLE_WORD}(?:s)?)*",
    re.IGNORECASE,
)
# Connectors that separate parties / procedural stages (stripped from name heads).
_PARTY_CONNECTOR = re.compile(
    r"^\s*(?:"
    r"V[Ss]?\.?,?|versus|"                                  # Vs / V. / VS / versus
    r"AND\s+NOW\s+BETWEEN|NOW\s+BETWEEN|AND\s+BETWEEN|And\s+between|BETWEEN|AND|&"
    r")\s+",
    re.IGNORECASE,
)
# A numbered list marker that begins a distinct party: "1.", "2.", "1(iii).".
_PARTY_NUM = re.compile(r"(?<![\w/])\d{1,2}(?:\([ivxa-d]+\))?\.\s+(?=[A-Z(])")
_ROLE_SIDE = {  # which "side" each role sits on (for grouping / display)
    "petitioner": "petitioner", "applicant": "petitioner", "appellant": "petitioner",
    "plaintiff": "petitioner", "complainant": "petitioner",
    "respondent": "respondent", "defendant": "respondent", "accused": "respondent",
    "defaulter": "respondent",
}


def _role_side(role: str) -> str:
    """Bucket a (possibly compound) role onto petitioner/respondent by its LAST
    component — the party's current posture (e.g. 'Defendant-Appellant' → appellant)."""
    words = re.findall(_ROLE_WORD, role, re.IGNORECASE)
    return _ROLE_SIDE.get(words[-1].lower(), "") if words else ""


def _clean_role(role: str) -> str:
    """Normalise a (compound) role: collapse '- ' spacing and drop consecutive
    duplicate components ('Respondent-Respondent' → 'Respondent')."""
    role = re.sub(r"\s*[-–]\s*", "-", re.sub(r"\s+", " ", role).strip())
    parts, out = role.split("-"), []
    for p in parts:
        if not out or out[-1].lower() != p.lower():
            out.append(p)
    return "-".join(out)


def _tidy_party_name(block: str) -> str:
    prev = None
    block = block.strip()
    # Peel leading connectors / numbering repeatedly ("-Vs- 1." → "").
    while block and block != prev:
        prev = block
        block = _PARTY_CONNECTOR.sub("", block)
        block = re.sub(r"^[\s,.;:\-–]+", "", block)
        block = re.sub(r"^\s*\d{1,2}(?:\([ivxa-d]+\))?\.\s*", "", block)
        block = re.sub(r"^(?:vs?|versus)\b[\s.,;:\-–]*", "", block, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", block).strip(" ,.;:-–")


# A fragment that is ONLY a connector/role word is not a party name.
_PARTY_JUNK = re.compile(r"^(?:v[s]?|versus|and|&|between|now)$", re.IGNORECASE)


# "Vs" between two role-less party blocks ("S. Rajendran Chettiar Vs S. Narayanan…").
_VS_SPLIT = re.compile(r"\s+(?:V[Ss]\.?,?|V\.|VS\.?|versus)\s+")


def _split_side(block: str, side: str) -> list[dict]:
    """One side of a role-less caption → entries. Numbered markers split
    multi-party sides; otherwise the whole side is a single party."""
    pieces = _PARTY_NUM.split(block) if _PARTY_NUM.search(block) else [block]
    out = []
    for piece in pieces:
        name = _tidy_party_name(piece)
        if len(name) >= 3 and re.search(r"[A-Za-z]{2,}", name) and not _PARTY_JUNK.match(name):
            out.append({"name": name, "role": "", "side": side})
    return out


def parse_parties(text: str) -> list[dict]:
    """Split the clumped parties string into ordered ``{name, role, side}`` entries.

    Roles terminate each party; numbered markers split multi-party sides; identical
    names (repeated across appeal stages) are merged, keeping the most descriptive
    role. Role-less captions fall back to a "Vs" split (left = petitioner side,
    right = respondent side) or a plain numbered list. Returns ``[]`` only when
    nothing parseable is found (caller falls back to the raw string)."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    entries: list[dict] = []
    prev_end = 0
    for m in _PARTY_ROLE.finditer(text):
        block = text[prev_end:m.start()]
        role = re.sub(r"\s+", " ", m.group(0)).strip()
        prev_end = m.end()
        # A side may pack several numbered parties that share this one role label.
        pieces = _PARTY_NUM.split(block) if _PARTY_NUM.search(block) else [block]
        for piece in pieces:
            name = _tidy_party_name(piece)
            if len(name) >= 3 and re.search(r"[A-Za-z]{2,}", name) and not _PARTY_JUNK.match(name):
                entries.append({"name": name, "role": _clean_role(role), "side": _role_side(role)})
    if not entries:
        # Fallback A: role-less "X Vs Y" — left of the first Vs is the
        # petitioner/appellant side, everything right of it the respondents.
        m = _VS_SPLIT.search(text)
        if m:
            entries = _split_side(text[:m.start()], "petitioner") + \
                      _split_side(text[m.end():], "respondent")
        else:
            # Fallback B: no Vs at all — a bare (possibly numbered) list of names.
            entries = _split_side(text, "")
    if not entries:
        return []
    # Merge repeats (the same person reappears across stages) — keep the longest role.
    merged: dict[str, dict] = {}
    order: list[str] = []
    for e in entries:
        key = re.sub(r"[^a-z0-9]", "", e["name"].lower())[:60]
        if key in merged:
            if len(e["role"]) > len(merged[key]["role"]):
                merged[key]["role"] = e["role"]
                merged[key]["side"] = e["side"]
        else:
            merged[key] = e
            order.append(key)
    return [merged[k] for k in order]


# --------------------------------------------------------------------------- #
# Personal repository                                                          #
# --------------------------------------------------------------------------- #
def _clean_subject(name: str) -> str:
    return re.sub(r"^\d+\s*-\s*", "", name).strip()


def list_note_files(directory: str) -> list[tuple[Path, str, str]]:
    """Return (path, subject, category) for every supported note file."""
    root = Path(directory)
    out: list[tuple[Path, str, str]] = []
    for p in sorted(root.rglob("*")):
        if (
            p.is_file()
            and p.suffix.lower() in NOTE_EXTS
            and not p.name.startswith("~$")
            and not p.name.startswith(".")
        ):
            parts = p.relative_to(root).parts
            subject = _clean_subject(parts[0]) if parts else "General"
            category = parts[1] if len(parts) > 2 else "General"
            out.append((p, subject, category))
    return out


def chunks_for_note(path: Path, subject: str, category: str) -> list[Chunk]:
    """Extract + chunk one note file, tagged with subject / category."""
    label = f"{subject} / {category} / {path.name}"
    meta = {"subject": subject, "category": category, "filename": path.name}
    chunks = chunk_pages(extract_document(str(path)), case_no=label, source="personal_repo")
    for c in chunks:
        c.metadata.update(meta)
    return chunks


def load_personal_repo(directory: str, limit: int | None = None) -> list[Chunk]:
    """Walk the notes folder; tag chunks with Subject / Category from the layout."""
    files = list_note_files(directory)
    if limit:
        files = files[:limit]
    chunks: list[Chunk] = []
    for p, subject, category in files:
        try:
            chunks.extend(chunks_for_note(p, subject, category))
        except Exception:
            continue  # unreadable file — skip rather than fail the batch
    return chunks


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Extract + chunk a judgment PDF.")
    ap.add_argument("pdf")
    ap.add_argument("--out", type=Path, default=None, help="write per-page extracted text here")
    ap.add_argument("--show", type=int, default=3, help="number of sample chunks to print")
    args = ap.parse_args(argv)

    case_no = case_no_from_filename(args.pdf)
    pages = extract_pages(args.pdf)
    chunks = chunk_pages(pages, case_no)
    print(f"{Path(args.pdf).name}: {len(pages)} pages, {len(chunks)} chunks, case_no={case_no!r}")
    for c in chunks[: args.show]:
        print(f"\n  {c.anchor()}\n   {c.text[:220].strip().replace(chr(10), ' ')}…")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(f"\n===== Page {pno} =====\n{text}" for pno, text in enumerate(pages, 1))
        args.out.write_text(body)
        print(f"\nExtracted text -> {args.out}")


if __name__ == "__main__":
    main()
