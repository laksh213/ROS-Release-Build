"""Phase 1b — lankalaw.net scraper: Acts/legislation + case judgements.

lankalaw.net is a server-rendered WordPress legal library (Yoast sitemaps, no JS
wall, PDFs served straight from /wp-content/uploads or mirrored from
documents.gov.lk / parliament.lk). Two content kinds we want:

  • Statutes — curated index pages each list a column of Act/statute PDFs:
      /legislations/core-legislations/<slug>/                 (Penal Code, Evidence…)
      /legislations/constitution-of-sri-lanka/
      /legislations/acts-and-laws/.../sri-lanka-acts-<YYYY>/   (Acts, by year)
      /legislations/acts-and-laws/{acts-and-amendments,consolidated-acts}-…/<A-Z>/
      /legislative-enactments/…                                (older consolidations)
      /sri-lanka-acts-2025/  /sri-lanka-acts-2026/             (recent, top-level)
    The link text is the Act's name; the filename usually encodes No./year
    ("09-2024_E.pdf" = Act No. 9 of 2024).

  • Cases — leaf pages each link one judgment PDF (the case name is the page
    title), e.g.
      /case-laws/sri-lanka-law-reports/sri-lanka-law-reports-<YYYY>/<slug>/
    Some judgment PDFs reuse the supremecourt.lk filenames (e.g. sc_appeal_…),
    so the existing content dedupe merges any overlap at index time.

Discovery is driven by the Yoast sitemaps (sitemap_index.xml → page/post
sitemaps) — we never blind-crawl. Polite by default, mirroring src/scrape.py:
robots.txt honoured, realistic User-Agent, delay between requests, retry with
backoff, and skips files already on disk (resumable).

Outputs:
  data/statutes/<file>.pdf          downloaded Act/statute PDFs
  data/lankalaw_cases/<file>.pdf    downloaded judgement PDFs
  data/lankalaw_manifest.json       one record per item (audit trail)

CLI:
  python -m src.scrape_lankalaw --what all --metadata-only       # catalogue EVERY pdf
  python -m src.scrape_lankalaw --what all                       # download the whole site
  python -m src.scrape_lankalaw --what all --deep                # + follow links (max coverage)
  python -m src.scrape_lankalaw --what all --from-manifest       # resume a download
  python -m src.scrape_lankalaw --what statutes --years 2015-2026
  python -m src.scrape_lankalaw --dry-run

For an unattended full harvest from your own terminal, use scrape_lankalaw_all.sh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

BASE = "https://lankalaw.net"
SITEMAP_INDEX = f"{BASE}/sitemap_index.xml"
ROBOTS_URL = f"{BASE}/robots.txt"

REPO_ROOT = Path(__file__).resolve().parents[1]
STATUTES_DIR = REPO_ROOT / "data" / "statutes"
CASES_DIR = REPO_ROOT / "data" / "lankalaw_cases"
DOCS_DIR = REPO_ROOT / "data" / "lankalaw_docs"   # everything else (--what all)
MANIFEST_PATH = REPO_ROOT / "data" / "lankalaw_manifest.json"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 ROScribe/0.1 (legal research; respectful crawler)"
)

# Which page-sitemap entries are statute index/leaf pages vs case pages. Driven
# off the URL path so we only ever fetch pages that can carry the content we want.
_STATUTE_PATH = re.compile(
    r"/(?:legislations|legislative-enactments)/|/sri-lanka-acts-\d{4}/", re.I
)
_CASE_PATH = re.compile(r"/case-laws/", re.I)
_DATED_POST = re.compile(r"/\d{4}/\d{2}/\d{2}/")  # dated articles (case write-ups)

# --deep crawl link filter: stay on lankalaw.net content pages; skip the store /
# account / admin / feed machinery and binary assets (we only want HTML pages).
_SKIP_PATH = re.compile(
    r"^/(?:wp-admin|wp-json|wp-content|cart|checkout|feed)\b"
    r"|/(?:cart|checkout|gp-checkout|no-access|thank-you)/?$"
    r"|^/home/(?:membership|my-account)"
    r"|^/(?:my-account|membership-login|login|register)\b", re.I)
_ASSET_EXT = re.compile(
    r"\.(?:jpg|jpeg|png|gif|svg|webp|css|js|ico|zip|rar|mp4|mp3|woff2?|ttf|eot|xml)(?:[?#]|$)", re.I)


def _classify_kind(page_url: str) -> str:
    """statute / case / document, from the source page's path."""
    if _STATUTE_PATH.search(page_url):
        return "statute"
    if _CASE_PATH.search(page_url) or _DATED_POST.search(page_url):
        return "case"
    return "document"


def _crawlable(url: str) -> bool:
    """A lankalaw.net HTML content page worth fetching (for --deep link-follow)."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.netloc.endswith("lankalaw.net"):
        return False
    if "add-to-cart" in (p.query or "") or _ASSET_EXT.search(p.path) or _SKIP_PATH.search(p.path):
        return False
    return True

# Anchor text that isn't a real title — fall back to the page heading.
_GENERIC_LINK = re.compile(
    r"^(?:download|click\b|view\b|read\s+more|pdf|here|open|see\b|link"
    r"|english|sinhala|sinhalese|tamil)\b", re.I)
# PDF hrefs that are site chrome, not content (favicons/logos saved as .pdf? none,
# but uploaded cropped logos are images; guard anyway).
_NON_CONTENT = re.compile(r"cropped-|/logos?/|favicon", re.I)


@dataclass
class LankaLawRecord:
    kind: str                 # "statute" | "case"
    title: str                # human name (Act name / case name)
    pdf_url: str
    source_page: str          # the index/leaf page the link was found on
    category: str = ""        # path segment, e.g. "core-legislations", "sri-lanka-acts-2024"
    act_no: str = ""          # statutes: Act number, when derivable
    year: str = ""            # 4-digit year, when derivable
    host: str = ""            # PDF host (lankalaw.net / documents.gov.lk / parliament.lk)
    filename: str = ""        # local safe filename
    downloaded: bool = False
    local_path: str = ""
    error: str = ""
    scraped_at: str = ""


# --------------------------------------------------------------------------- #
# HTTP + robots                                                                #
# --------------------------------------------------------------------------- #
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def load_robots(session: requests.Session) -> RobotFileParser:
    """Fetch + parse robots.txt. On any failure we default to allow but say so —
    lankalaw's robots is permissive (Yoast emits an empty Disallow for '*')."""
    rp = RobotFileParser()
    rp.set_url(ROBOTS_URL)
    try:
        rp.parse(fetch(session, ROBOTS_URL).text.splitlines())
    except Exception as e:  # noqa: BLE001 — never let robots fetch kill the run
        print(f"  robots.txt unavailable ({e}); proceeding (paths are public).")
        rp.allow_all = True
    return rp


def allowed(rp: RobotFileParser, url: str) -> bool:
    try:
        return rp.can_fetch(USER_AGENT, url)
    except Exception:  # noqa: BLE001
        return True


def fetch(session: requests.Session, url: str, retries: int = 3) -> requests.Response:
    """GET with simple exponential backoff (mirrors src/scrape.py)."""
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=45)
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- #
# Sitemap-driven discovery                                                     #
# --------------------------------------------------------------------------- #
def _locs(xml: str) -> list[str]:
    return re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml)


def sitemap_pages(session: requests.Session) -> dict[str, list[str]]:
    """Return {sub_sitemap_name: [page urls]} for the page + post sitemaps."""
    out: dict[str, list[str]] = {}
    index = _locs(fetch(session, SITEMAP_INDEX).text)
    for sm in index:
        name = urlparse(sm).path.strip("/").rsplit("/", 1)[-1]
        if name in ("page-sitemap.xml", "post-sitemap.xml"):
            out[name] = _locs(fetch(session, sm).text)
    return out


def discover(session: requests.Session, what: str, include_posts: bool) -> list[str]:
    """Seed page URLs to scan, from the Yoast sitemaps. ``what``:
      all       — every page + post (the whole published site)
      statutes  — legislation / acts pages
      cases     — case-law pages + dated case write-ups
      both      — statutes + cases (+ posts if include_posts)
    """
    pages = sitemap_pages(session)
    page_locs = pages.get("page-sitemap.xml", [])
    post_locs = pages.get("post-sitemap.xml", [])
    if what == "all":
        seeds = page_locs + post_locs + [f"{BASE}/"]
    elif what == "statutes":
        seeds = [u for u in page_locs if _STATUTE_PATH.search(u)]
    elif what == "cases":
        seeds = [u for u in page_locs if _CASE_PATH.search(u)]
        seeds += [u for u in post_locs if _DATED_POST.search(u)]
    else:  # both
        seeds = [u for u in page_locs if _STATUTE_PATH.search(u) or _CASE_PATH.search(u)]
        if include_posts:
            seeds += [u for u in post_locs if _DATED_POST.search(u)]
    return sorted(set(seeds))


# --------------------------------------------------------------------------- #
# Per-page PDF extraction                                                      #
# --------------------------------------------------------------------------- #
def _page_title(soup: BeautifulSoup) -> str:
    h = soup.select_one("h1.wp-block-post-title, h2.wp-block-post-title, h1, h2.wp-block-post-title")
    if h and h.get_text(strip=True):
        return h.get_text(" ", strip=True)
    t = soup.title.get_text(strip=True) if soup.title else ""
    return re.sub(r"\s*[-–|]\s*Lanka\s*Law\s*$", "", t, flags=re.I).strip()


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def extract_pdf_rows(html: str, page_url: str) -> list[tuple[str, str]]:
    """(title, absolute_pdf_url) pairs on a page. Link text is the title; for a
    leaf page whose link reads 'Download' we fall back to the page heading."""
    soup = BeautifulSoup(html, "html.parser")
    page_title = _page_title(soup)
    rows: dict[str, str] = {}  # pdf_url -> best title (first non-generic wins)
    for a in soup.find_all("a", href=re.compile(r"\.pdf(?:[?#]|$)", re.I)):
        href = a.get("href", "")
        if not href or _NON_CONTENT.search(href):
            continue
        pdf_url = urljoin(page_url, href.strip())
        title = _clean(a.get_text(" ", strip=True))
        if not title or _GENERIC_LINK.match(title) or len(title) < 3:
            title = page_title
        # Prefer the first descriptive title we see for a given PDF.
        if pdf_url not in rows or (rows[pdf_url] == page_title and title != page_title):
            rows[pdf_url] = title
    return [(t, u) for u, t in rows.items()]


# --------------------------------------------------------------------------- #
# Metadata + filenames                                                         #
# --------------------------------------------------------------------------- #
_ACTNO_YEAR_FILE = re.compile(r"^(\d{1,3})[-_](\d{4})", )          # 09-2024_E.pdf
_ACTNO_YEAR_TEXT = re.compile(r"\bNo\.?\s*(\d{1,3})\s+of\s+(\d{4})", re.I)
_YEAR_ANY = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _category_of(page_url: str) -> str:
    """Last meaningful path segment of the source page (sans trailing slash)."""
    segs = [s for s in urlparse(page_url).path.split("/") if s]
    return segs[-1] if segs else ""


def _derive_statute_meta(title: str, pdf_url: str, page_url: str) -> tuple[str, str]:
    """Best-effort (act_no, year) from the filename, then the title, then the
    source page slug (e.g. 'sri-lanka-acts-2024')."""
    stem = unquote(urlparse(pdf_url).path.rsplit("/", 1)[-1])
    act_no = year = ""
    m = _ACTNO_YEAR_FILE.match(stem)
    if m:
        act_no, year = m.group(1), m.group(2)
    if not (act_no and year):
        m = _ACTNO_YEAR_TEXT.search(title)
        if m:
            act_no, year = m.group(1), m.group(2)
    if not year:
        m = _YEAR_ANY.search(_category_of(page_url)) or _YEAR_ANY.search(title)
        if m:
            year = m.group(1)
    return act_no, year


def _slug(text: str, n: int = 55) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return s[:n].strip("-")


def _hash6(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:6]


def _local_filename(rec: LankaLawRecord) -> str:
    """Deterministic, readable, collision-proof local name (so re-runs resume).
    The pretty title lives in the manifest; this is just the on-disk key."""
    stem = unquote(urlparse(rec.pdf_url).path.rsplit("/", 1)[-1])
    base = _slug(rec.title) or _slug(Path(stem).stem) or "doc"
    tag = f"-{rec.act_no}-{rec.year}" if (rec.act_no and rec.year) else ""
    return f"{base}{tag}-{_hash6(rec.pdf_url)}.pdf"


# --------------------------------------------------------------------------- #
# Collect                                                                      #
# --------------------------------------------------------------------------- #
def collect(session: requests.Session, rp: RobotFileParser, what: str,
            include_posts: bool, years: tuple[int, int] | None, *,
            deep: bool = False, max_pages: int = 20000,
            page_delay: float = 0.3) -> list[LankaLawRecord]:
    """Scan pages (sitemap seeds; with ``deep`` also BFS-follow internal links)
    and harvest every PDF, classified statute / case / document. Resumable
    discovery is cheap — only HTML pages are fetched here, not the PDFs."""
    from collections import deque

    seeds = discover(session, what, include_posts)
    print(f"  {len(seeds)} seed pages from sitemap"
          + (f"; deep link-follow on (cap {max_pages})" if deep else ""), flush=True)

    by_url: dict[str, LankaLawRecord] = {}
    visited: set[str] = set()
    queue: deque[str] = deque(seeds)
    scanned = 0

    while queue and scanned < max_pages:
        page_url = queue.popleft()
        key = page_url.split("#")[0].rstrip("/")
        if key in visited:
            continue
        visited.add(key)
        if not _crawlable(page_url) or not allowed(rp, page_url):
            continue
        try:
            resp = fetch(session, page_url)
        except requests.RequestException:
            continue
        if "text/html" not in resp.headers.get("Content-Type", "").lower():
            continue
        scanned += 1
        html = resp.text
        kind = _classify_kind(page_url)
        for title, pdf_url in extract_pdf_rows(html, page_url):
            if pdf_url in by_url:
                continue
            rec = LankaLawRecord(kind=kind, title=title, pdf_url=pdf_url, source_page=page_url,
                                 category=_category_of(page_url), host=urlparse(pdf_url).netloc)
            if kind == "statute":
                rec.act_no, rec.year = _derive_statute_meta(title, pdf_url, page_url)
            else:
                m = _YEAR_ANY.search(title) or _YEAR_ANY.search(_category_of(page_url))
                rec.year = m.group(1) if m else ""
            if years and rec.year and not (years[0] <= int(rec.year) <= years[1]):
                continue
            rec.filename = _local_filename(rec)
            by_url[pdf_url] = rec
        if deep:  # enqueue internal content links for the next BFS level
            for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
                nxt = urljoin(page_url, a["href"]).split("#")[0]
                if _crawlable(nxt) and nxt.rstrip("/") not in visited:
                    queue.append(nxt)
        if scanned % 25 == 0:
            print(f"    …scanned {scanned} pages, {len(by_url)} PDFs"
                  + (f", {len(queue)} queued" if deep else ""), flush=True)
        time.sleep(page_delay)

    print(f"  scanned {scanned} pages → {len(by_url)} unique PDFs", flush=True)
    return sorted(by_url.values(), key=lambda r: (r.kind, r.year, r.title))


# --------------------------------------------------------------------------- #
# Download                                                                     #
# --------------------------------------------------------------------------- #
def _dest_dir(rec: LankaLawRecord) -> Path:
    return {"statute": STATUTES_DIR, "case": CASES_DIR}.get(rec.kind, DOCS_DIR)


def _free_gb(path: Path) -> float:
    """Free space (GB) on the volume holding `path` — for the low-disk safety stop."""
    try:
        return shutil.disk_usage(path.parent if not path.exists() else path).free / 1e9
    except Exception:  # noqa: BLE001
        return float("inf")


def download(session: requests.Session, rp: RobotFileParser, rec: LankaLawRecord,
             delay: float) -> None:
    path = _dest_dir(rec) / rec.filename
    if path.exists() and path.stat().st_size > 0:           # resume — already have it
        rec.downloaded, rec.local_path = True, str(path)
        return
    if not allowed(rp, rec.pdf_url):
        rec.error = "disallowed by robots.txt"
        return
    try:
        resp = fetch(session, rec.pdf_url)
    except requests.RequestException as e:
        rec.error = f"{type(e).__name__}: {e}"
        return
    body = resp.content
    ctype = resp.headers.get("Content-Type", "")
    if not (body[:5] == b"%PDF-" or "application/pdf" in ctype.lower()):
        rec.error = f"not a PDF (Content-Type={ctype!r})"  # dead govt mirror → HTML 404
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    rec.downloaded, rec.local_path, rec.error = True, str(path), ""
    rec.scraped_at = datetime.now(timezone.utc).isoformat()
    time.sleep(delay)


def save_manifest(records: list[LankaLawRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in records], indent=2, ensure_ascii=False))


def load_manifest_records(path: Path) -> list[LankaLawRecord]:
    """Rebuild records from a prior manifest — lets `--from-manifest` resume the
    download without re-fetching every index page."""
    from dataclasses import fields

    if not path.exists():
        raise SystemExit(f"--from-manifest: {path} not found — run a crawl first.")
    known = {f.name for f in fields(LankaLawRecord)}
    return [LankaLawRecord(**{k: v for k, v in r.items() if k in known})
            for r in json.loads(path.read_text())]


def _merge_prior(records: list[LankaLawRecord], path: Path) -> None:
    """Carry over downloaded/local_path/scraped_at from a prior manifest so a
    re-run that re-discovers the same PDF keeps its audit trail."""
    if not path.exists():
        return
    try:
        prior = {r.get("pdf_url"): r for r in json.loads(path.read_text())}
    except Exception:  # noqa: BLE001
        return
    for rec in records:
        p = prior.get(rec.pdf_url)
        if p and p.get("downloaded"):
            rec.downloaded = True
            rec.local_path = p.get("local_path", "")
            rec.scraped_at = p.get("scraped_at", "")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _parse_years(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    m = re.match(r"^\s*(\d{4})\s*(?:[-:]\s*(\d{4}))?\s*$", spec)
    if not m:
        raise SystemExit(f"--years expects YYYY or YYYY-YYYY, got {spec!r}")
    lo = int(m.group(1))
    hi = int(m.group(2)) if m.group(2) else lo
    return (min(lo, hi), max(lo, hi))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Scrape lankalaw.net Acts + judgements + docs.")
    ap.add_argument("--what", choices=["all", "statutes", "cases", "both"], default="statutes",
                    help="'all' = every PDF on the site (statutes + cases + documents)")
    ap.add_argument("--limit", type=int, default=None, help="max PDFs to download")
    ap.add_argument("--years", type=str, default=None, help="filter by year, e.g. 2015-2026")
    ap.add_argument("--include-posts", action="store_true",
                    help="also scan dated articles (case write-ups) for judgment PDFs")
    ap.add_argument("--deep", action="store_true",
                    help="follow internal links (BFS) beyond the sitemap for max completeness")
    ap.add_argument("--max-pages", type=int, default=20000, help="--deep page-fetch cap")
    ap.add_argument("--page-delay", type=float, default=0.3, help="seconds between page fetches")
    ap.add_argument("--metadata-only", action="store_true", help="build manifest, no downloads")
    ap.add_argument("--from-manifest", action="store_true",
                    help="download straight from the existing manifest (skip re-crawl; for resuming)")
    ap.add_argument("--dry-run", action="store_true", help="print plan, change nothing")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between downloads")
    ap.add_argument("--min-free-gb", type=float, default=5.0,
                    help="halt safely when free disk space drops below this (GB)")
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = ap.parse_args(argv)

    years = _parse_years(args.years)
    session = make_session()
    print("Loading robots.txt …")
    rp = load_robots(session)

    if args.from_manifest:
        records = load_manifest_records(args.manifest)
        print(f"Loaded {len(records)} records from {args.manifest} (no re-crawl).")
    else:
        print(f"Collecting lankalaw.net listings (what={args.what}) …")
        records = collect(session, rp, args.what, args.include_posts, years,
                          deep=args.deep, max_pages=args.max_pages, page_delay=args.page_delay)
    n_stat = sum(r.kind == "statute" for r in records)
    n_case = sum(r.kind == "case" for r in records)
    n_doc = sum(r.kind == "document" for r in records)
    print(f"  {len(records)} PDFs ({n_stat} statutes, {n_case} cases, {n_doc} documents).")

    # Which of the (possibly full) record set to actually download this run.
    def _wanted(r: LankaLawRecord) -> bool:
        if args.what == "statutes" and r.kind != "statute":
            return False
        if args.what == "cases" and r.kind != "case":
            return False
        if years and not (r.year and years[0] <= int(r.year) <= years[1]):
            return False
        return True

    if args.dry_run:
        for r in [r for r in records if _wanted(r)][: args.limit or 15]:
            tag = f"No.{r.act_no}/{r.year}" if r.act_no else (r.year or "")
            print(f"  - [{r.kind}] {tag:>10}  {r.title[:70]}  <- {r.host}")
        print("Dry run — nothing written.")
        return

    # A fresh crawl writes the manifest; --from-manifest keeps the full set on disk
    # (we only update downloaded flags as we go) so filtering never drops records.
    if not args.from_manifest:
        _merge_prior(records, args.manifest)
        save_manifest(records, args.manifest)
        print(f"Manifest written: {args.manifest} ({len(records)} records)")
    if args.metadata_only:
        return

    todo = [r for r in records if _wanted(r) and not r.downloaded]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"Downloading {len(todo)} PDFs (delay={args.delay}s, stop at <{args.min_free_gb} GB free) …",
          flush=True)
    ok = fail = 0
    for i, rec in enumerate(todo, 1):
        if _free_gb(STATUTES_DIR) < args.min_free_gb:
            save_manifest(records, args.manifest)
            print(f"  STOP: free disk below {args.min_free_gb} GB — halting safely after "
                  f"{ok} downloads. Free space (or attach a drive), then re-run to resume.", flush=True)
            break
        download(session, rp, rec, args.delay)
        if rec.downloaded:
            ok += 1
        else:
            fail += 1
            print(f"  [{i}/{len(todo)}] SKIP {rec.title[:50]}: {rec.error}", flush=True)
        if i % 25 == 0:
            print(f"  …{i}/{len(todo)} ({ok} ok, {fail} failed)", flush=True)
            save_manifest(records, args.manifest)  # periodic checkpoint
    save_manifest(records, args.manifest)
    print(f"Done. {ok} downloaded, {fail} failed/skipped.\n"
          f"  statutes → {STATUTES_DIR}\n  cases    → {CASES_DIR}\n  documents→ {DOCS_DIR}")


if __name__ == "__main__":
    main()
