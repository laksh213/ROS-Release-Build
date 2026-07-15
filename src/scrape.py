"""Phase 1 — Automatic judgement scraper for the Supreme Court of Sri Lanka.

Two sources, merged by PDF filename:

  A. Archive table   https://supremecourt.lk/judgements/
     One server-rendered page (~2,580 rows) carrying rich metadata:
     date, case no, parties, judge(s), keywords, legislation, pdf url.
     Despite using DataTables for client-side filtering, the full dataset is
     in the initial HTML — a single GET retrieves everything.

  B. Directory index https://supremecourt.lk/wp-content/uploads/judgements/
     Apache autoindex of ~3,790 PDFs — the completeness spine (older files
     not present in the table).

Outputs:
  data/sc_judgements/<filename>.pdf   downloaded PDFs
  data/manifest.json                  one record per judgement (audit trail)

Polite by default: realistic User-Agent, delay between requests, retry with
backoff, and skips files already downloaded (resumable).

CLI:
  python -m src.scrape --metadata-only      # build manifest, no downloads
  python -m src.scrape --limit 10           # download first 10 (testing)
  python -m src.scrape --year 2024          # only that year's judgements
  python -m src.scrape                       # full crawl + download
  python -m src.scrape --dry-run            # print plan, touch nothing
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE = "https://supremecourt.lk"
ARCHIVE_URL = f"{BASE}/judgements/"
INDEX_URL = f"{BASE}/wp-content/uploads/judgements/"

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "sc_judgements"
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 ROScribe/0.1 (legal research; respectful crawler)"
)


@dataclass
class JudgementRecord:
    filename: str
    pdf_url: str
    case_no: str = ""
    date: str = ""            # ISO (YYYY-MM-DD) when available
    date_text: str = ""       # human-readable as shown on the site
    parties: str = ""
    judges: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    legislation: list[str] = field(default_factory=list)
    source: str = ""          # "archive" | "index" | "both"
    downloaded: bool = False
    local_path: str = ""
    scraped_at: str = ""


# --------------------------------------------------------------------------- #
# HTTP                                                                         #
# --------------------------------------------------------------------------- #
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def fetch(session: requests.Session, url: str, retries: int = 3) -> requests.Response:
    """GET with simple exponential backoff."""
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- #
# Parsing                                                                      #
# --------------------------------------------------------------------------- #
def _filename_from_url(url: str) -> str:
    return unquote(urlparse(url).path.rsplit("/", 1)[-1])


def _split(text: str) -> list[str]:
    parts = re.split(r"[;,\n]+", text or "")
    return [p.strip() for p in parts if p.strip()]


def parse_archive(html: str) -> dict[str, JudgementRecord]:
    """Parse the /judgements/ archive table into records keyed by filename."""
    soup = BeautifulSoup(html, "html.parser")
    records: dict[str, JudgementRecord] = {}

    for row in soup.select("tr"):
        link = row.find("a", href=re.compile(r"\.pdf", re.I))
        if not link:
            continue
        pdf_url = urljoin(BASE, link["href"])
        filename = _filename_from_url(pdf_url)
        cells = row.find_all("td")

        def cell_text(sel: str, idx: int) -> str:
            node = row.select_one(sel)
            if node and node.get_text(strip=True):
                return node.get_text(" ", strip=True)
            return cells[idx].get_text(" ", strip=True) if idx < len(cells) else ""

        date_node = row.select_one("[data-order]")
        date_iso = date_node["data-order"] if date_node else ""

        judges = [
            re.sub(r"^[^A-Za-z]+", "", b.get_text(" ", strip=True))
            for b in row.select(".jm-fe-judge-badge")
        ]
        if not judges and len(cells) > 3:
            txt = cells[3].get_text(" ", strip=True)
            judges = [re.sub(r"^[^A-Za-z]+", "", txt)] if txt else []

        parties_node = row.select_one(".jm-fe-parties-cell .full-text")
        parties = (
            parties_node.get_text(" ", strip=True)
            if parties_node
            else (cells[2].get_text(" ", strip=True) if len(cells) > 2 else "")
        )

        records[filename] = JudgementRecord(
            filename=filename,
            pdf_url=pdf_url,
            case_no=cell_text(".jm-fe-case-code", 1),
            date=date_iso,
            date_text=cell_text(".jm-fe-date-badge", 0),
            parties=parties,
            judges=judges,
            keywords=_split(cells[4].get_text(" ", strip=True)) if len(cells) > 4 else [],
            legislation=_split(cells[5].get_text(" ", strip=True)) if len(cells) > 5 else [],
            source="archive",
        )
    return records


def parse_index(html: str) -> dict[str, str]:
    """Parse the Apache directory index into {filename: absolute_url}."""
    soup = BeautifulSoup(html, "html.parser")
    out: dict[str, str] = {}
    for a in soup.find_all("a", href=re.compile(r"\.pdf$", re.I)):
        url = urljoin(INDEX_URL, a["href"])
        out[_filename_from_url(url)] = url
    return out


# --------------------------------------------------------------------------- #
# Collect + merge                                                             #
# --------------------------------------------------------------------------- #
def collect(session: requests.Session) -> list[JudgementRecord]:
    archive = parse_archive(fetch(session, ARCHIVE_URL).text)
    index = parse_index(fetch(session, INDEX_URL).text)

    for filename, url in index.items():
        if filename in archive:
            archive[filename].source = "both"
        else:
            archive[filename] = JudgementRecord(
                filename=filename, pdf_url=url, source="index"
            )
    return sorted(archive.values(), key=lambda r: (r.date or "", r.filename), reverse=True)


# --------------------------------------------------------------------------- #
# Download                                                                     #
# --------------------------------------------------------------------------- #
def _safe_name(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", filename)


def download(session: requests.Session, rec: JudgementRecord, dest: Path, delay: float) -> None:
    path = dest / _safe_name(rec.filename)
    if path.exists() and path.stat().st_size > 0:
        rec.downloaded, rec.local_path = True, str(path)
        return
    resp = fetch(session, rec.pdf_url)
    path.write_bytes(resp.content)
    rec.downloaded, rec.local_path = True, str(path)
    rec.scraped_at = datetime.now(timezone.utc).isoformat()
    time.sleep(delay)


def save_manifest(records: list[JudgementRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in records], indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Scrape SL Supreme Court judgements.")
    ap.add_argument("--limit", type=int, default=None, help="max PDFs to download")
    ap.add_argument("--year", type=str, default=None, help="filter by year, e.g. 2024")
    ap.add_argument("--metadata-only", action="store_true", help="build manifest, no downloads")
    ap.add_argument("--dry-run", action="store_true", help="print plan, change nothing")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between downloads")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = ap.parse_args(argv)

    session = make_session()
    print("Collecting judgement listings ...")
    records = collect(session)
    if args.year:
        records = [r for r in records if (r.date or r.filename).startswith(args.year)]
    print(f"  {len(records)} judgements found "
          f"({sum(r.source != 'index' for r in records)} with table metadata).")

    if args.dry_run:
        for r in records[: args.limit or 10]:
            print(f"  - {r.date or '????'}  {r.case_no or r.filename}")
        print("Dry run — nothing written.")
        return

    save_manifest(records, args.manifest)
    print(f"Manifest written: {args.manifest} ({len(records)} records)")
    if args.metadata_only:
        return

    args.data_dir.mkdir(parents=True, exist_ok=True)
    todo = records if args.limit is None else records[: args.limit]
    for i, rec in enumerate(todo, 1):
        try:
            download(session, rec, args.data_dir, args.delay)
            print(f"  [{i}/{len(todo)}] {rec.filename}")
        except requests.RequestException as e:
            print(f"  [{i}/{len(todo)}] FAILED {rec.filename}: {e}")
    save_manifest(records, args.manifest)  # persist downloaded/local_path
    print(f"Done. PDFs in {args.data_dir}")


if __name__ == "__main__":
    main()
