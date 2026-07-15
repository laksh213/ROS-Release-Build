"""Phase 3 — Storage (zero-config, local files).

SQLite holds structured metadata (one row per judgement, from data/manifest.json
plus extracted fields) for fast filtering by judge / date / case_no. ChromaDB
holds the embeddings for semantic search. Both are plain local files under data/
— no server.

Embeddings: uses `BAAI/bge-m3` (multilingual) when sentence-transformers is
installed; otherwise falls back to Chroma's built-in default embedder (English,
light) so the pipeline runs out-of-the-box. The collection name encodes which,
so the two never collide. After `pip install -r requirements.txt`, re-run the
index to switch to bge-m3.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

# chromadb is imported lazily inside functions that use it to optimize startup times

from .config import settings
from .ingest import Chunk


def _embedder_tag() -> str:
    """Which embedder/collection — computed WITHOUT loading the model."""
    if os.getenv("ROSCRIBE_EMBEDDER", "").lower() == "default":
        return "default"
    import importlib.util
    if importlib.util.find_spec("sentence_transformers") is not None:
        return "bge_m3"
    return "default"


_TAG = _embedder_tag()
COLLECTION = f"judgements_{_TAG}"
_EF = None  # the heavy model is loaded lazily, only when actually embedding
_EF_LOCK = threading.RLock()  # two early callers must not both load the model
_COLLECTION_CACHE = None
_EMBEDDER_READY = False  # flipped by warm_embedder(); semantic search waits for it


def _get_ef():
    global _EF
    with _EF_LOCK:
        if _EF is None and _TAG == "bge_m3":
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            _EF = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
        return _EF  # None → Chroma's built-in default embedder


def get_collection():
    import chromadb
    global _COLLECTION_CACHE
    if _COLLECTION_CACHE is not None:
        return _COLLECTION_CACHE
    with _EF_LOCK:
        if _COLLECTION_CACHE is None:
            client = chromadb.PersistentClient(path=settings.chroma_dir)
            kwargs = {"name": COLLECTION, "metadata": {"hnsw:space": "cosine"}}
            ef = _get_ef()
            if ef is not None:
                kwargs["embedding_function"] = ef
            _COLLECTION_CACHE = client.get_or_create_collection(**kwargs)
    return _COLLECTION_CACHE


def embedder_ready() -> bool:
    return _EMBEDDER_READY


def warm_embedder() -> bool:
    """Load bge-m3 + the Chroma collection once (~10-20 s) so semantic search is
    instant afterwards. Safe to call from a background thread at app startup."""
    global _EMBEDDER_READY
    if _TAG != "bge_m3":
        return False
    if _EMBEDDER_READY:
        return True
    try:
        similarity_search("warm up", k=1)
        _EMBEDDER_READY = True
        return True
    except Exception as e:  # noqa: BLE001 — non-fatal: search degrades to FTS-only
        print(f"[store] embedder warm-up failed: {e}")
        return False


def init_db() -> sqlite3.Connection:
    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(settings.sqlite_path)
    con.execute(
        """CREATE TABLE IF NOT EXISTS judgements (
            case_no TEXT, filename TEXT PRIMARY KEY, date TEXT, parties TEXT,
            judges TEXT, keywords TEXT, legislation TEXT, pdf_url TEXT,
            local_path TEXT, n_chunks INTEGER, indexed_at TEXT)"""
    )
    # Collection-aware index state — survives crashes; correct across embedder switches.
    con.execute(
        """CREATE TABLE IF NOT EXISTS indexed (
            collection TEXT, source TEXT, key TEXT, n_chunks INTEGER, indexed_at TEXT,
            PRIMARY KEY (collection, source, key))"""
    )
    # On-demand breakdown cache.
    con.execute(
        """CREATE TABLE IF NOT EXISTS analyses (
            case_no TEXT PRIMARY KEY, model TEXT, json TEXT, created_at TEXT)"""
    )
    # Statutes / Acts (lankalaw.net). `statute_id` is the canonical label used as
    # the chunk's case_no (so chunks ↔ row tie exactly like judgements). Lets the
    # breakdown's "Legislation Cited" links resolve to the actual statute text.
    con.execute(
        """CREATE TABLE IF NOT EXISTS statutes (
            statute_id TEXT PRIMARY KEY, title TEXT, short_name TEXT, act_no TEXT,
            year TEXT, kind TEXT, filename TEXT, local_path TEXT, pdf_url TEXT,
            source_url TEXT, n_chunks INTEGER, indexed_at TEXT)"""
    )
    # Migration: law-report citation ("68 NLR", "SLR 1982") for lankalaw report
    # cases — they carry no court case number, so this is their resolvable id.
    cols = {r[1] for r in con.execute("PRAGMA table_info(judgements)")}
    if "report_cite" not in cols:
        con.execute("ALTER TABLE judgements ADD COLUMN report_cite TEXT DEFAULT ''")
    con.commit()
    return con


def is_indexed(con: sqlite3.Connection, key: str, source: str) -> bool:
    """Has this file's chunks already been embedded into the current collection?"""
    return con.execute(
        "SELECT 1 FROM indexed WHERE collection=? AND source=? AND key=?",
        (COLLECTION, source, key),
    ).fetchone() is not None


def mark_indexed(con: sqlite3.Connection, key: str, source: str, n_chunks: int) -> None:
    con.execute(
        "INSERT OR REPLACE INTO indexed (collection, source, key, n_chunks, indexed_at) "
        "VALUES (?,?,?,?,?)",
        (COLLECTION, source, key, n_chunks, datetime.now().isoformat(timespec="seconds")),
    )
    con.commit()


def get_analysis(con: sqlite3.Connection, case_no: str) -> dict | None:
    row = con.execute("SELECT json FROM analyses WHERE case_no=?", (case_no,)).fetchone()
    return json.loads(row[0]) if row else None


def save_analysis(con: sqlite3.Connection, case_no: str, data: dict, model: str) -> None:
    con.execute(
        "INSERT OR REPLACE INTO analyses (case_no, model, json, created_at) VALUES (?,?,?,?)",
        (case_no, model, json.dumps(data, ensure_ascii=False),
         datetime.now().isoformat(timespec="seconds")),
    )
    con.commit()


def upsert_judgement(con: sqlite3.Connection, meta: dict, n_chunks: int) -> None:
    con.execute(
        """INSERT OR REPLACE INTO judgements
           (case_no, filename, date, parties, judges, keywords, legislation,
            pdf_url, local_path, report_cite, n_chunks, indexed_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            meta.get("case_no", ""), meta.get("filename", ""), meta.get("date", ""),
            meta.get("parties", ""), json.dumps(meta.get("judges", [])),
            json.dumps(meta.get("keywords", [])), json.dumps(meta.get("legislation", [])),
            meta.get("pdf_url", ""), meta.get("local_path", ""),
            meta.get("report_cite", ""), n_chunks,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    con.commit()


def statute_label(title: str, act_no: str = "", year: str = "") -> str:
    """Canonical, readable, near-unique id for a statute — used as the chunk
    `case_no` so anchors read like `[Online Safety Act, No. 9 of 2024 | p3:2]`."""
    title = re.sub(r"\s+", " ", (title or "").strip()).rstrip(".,")
    act_no, year = (act_no or "").strip(), (year or "").strip()
    if act_no and year:
        return f"{title}, No. {act_no} of {year}"
    if year and year not in title:
        return f"{title} ({year})"
    return title


def statute_short_name(title: str) -> str:
    """A terse alias to also match on. Strips lankalaw's listing prefixes
    ('Chap 19 : Penal Code' → 'Penal Code', '25/2021 : Penal Code (Amendment)' →
    'Penal Code (Amendment)') and a trailing 'of Sri Lanka' ('Constitution of Sri
    Lanka' → 'Constitution'). Keeps '(Amendment)' so an amendment never aliases to
    the base Act."""
    s = re.sub(r"\s+", " ", (title or "").strip())
    s = re.sub(r"^\s*chap(?:ter)?\.?\s*\d+\s*[:\-–]\s*", "", s, flags=re.I)
    s = re.sub(r"^\s*\d{1,3}\s*/\s*\d{2,4}\s*[:\-–]\s*", "", s)
    s = re.sub(r"\s+of\s+Sri\s+Lanka\s*$", "", s, flags=re.I).strip()
    return s or title


def upsert_statute(con: sqlite3.Connection, meta: dict, n_chunks: int) -> None:
    con.execute(
        """INSERT OR REPLACE INTO statutes
           (statute_id, title, short_name, act_no, year, kind, filename,
            local_path, pdf_url, source_url, n_chunks, indexed_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            meta.get("statute_id", ""), meta.get("title", ""),
            meta.get("short_name") or statute_short_name(meta.get("title", "")),
            meta.get("act_no", ""), meta.get("year", ""), meta.get("kind", "statute"),
            meta.get("filename", ""), meta.get("local_path", ""), meta.get("pdf_url", ""),
            meta.get("source_url", ""), n_chunks,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    con.commit()
    _invalidate_statute_index()


def get_statute(statute_id: str) -> dict | None:
    """Full statute row by its canonical id (what resolve_statute returns)."""
    con = sqlite3.connect(settings.sqlite_path)
    try:
        cur = con.execute("SELECT * FROM statutes WHERE statute_id=?", (statute_id,))
        row = cur.fetchone()
        cols = [d[0] for d in cur.description]
    except sqlite3.OperationalError:
        return None
    finally:
        con.close()
    return dict(zip(cols, row)) if row else None


def list_statutes() -> list[dict]:
    """All statute rows (id, title, act_no, year, n_chunks) — newest year first."""
    con = sqlite3.connect(settings.sqlite_path)
    try:
        rows = con.execute(
            "SELECT statute_id, title, act_no, year, n_chunks FROM statutes "
            "ORDER BY year DESC, title"
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    con.close()
    return [{"statute_id": a, "title": b, "act_no": c, "year": d, "n_chunks": e}
            for a, b, c, d, e in rows]


def add_chunks(chunks: list[Chunk], extra_meta: dict | None = None) -> None:
    """Embed and upsert chunks into Chroma (idempotent on chunk id)."""
    if not chunks:
        return
    col = get_collection()
    ids, docs, metas = [], [], []
    for i, c in enumerate(chunks):
        ids.append(f"{c.case_no}|p{c.page}|{i}")
        docs.append(c.text)
        m = {
            "case_no": c.case_no, "page": c.page, "para": c.para or "",
            "source": c.source, "anchor": c.anchor(),
        }
        m.update({k: v for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))})
        if extra_meta:
            m.update({k: v for k, v in extra_meta.items() if isinstance(v, (str, int, float))})
        metas.append(m)
    col.upsert(ids=ids, documents=docs, metadatas=metas)


def similarity_search(query: str, k: int = 20, source: str | None = None) -> list[dict]:
    col = get_collection()
    where = {"source": source} if source else None
    res = col.query(query_texts=[query], n_results=k, where=where)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "meta": meta, "distance": dist})
    return hits


# --------------------- keyword / full-text search ------------------------ #
def build_fts(collection_name: str = "judgements_bge_m3", rebuild: bool = False) -> None:
    """One-time: build a SQLite FTS5 index over chunk text for keyword search.

    Reads documents straight from Chroma (no embedding model needed)."""
    import chromadb

    con = sqlite3.connect(settings.sqlite_path)
    if rebuild:
        con.execute("DROP TABLE IF EXISTS chunks_fts")
    con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(case_no, source, page UNINDEXED, text)")
    con.commit()
    if con.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] and not rebuild:
        print("FTS already built."); con.close(); return

    col = chromadb.PersistentClient(path=settings.chroma_dir).get_collection(collection_name)
    total = col.count()
    print(f"Building FTS over {total} chunks from {collection_name} …", flush=True)
    BATCH, done = 5000, 0
    for off in range(0, total, BATCH):
        res = col.get(include=["documents", "metadatas"], limit=BATCH, offset=off)
        rows = [
            (m.get("case_no", ""), m.get("source", ""), str(m.get("page", "")), d or "")
            for d, m in zip(res["documents"], res["metadatas"])
        ]
        con.executemany("INSERT INTO chunks_fts (case_no, source, page, text) VALUES (?,?,?,?)", rows)
        con.commit()
        done += len(rows)
        print(f"  {done}/{total}", flush=True)
    con.close()
    print("FTS built.", flush=True)


def fts_index_cases(case_nos: list[str], collection_name: str | None = None) -> int:
    """Incrementally add (or refresh) FTS rows for specific cases — used by the
    monthly corpus update so new judgements become keyword-searchable without a
    full rebuild over the whole corpus. Idempotent: existing rows for a case are
    replaced. Reads chunk text straight from Chroma (no embedding model loaded).
    Returns the number of chunk rows written."""
    if not case_nos:
        return 0
    import chromadb

    name = collection_name or COLLECTION
    con = sqlite3.connect(settings.sqlite_path)
    con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(case_no, source, page UNINDEXED, text)")
    con.commit()
    try:
        col = chromadb.PersistentClient(path=settings.chroma_dir).get_collection(name)
    except Exception as e:  # noqa: BLE001 — collection missing → nothing to index
        print(f"[fts] collection {name!r} unavailable: {e}")
        con.close()
        return 0
    added = 0
    for cn in case_nos:
        con.execute("DELETE FROM chunks_fts WHERE case_no=?", (cn,))
        res = col.get(where={"case_no": cn}, include=["documents", "metadatas"])
        docs, metas = res.get("documents") or [], res.get("metadatas") or []
        rows = [
            (m.get("case_no", ""), m.get("source", ""), str(m.get("page", "")), d or "")
            for d, m in zip(docs, metas)
        ]
        if rows:
            con.executemany("INSERT INTO chunks_fts (case_no, source, page, text) VALUES (?,?,?,?)", rows)
            added += len(rows)
    con.commit()
    con.close()
    return added


# --- query understanding: acronyms, stopwords, tiered FTS ----------------- #

# Common Sri Lankan legal acronyms <-> their expansions. Queries are expanded in
# BOTH directions ("RDA" also finds "Road Development Authority" and vice versa)
# because judgments switch freely between the two forms. Extend as needed.
LEGAL_ACRONYMS: dict[str, list[str]] = {
    "rda": ["road development authority"],
    "uda": ["urban development authority"],
    "ceb": ["ceylon electricity board"],
    "cpc": ["ceylon petroleum corporation", "civil procedure code"],
    "slpa": ["sri lanka ports authority"],
    "nwsdb": ["national water supply and drainage board"],
    "boi": ["board of investment"],
    "cbsl": ["central bank of sri lanka"],
    "epf": ["employees provident fund"],
    "etf": ["employees trust fund"],
    "ag": ["attorney general"],
    "igp": ["inspector general of police"],
    "oic": ["officer in charge"],
    "cid": ["criminal investigation department"],
    "fcid": ["financial crimes investigation division"],
    "tid": ["terrorist investigation division"],
    "pta": ["prevention of terrorism act"],
    "iccpr": ["international covenant on civil and political rights"],
    "nic": ["national identity card"],
    "rti": ["right to information"],
    "hrcsl": ["human rights commission"],
    "hrc": ["human rights commission"],
    "jsc": ["judicial service commission"],
    "psc": ["public service commission"],
    "nsb": ["national savings bank"],
    "boc": ["bank of ceylon"],
    "slic": ["sri lanka insurance"],
    "sltb": ["sri lanka transport board"],
    "ctb": ["ceylon transport board"],
    "cmc": ["colombo municipal council"],
    "cea": ["central environmental authority"],
    "mepa": ["marine environment protection authority"],
    "ugc": ["university grants commission"],
    "slmc": ["sri lanka medical council"],
    "trc": ["telecommunications regulatory commission"],
    "caa": ["consumer affairs authority"],
    "vat": ["value added tax"],
    "ltte": ["liberation tigers of tamil eelam"],
    "slas": ["sri lanka administrative service"],
}

# Words that carry no signal in a case search ("cases involving the RDA").
# Loose tokens matching these are dropped; quoted phrases are never touched.
_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "could", "did",
    "do", "does", "for", "from", "had", "has", "have", "he", "her", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "me", "my", "no", "not", "of",
    "on", "or", "our", "s", "she", "should", "so", "such", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those", "to",
    "under", "up", "upon", "was", "we", "were", "what", "when", "where", "which",
    "who", "whose", "why", "will", "with", "would", "you", "your",
    "case", "cases", "involving", "involve", "involved", "involves", "related",
    "relating", "relate", "regarding", "concerning", "concern", "about", "matter",
    "matters", "any", "all", "find", "show", "search", "judgment", "judgments",
    "judgement", "judgements", "decision", "decisions",
}

_ACRO_NGRAMS: dict[tuple[str, ...], str] | None = None  # expansion tokens -> acronym


def _tokenize(text: str) -> list[str]:
    """Unicode-aware tokens (keeps Sinhala/Tamil — FTS5 unicode61 indexes them)."""
    return [t.lower() for t in re.findall(r"\w+", text or "")]


def _acro_ngrams() -> dict[tuple[str, ...], str]:
    global _ACRO_NGRAMS
    if _ACRO_NGRAMS is None:
        _ACRO_NGRAMS = {}
        for acro, exps in LEGAL_ACRONYMS.items():
            for e in exps:
                _ACRO_NGRAMS[tuple(_tokenize(e))] = acro
    return _ACRO_NGRAMS


# Concept synonyms — a query word -> related legal terms, so a lay phrasing reaches
# the term the judgments actually use ("sacking" -> "termination of employment").
# Kept small and high-precision; the embedding (deep) search covers the long tail.
LEGAL_SYNONYMS: dict[str, list[str]] = {
    "dismissal": ["termination of employment", "termination of services"],
    "sacking": ["termination of employment", "dismissal"],
    "fired": ["termination of employment", "dismissal"],
    "bribery": ["corruption", "illicit enrichment"],
    "corruption": ["bribery", "illicit enrichment"],
    "negligence": ["delict", "duty of care"],
    "defamation": ["libel", "slander"],
    "bail": ["enlargement on bail", "remand"],
    "eviction": ["ejectment"],
    "rape": ["sexual assault", "grave sexual abuse"],
    "murder": ["culpable homicide"],
    "divorce": ["dissolution of marriage", "matrimonial"],
    "inheritance": ["intestate succession", "last will", "testamentary"],
    "pension": ["gratuity", "retirement benefits"],
    "torture": ["cruel inhuman or degrading treatment", "article 11"],
    "disappearance": ["habeas corpus", "enforced disappearance"],
}


# --- fuzzy term matching: bridge spelling / transliteration variants of names --- #
# Sri Lankan names transliterate many ways — the SC corpus alone spells one family
# Rajapaksa / Rajapakse / Rajapaksha / Rajapakshe / Rajapakshalage ... . FTS5 is
# exact-token, so one spelling silently misses the rest. We build a lexicon of the
# terms that actually occur in the corpus (names live in parties/judges/keywords),
# index it by character trigram for fast candidate lookup, and expand a query token
# to the close variants that EXIST in the corpus — a corpus-grounded "did you mean",
# never an invented word. This is the core of the smart/AI search: typing any one
# spelling (or a typo) surfaces every spelling of the same name.
_LEXICON: set[str] | None = None
_TRIGRAM_IX: dict[str, set[str]] | None = None
_FUZZY_CACHE: dict[str, list[str]] = {}


def _trigrams(s: str) -> set[str]:
    """Character trigrams with start/end padding so prefixes & suffixes count."""
    s = f"  {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _build_lexicon() -> None:
    """One-time: collect the corpus's own vocabulary (>=4-letter tokens from the
    party / judge / keyword metadata) and a trigram inverted index over it."""
    global _LEXICON, _TRIGRAM_IX
    if _LEXICON is not None:
        return
    lex: set[str] = set()
    con = sqlite3.connect(settings.sqlite_path)
    try:
        for col in ("parties", "judges", "keywords"):
            try:
                rows = con.execute(
                    f"SELECT {col} FROM judgements WHERE {col} IS NOT NULL AND {col}!=''"
                )
            except sqlite3.OperationalError:
                continue
            for (v,) in rows:
                for tok in re.findall(r"[A-Za-z]{4,}", v or ""):
                    t = tok.lower()
                    if t not in _QUERY_STOPWORDS:
                        lex.add(t)
    finally:
        con.close()
    ix: dict[str, set[str]] = {}
    for term in lex:
        for tri in _trigrams(term):
            ix.setdefault(tri, set()).add(term)
    _LEXICON, _TRIGRAM_IX = lex, ix


def fuzzy_variants(token: str, max_variants: int = 6, min_ratio: float = 0.82) -> list[str]:
    """Corpus terms that are close spelling / transliteration variants of `token`
    (e.g. 'rajapakshe' -> ['rajapaksa', 'rajapakse', 'rajapaksha', ...]). Trigram
    pre-filtered, confirmed by edit-distance ratio. Returns ONLY terms present in
    the corpus, most-similar first. Empty for short tokens (<5 chars)."""
    from difflib import SequenceMatcher

    token = (token or "").lower()
    if len(token) < 5:
        return []
    if token in _FUZZY_CACHE:
        return _FUZZY_CACHE[token]
    _build_lexicon()
    assert _TRIGRAM_IX is not None
    qtri = _trigrams(token)
    cand: dict[str, int] = {}
    for tri in qtri:
        for term in _TRIGRAM_IX.get(tri, ()):
            cand[term] = cand.get(term, 0) + 1
    scored: list[tuple[float, str]] = []
    for term, shared in cand.items():
        if term == token:
            continue
        # cheap recall guard: share enough trigrams and be within ~half the length
        if shared < 2 and len(token) > 6:
            continue
        if abs(len(term) - len(token)) > max(3, len(token) // 2):
            continue
        ratio = SequenceMatcher(None, token, term).ratio()
        if ratio >= min_ratio:
            scored.append((ratio, term))
    scored.sort(reverse=True)
    out = [t for _r, t in scored[:max_variants]]
    _FUZZY_CACHE[token] = out
    return out


def _query_units(tokens: list[str]) -> list[list[str]]:
    """Collapse the token stream into search units (each a list of equivalent
    variants): known entity phrases become [phrase, acronym], acronyms expand to
    [acronym, *expansions], plain tokens stay single. Stopwords drop out."""
    ngrams = _acro_ngrams()
    max_n = max((len(k) for k in ngrams), default=1)
    units: list[list[str]] = []
    i = 0
    while i < len(tokens):
        hit = None
        for n in range(min(max_n, len(tokens) - i), 1, -1):
            acro = ngrams.get(tuple(tokens[i:i + n]))
            if acro:
                hit = (n, acro)
                break
        if hit:
            n, acro = hit
            units.append([" ".join(tokens[i:i + n]), acro])
            i += n
            continue
        t = tokens[i]
        if t in LEGAL_ACRONYMS:
            units.append([t, *LEGAL_ACRONYMS[t]])
        elif t not in _QUERY_STOPWORDS:
            variants = [t]
            for syn in LEGAL_SYNONYMS.get(t, []):     # concept synonyms (sacking -> termination)
                if syn not in variants:
                    variants.append(syn)
            for fv in fuzzy_variants(t):              # spelling/transliteration variants in the corpus
                if fv not in variants:
                    variants.append(fv)
            units.append(variants)
        i += 1
    if not units:  # the whole query was stopwords — search it literally
        units = [[t] for t in tokens]
    return units


def _fts_group(variants: list[str], prefix: bool = False) -> str:
    """One unit -> FTS5 syntax: ("rda" OR "road development authority")."""
    star = "*" if prefix else ""
    parts = [f'"{v}"{star}' for v in variants]
    return "(" + " OR ".join(parts) + ")" if len(parts) > 1 else parts[0]


def _fts_queries(query: str) -> list[tuple[str, str]]:
    """Translate a natural-language query into tiered FTS5 MATCH strings, tried
    in order: ("phrase", …) exact phrases → ("all", …) every term anywhere in a
    chunk → ("any", …) at least one term (last-resort, only used when the
    stricter tiers return nothing). Honours "quoted phrases", drops query noise
    ("cases involving the…"), and expands acronyms in both directions."""
    q = (query or "").replace("“", '"').replace("”", '"').strip()
    if not q:
        return []
    quoted = [p.strip() for p in re.findall(r'"([^"]+)"', q) if p.strip()]
    rest = re.sub(r'"[^"]*"?', " ", q)
    tokens = _tokenize(rest)
    units = _query_units(tokens) if tokens else []

    groups: list[list[str]] = []
    for ph in quoted:  # quoted phrases kept verbatim (plus acronym variant if known)
        key = tuple(_tokenize(ph))
        if not key:
            continue
        acro = _acro_ngrams().get(key)
        groups.append([" ".join(key), acro] if acro else [" ".join(key)])
    groups += units
    if not groups:
        return []

    # While the user is mid-word, prefix-match the last term ("bunker fu" works).
    # Single characters are not starred — "r"* would expand to a huge term set.
    star_last = bool(units) and bool(tokens) and len(tokens[-1]) >= 2 \
        and bool(re.search(r"\w$", q)) and units[-1][0].split(" ")[-1] == tokens[-1]

    def render(gs: list[list[str]], op: str, star: bool) -> str:
        return f" {op} ".join(
            _fts_group(g, prefix=(star and i == len(gs) - 1)) for i, g in enumerate(gs)
        )

    # Tier 1: runs of consecutive plain tokens become exact phrases
    # ("bunker fuel related cases" -> "bunker fuel").
    t1_groups: list[list[str]] = [g for g in groups[: len(groups) - len(units)]]
    run: list[str] = []

    def _flush_run():
        if len(run) >= 2:
            t1_groups.append([" ".join(run)])
        elif run:
            t1_groups.append([run[0]])
        run.clear()

    for u in units:
        if len(u) == 1 and " " not in u[0]:
            run.append(u[0])
        else:
            _flush_run()
            t1_groups.append(u)
    _flush_run()

    t2 = render(groups, "AND", star_last)
    t1 = render(t1_groups, "AND", False)
    tiers: list[tuple[str, str]] = []
    if t1 != t2:
        tiers.append(("phrase", t1))
    tiers.append(("all", t2))
    if len(groups) > 1:
        tiers.append(("any", render(groups, "OR", star_last)))
    return tiers


_WHY_TIER = {"metadata": 0, "phrase": 1, "text": 2, "broad": 3, "semantic": 4}


def _date_key(d: str) -> int:
    digits = re.sub(r"\D", "", (d or "")[:10])
    return int(digits) if digits else 0


def _rank_key(h: dict) -> tuple:
    """Sort: tier (metadata → phrase → all-terms → broad → semantic), then
    relevance (bm25/distance — lower is better), then newest first."""
    return (h.get("tier", 9), h.get("score", 0.0), -_date_key(h.get("date", "")))


def _metadata_variants(query: str) -> list[str]:
    """Strings to LIKE against metadata: the raw query plus acronym expansions
    (so "RDA" also matches parties named "Road Development Authority")."""
    out = [query]
    toks = _tokenize(query)
    if len(toks) == 1 and toks[0] in LEGAL_ACRONYMS:
        out += LEGAL_ACRONYMS[toks[0]]
    acro = _acro_ngrams().get(tuple(toks))
    if acro:
        out.append(acro)
    # bridge spelling variants of a single-name query so the parties/judges LIKE
    # matches every transliteration (Rajapakshe -> Rajapaksa, Rajapakse, ...).
    if len(toks) == 1:
        out += fuzzy_variants(toks[0])
    return list(dict.fromkeys(out))[:12]


def _query_case_hits(con: sqlite3.Connection, query: str, per_tier: int = 400) -> dict[str, dict]:
    """The one engine behind keyword/combined search. Returns
    {case_no: {tier, score, hits, snippet, date, why}} — metadata LIKE matches
    plus tiered FTS matches ranked by bm25, best matching chunk as the snippet."""
    q = (query or "").strip()
    out: dict[str, dict] = {}
    if not q:
        return out

    for v in _metadata_variants(q):
        like = f"%{v}%"
        # Substring LIKE on a short token ("RDA" in "Jayawardana") is noise —
        # short single words get a word-boundary check instead.
        boundary = re.compile(rf"\b{re.escape(v)}", re.IGNORECASE) if (len(v) < 6 and " " not in v) else None
        for cn, date, parties, judges, kws, leg in con.execute(
            "SELECT case_no, date, parties, judges, keywords, legislation FROM judgements "
            "WHERE case_no LIKE ? OR parties LIKE ? OR judges LIKE ? OR keywords LIKE ? OR legislation LIKE ? "
            "ORDER BY date DESC LIMIT ?",
            (like, like, like, like, like, per_tier),
        ):
            if boundary and not boundary.search(" ".join(filter(None, (cn, parties, judges, kws, leg)))):
                continue
            out.setdefault(cn, {"tier": 0, "score": 0.0, "hits": 1, "why": "metadata",
                                "snippet": (parties or "")[:110], "date": date or ""})

    for why, match in _fts_queries(q):
        if why == "any" and out:
            break  # broad OR-matching only when nothing stricter matched
        tier_why = {"phrase": "phrase", "all": "text", "any": "broad"}[why]
        try:
            # bm25()/snippet() must run in the FTS row context, and the inner
            # LIMIT stops SQLite flattening the subquery into the aggregate
            # (which raises "unable to use function bm25 in the requested
            # context"). min(score) keeps the best chunk's snippet+page per case.
            rows = con.execute(
                "SELECT sub.case_no, j.date, sub.snip, sub.page, count(*), min(sub.score) AS best "
                "FROM ("
                "  SELECT case_no, page, snippet(chunks_fts, 3, '«', '»', '…', 12) AS snip, "
                "         bm25(chunks_fts) AS score "
                "  FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 20000"
                ") sub JOIN judgements j ON j.case_no = sub.case_no "
                "GROUP BY sub.case_no ORDER BY best LIMIT ?",
                (match, per_tier),
            ).fetchall()
        except sqlite3.OperationalError:  # no FTS table / malformed edge case
            continue
        for cn, date, snip, page, nhits, score in rows:
            out.setdefault(cn, {"tier": _WHY_TIER[tier_why], "score": float(score or 0.0),
                                "hits": int(nhits), "why": tier_why,
                                "snippet": snip or "", "date": date or "",
                                "page": int(page) if str(page or "").isdigit() else None})
    return out


def semantic_case_hits(query: str, k: int = 40, max_cases: int = 15) -> dict[str, dict]:
    """Embedding-based matches (bge-m3 + Chroma) for concept queries whose words
    differ from the judgment's ("bunker fuel" ≈ "furnace oil"). Returns {} until
    warm_embedder() has run, so it never blocks a search on a cold model."""
    if not _EMBEDDER_READY:
        return {}
    out: dict[str, dict] = {}
    try:
        for h in similarity_search(query, k=k, source="judgment"):
            cn = (h.get("meta") or {}).get("case_no") or ""
            d = float(h.get("distance") or 1.0)
            if not cn or d > 0.65:
                continue
            cur = out.get(cn)
            if cur is None:
                snip = " ".join((h.get("text") or "").split())[:110]
                pg = (h.get("meta") or {}).get("page")
                out[cn] = {"tier": 4, "score": d, "hits": 1, "why": "semantic",
                           "snippet": snip, "date": "",
                           "page": int(pg) if str(pg or "").isdigit() else None}
            else:
                cur["hits"] += 1
                cur["score"] = min(cur["score"], d)
    except Exception as e:  # noqa: BLE001 — semantic is a bonus, never break search
        print(f"[store] semantic search failed: {e}")
        return {}
    return dict(sorted(out.items(), key=lambda kv: kv[1]["score"])[:max_cases])


def keyword_search(query: str, limit: int = 60) -> list[dict]:
    """Keyword/phrase search over metadata + full judgment text. Understands
    "quoted phrases", drops query noise ("cases involving …"), expands SL legal
    acronyms (RDA ↔ Road Development Authority) and ranks by tier then bm25."""
    q = (query or "").strip()
    if not q:
        return []
    con = sqlite3.connect(settings.sqlite_path)
    hits = _query_case_hits(con, q)
    con.close()
    rows = [{"case_no": cn, "date": h["date"], "snippet": h["snippet"],
             "why": h["why"], "hits": h["hits"], "page": h.get("page")}
            for cn, h in hits.items()]
    rows.sort(key=lambda r: _rank_key(hits[r["case_no"]]))
    return rows[:limit]


# Curated legal-area taxonomy (real practice areas) -> case-no prefixes + the
# legal keywords that identify each area in the judgment text.
LEGAL_AREAS: dict[str, dict] = {
    "Fundamental Rights": {"prefix": ["SC/FR", "SC FR"], "terms": ["fundamental rights", "article 12", "article 126", "article 14", "equal protection"]},
    "Constitutional & Administrative": {"terms": ["writ of certiorari", "mandamus", "judicial review", "ultra vires", "natural justice", "legitimate expectation"]},
    "Labour & Employment": {"terms": ["labour tribunal", "termination of employment", "reinstatement", "industrial dispute", "workman", "unfair dismissal", "compensation in lieu"]},
    "Land & Property": {"terms": ["title to land", "ejectment", "deed of transfer", "co-owner", "encroachment", "declaration of title"]},
    "Partition": {"terms": ["partition action", "partition", "co-owners", "preliminary plan"]},
    "Prescription & Laches": {"terms": ["prescription", "laches", "adverse possession", "prescriptive title"]},
    "Trusts": {"terms": ["constructive trust", "resulting trust", "fiduciary", "trustee", "trust property"]},
    "Testamentary & Probate": {"terms": ["last will", "executor", "administrator", "intestate", "probate", "letters of administration"]},
    "Contract": {"terms": ["breach of contract", "consideration", "specific performance", "agreement to sell", "rescission"]},
    "Delict & Negligence": {"terms": ["negligence", "delict", "duty of care", "damages", "vicarious liability"]},
    "Defamation": {"terms": ["defamation", "libel", "slander"]},
    "Criminal Law & Procedure": {"terms": ["indictment", "penal code", "criminal procedure", "conviction", "culpable homicide", "sentence"]},
    "Bail": {"terms": ["bail", "remand", "anticipatory bail"]},
    "Evidence": {"terms": ["burden of proof", "admissibility", "evidence ordinance", "hearsay", "circumstantial evidence", "dock identification"]},
    "Civil Procedure": {"terms": ["civil procedure code", "summons", "plaint", "interlocutory", "summary procedure", "default judgment"]},
    "Commercial & Company": {"prefix": ["SC/CHC", "SC CHC"], "terms": ["company", "shares", "winding up", "director", "commercial high court", "shareholder"]},
    "Banking & Finance": {"terms": ["mortgage", "promissory note", "guarantee", "recovery of loans", "parate execution", "hypothecary"]},
    "Tax & Revenue": {"terms": ["income tax", "value added tax", "customs", "revenue", "tax assessment"]},
    "Intellectual Property": {"terms": ["trademark", "patent", "copyright", "passing off", "infringement"]},
    "Family & Matrimonial": {"terms": ["matrimonial", "divorce", "maintenance", "custody", "matrimonial home", "judicial separation"]},
    "Tenancy & Rent": {"terms": ["rent act", "tenant", "premises", "ejectment of tenant", "controlled premises"]},
    "Election Law": {"terms": ["election petition", "election", "franchise", "polling"]},
    "Bribery & Corruption": {"terms": ["bribery", "corruption", "commission to investigate allegations"]},
    "Arbitration": {"terms": ["arbitration", "arbitral award", "arbitration act"]},
    "Insurance": {"terms": ["insurance", "insurer", "policy of insurance", "indemnity"]},
    "Citizenship & Immigration": {"terms": ["citizenship", "immigration", "passport", "emigration"]},
    "Writ Applications": {"terms": ["writ", "certiorari", "mandamus", "prohibition", "quo warranto"]},
}


def area_search(area: str, limit: int = 120) -> list[dict]:
    """Cases for a curated legal area — by case-no prefix and/or FTS keywords."""
    spec = LEGAL_AREAS.get(area)
    if not spec:
        return []
    con = sqlite3.connect(settings.sqlite_path)
    results: dict[str, dict] = {}
    for pfx in spec.get("prefix", []):
        for cn, date, parties in con.execute(
            "SELECT case_no, date, parties FROM judgements WHERE case_no LIKE ? ORDER BY date DESC LIMIT ?",
            (pfx + "%", limit),
        ):
            results[cn] = {"case_no": cn, "date": date or "", "snippet": (parties or "")[:100]}
    if spec.get("terms"):
        fts_q = " OR ".join(f'"{t}"' for t in spec["terms"])
        try:
            for cn, date, snip in con.execute(
                "SELECT j.case_no, j.date, snippet(chunks_fts, 3, '«', '»', '…', 12) "
                "FROM chunks_fts f JOIN judgements j ON j.case_no = f.case_no "
                "WHERE chunks_fts MATCH ? LIMIT ?",
                (fts_q, limit * 4),
            ):
                results.setdefault(cn, {"case_no": cn, "date": date or "", "snippet": snip})
        except Exception:
            pass
    con.close()
    return sorted(results.values(), key=lambda r: r["date"], reverse=True)[:limit]


_MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"]


def _year_month_of(case_no: str, date: str) -> tuple[str | None, str | None]:
    """Best-effort (year, month-name) for a judgement: prefer the real `date`
    column, fall back to a 4-digit year embedded in the case number."""
    import re

    y = None
    if date and date[:4].isdigit():
        y = date[:4]
    else:
        yrs = [int(x) for x in re.findall(r"(?:19|20)\d{2}", case_no or "") if 1950 <= int(x) <= 2027]
        if yrs:
            y = str(max(yrs))
    m = None
    if date and len(date) >= 7 and date[5:7].isdigit():
        mi = int(date[5:7])
        if 1 <= mi <= 12:
            m = _MONTH_NAMES[mi - 1]
    return y, m


def _area_case_nos(con: sqlite3.Connection, area: str) -> set[str]:
    """Case numbers for a curated legal area (prefix + FTS terms) — ids only, no
    snippets, so it stays cheap inside an intersected/combined query."""
    spec = LEGAL_AREAS.get(area)
    if not spec:
        return set()
    out: set[str] = set()
    for pfx in spec.get("prefix", []):
        for (cn,) in con.execute("SELECT case_no FROM judgements WHERE case_no LIKE ?", (pfx + "%",)):
            out.add(cn)
    if spec.get("terms"):
        fts_q = " OR ".join(f'"{t}"' for t in spec["terms"])
        try:
            for (cn,) in con.execute("SELECT DISTINCT case_no FROM chunks_fts WHERE chunks_fts MATCH ?", (fts_q,)):
                out.add(cn)
        except Exception:
            pass
    return out


def _query_case_nos(con: sqlite3.Connection, query: str) -> set[str]:
    """Case numbers matching a free-text query across metadata + full text."""
    return set(_query_case_hits(con, query))


def clean_display_name(name: str) -> str:
    """Strip all titles (Hon., Dr., Mr.), suffixes (J., CJ., PC., Chief Justice),
    and punctuation. Normalize to Title Case if all uppercase."""
    s = (name or "").strip()
    # Leading titles
    s = re.sub(r"^(?:hon(?:\x27?ble|ourable|orable)?\.?\s*)?(?:(?:mr|mrs|ms|dr)\.?\s*)?(?:justice|judge)\s+", "", s, flags=re.I)
    s = re.sub(r"^(?:hon(?:\x27?ble|ourable|orable)?\.?\s*)?(?:(?:mr|mrs|ms|dr)\.?\s*)", "", s, flags=re.I)
    
    # Trailing suffixes
    s = re.sub(r"[\s,]+(?:p\.?c\.?|q\.?c\.?)?[\s,]+(?:c\.?\s*j\.?|a\.?c\.?j\.?|d\.?c\.?j\.?|j\.?j\.?|j\.?)\s*$", "", s, flags=re.I)
    s = re.sub(r"[\s,]+(?:p\.?c\.?|q\.?c\.?)\s*$", "", s, flags=re.I)
    s = re.sub(r"[\s,]+(?:j\.?|c\.?j\.?|p\.?c\.?|q\.?c\.?)\s*$", "", s, flags=re.I)
    s = re.sub(r"[\s,]+chief\s+justice\s*$", "", s, flags=re.I)
    s = re.sub(r"[\s,]+acting\s+chief\s+justice\s*$", "", s, flags=re.I)
    s = s.strip(",. ")
    
    if s.isupper():
        s = s.title()
    return s


def _canonical_justice(name: str) -> str:
    """Collapse spelling variants of one justice to a single key — strips titles
    (Hon./Dr./Justice), trailing silk/suffix (PC, J., C.J.), and punctuation so
    'J.A.N. de Silva CJ' == 'Hon. J.A.N. De Silva, C.J.'."""
    s = clean_display_name(name)
    return re.sub(r"[^a-z0-9]", "", s.lower())


# --- reindexed metadata: full benches (case_judges) + structured parties ---- #
# Populated by scripts/reindex.py. All readers degrade gracefully (fall back to
# the scrape metadata / raw parties string) when the reindex hasn't been run.
_HAS_CASE_JUDGES: bool | None = None


def _table_has_rows(con: sqlite3.Connection, name: str) -> bool:
    try:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone():
            return False
        return con.execute(f"SELECT 1 FROM {name} LIMIT 1").fetchone() is not None
    except sqlite3.OperationalError:
        return False


def has_full_benches() -> bool:
    """True once scripts/reindex.py has populated the case_judges table."""
    global _HAS_CASE_JUDGES
    if _HAS_CASE_JUDGES is None:
        con = sqlite3.connect(settings.sqlite_path)
        _HAS_CASE_JUDGES = _table_has_rows(con, "case_judges")
        con.close()
    return _HAS_CASE_JUDGES


def case_bench(case_no: str) -> list[str]:
    """Full coram (display names in seat order) from the reindexed case_judges
    table; [] if the case isn't reindexed (caller falls back to live parsing)."""
    con = sqlite3.connect(settings.sqlite_path)
    try:
        rows = con.execute(
            "SELECT display FROM case_judges WHERE case_no=? ORDER BY seat", (case_no,)
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    con.close()
    return [r[0] for r in rows if r[0]]


def parties_for(case_no: str) -> list[dict]:
    """Structured parties ({name, role, side}) from the reindexed parties_json
    column; [] if not reindexed (caller falls back to the raw parties string)."""
    con = sqlite3.connect(settings.sqlite_path)
    try:
        row = con.execute(
            "SELECT parties_json FROM judgements WHERE case_no=? OR filename=? LIMIT 1",
            (case_no, case_no),
        ).fetchone()
    except sqlite3.OperationalError:
        row = None
    con.close()
    if row and row[0]:
        try:
            return json.loads(row[0])
        except Exception:  # noqa: BLE001
            return []
    return []


_JUSTICE_GROUPS: dict[str, list[str]] | None = None


# --- justice-name canonicalisation (surname-anchored, OCR-tolerant) ---------- #
# Sri Lankan reports print a justice many ways: 'Priyasath Dep, PC, J',
# 'P. Dep PCJ', 'PRIYASATH DEP, PC, CJ' … plus OCR garbling ('Marsoof'/'Marsoor')
# and multi-judge / sentence fragments. We reduce each raw string to
# (givens, surname), then cluster: same surname (fuzzy) AND a shared full given
# name (or, for initials-only forms, a shared calling-name initial). Two judges
# with one surname ('Sarath de Abrew' vs 'Sisira de Abrew') stay separate.
_J_ROLES = {"j", "jj", "cj", "acj", "pcj", "dcj", "pc", "qc", "c", "p",
            "actg", "acting", "a", "cij"}
_J_PARTICLES = {"de", "del", "le", "la", "van", "von", "di", "das"}
_J_STOP = {"justice", "judge", "court", "application", "matter", "appeal",
           "petitioner", "respondent", "the", "of", "and", "also", "added",
           "reasoning", "another", "others"}
_J_LEAD = re.compile(r"^\s*(?:before|coram|present|quorum)\s*[:\-]?\s*", re.I)
_J_JUNK = re.compile(r"\b(agree|dissent|judge?ment|deliver|supreme|court|with|"
                     r"majority|opinion|order|application|hon|honou?rable)\b", re.I)
_J_HON = re.compile(r"^(?:the\s+)?(?:hon['’]?(?:ble|ourable|orable)?\.?\s*)?"
                    r"(?:(?:mr|mrs|ms|dr)\.?\s*)?(?:chief\s+)?(?:justice|judge)\s+", re.I)


def _j_parse(raw: str):
    """Raw judge string -> (givens:list[str], surname:str), or None if not a name."""
    s = _J_LEAD.sub("", raw or "").strip(" .,:;")
    m = re.search(r"judge?ment of\s+(?:the\s+)?(?:hon\.?\s*)?(?:justice\s+)?(.+)$", s, re.I)
    if m:
        s = m.group(1)
    s = _J_HON.sub("", s).strip(" .,:;")
    s = re.split(r"\s+(?:and|&|with)\s+", s, maxsplit=1, flags=re.I)[0]     # multi-judge -> first
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s).strip(" .,:;")               # glued CamelCase
    if not s or _J_JUNK.search(s):
        return None
    toks: list[str] = []
    for t in re.split(r"[\s,]+", s):
        t = t.strip(".")
        if not t:
            continue
        if "." in t or (t.isupper() and 1 < len(t) <= 3 and t.isalpha()):
            toks.extend(x for x in t.split(".") if x)
        else:
            toks.append(t)
    toks = [t for t in toks if t.isalpha()]
    while toks and (toks[-1].lower() in _J_ROLES or len(toks[-1]) == 1):
        toks.pop()
    if not toks:
        return None
    surname, givens = toks[-1], toks[:-1]
    if givens and givens[-1].lower() in _J_PARTICLES:
        surname = f"{givens[-1]} {surname}"
        givens = givens[:-1]
    if len(surname.replace(" ", "")) < 3 or surname.split()[-1].lower() in _J_STOP:
        return None
    return givens, surname


def _j_skey(surname: str) -> str:
    s = re.sub(r"[^a-z]", "", surname.lower()).replace("v", "w").replace("y", "i")
    s = re.sub(r"(.)\1+", r"\1", s)
    s = re.sub(r"([bcdfghjklmnpqrstvwxz])h", r"\1", s)
    return re.sub(r"[eou]", "a", s)


def _j_fulls(givens):
    return [_j_skey(g) for g in givens if len(g) >= 3]


def _j_sig(givens):
    ins = [g[0].lower() for g in givens if g]
    return {ins[0], ins[-1]} if ins else set()


def _j_particle(sur):
    parts = sur.split()
    if len(parts) > 1 and parts[0].lower() in _J_PARTICLES:
        return parts[0].lower(), " ".join(parts[1:])
    return "", sur


def _j_same(a, b) -> bool:
    (ga, sa), (gb, sb) = a, b
    pa, ba = _j_particle(sa)
    pb, bb = _j_particle(sb)
    if pa != pb:
        return False
    ka, kb = _j_skey(ba), _j_skey(bb)
    if len(ka) < 3 or len(kb) < 3:
        return ka == kb
    if not (ka == kb or difflib.SequenceMatcher(None, ka, kb).ratio() >= 0.82):
        return False
    fa, fb = _j_fulls(ga), _j_fulls(gb)
    if fa and fb:
        return any(x == y or difflib.SequenceMatcher(None, x, y).ratio() >= 0.80
                   for x in fa for y in fb)
    siga = _j_sig(ga) | {f[0] for f in fa}
    sigb = _j_sig(gb) | {f[0] for f in fb}
    if not siga or not sigb:
        return True
    return bool(siga & sigb)


def _j_display(givens, surname) -> str:
    def cap(w):
        return w if w.lower() in _J_PARTICLES else (w[:1].upper() + w[1:].lower()
                                                    if w.isupper() else w[:1].upper() + w[1:])
    parts = [f"{g}." if len(g) == 1 else cap(g) for g in givens]
    return " ".join(parts + [" ".join(cap(w) for w in surname.split())]).strip()


def justices_grouped() -> dict[str, list[str]]:
    """Deduped justices: {canonical display name -> [all raw spelling variants]}.
    Filtering expands a chosen display back to its variants (see _judge_case_nos)."""
    global _JUSTICE_GROUPS
    if _JUSTICE_GROUPS is None:
        con = sqlite3.connect(settings.sqlite_path)
        raw_counts: dict[str, int] = {}
        if _table_has_rows(con, "case_judges"):
            for (disp,) in con.execute("SELECT display FROM case_judges WHERE display IS NOT NULL AND display!=''"):
                raw_counts[disp] = raw_counts.get(disp, 0) + 1
        else:
            for (j,) in con.execute("SELECT judges FROM judgements WHERE judges IS NOT NULL AND judges!='[]'"):
                try:
                    arr = json.loads(j)
                except Exception:
                    continue
                for n in arr:
                    if n:
                        raw_counts[n] = raw_counts.get(n, 0) + 1
        con.close()

        # Parse once; cluster greedily (compare against any existing member).
        parsed = [(raw, c, p) for raw, c in raw_counts.items() if (p := _j_parse(raw))]
        clusters: list[dict] = []
        for raw, c, p in sorted(parsed, key=lambda t: -t[1]):
            for cl in clusters:
                if any(_j_same(p, mp) for _, _, mp in cl["items"]):
                    cl["items"].append((raw, c, p))
                    break
            else:
                clusters.append({"items": [(raw, c, p)]})

        groups: dict[str, list[str]] = {}
        for cl in clusters:
            best = max(cl["items"], key=lambda t: (len([g for g in t[2][0] if len(g) >= 3]), t[1]))
            groups[_j_display(*best[2])] = [raw for raw, _, _ in cl["items"]]
        _JUSTICE_GROUPS = groups
    return _JUSTICE_GROUPS


def distinct_justices() -> list[str]:
    """Deduped justice display names, alphabetical (one option per justice)."""
    return sorted(justices_grouped().keys(), key=str.lower)


def _judge_case_nos(con, names) -> set[str]:
    """Case numbers for one or more (deduped) justices — OR across the justices.
    The reindexed full-bench table matches every justice who SAT (not just the
    author); the scrape `judges` LIKE is unioned in as well, covering judgements
    added by a corpus update that haven't been bench-reindexed yet."""
    out: set[str] = set()
    if has_full_benches():
        for nm in names:
            canon = _canonical_justice(nm)
            try:
                for (cn,) in con.execute("SELECT case_no FROM case_judges WHERE canonical=?", (canon,)):
                    out.add(cn)
            except sqlite3.OperationalError:
                break
    groups = justices_grouped()
    for nm in names:
        for v in groups.get(nm, [nm]):   # expand a display name to its raw variants
            for (cn,) in con.execute("SELECT case_no FROM judgements WHERE judges LIKE ?", (f"%{v}%",)):
                out.add(cn)
    return out


def combined_search(judge=None, area: str | None = None,
                    year: str | None = None, month: str | None = None,
                    query: str | None = None, semantic: bool = False,
                    limit: int = 200) -> list[dict]:
    """AND-combine any subset of facets — Justice · legal area · year · month ·
    free-text — returning judgements that match *all* supplied facets. With a
    free-text query, results are relevance-ranked (phrase > all-terms > broad)
    with the best matching passage as the snippet; otherwise newest first.
    `semantic=True` additionally merges embedding matches (once the model is warm).

    Each facet contributes a set of case numbers; the result is their
    intersection. An empty facet is ignored; no facets returns []."""
    if isinstance(judge, (list, tuple, set)):
        judge = [str(x).strip() for x in judge if str(x).strip()] or None
    else:
        judge = (judge or "").strip() or None
    area = (area or "").strip() or None
    year = (year or "").strip() or None
    month = (month or "").strip() or None
    query = (query or "").strip() or None
    if not any((judge, area, year, month, query)):
        return []

    con = sqlite3.connect(settings.sqlite_path)
    sets: list[set[str]] = []
    qhits: dict[str, dict] = {}

    if judge:
        names = judge if isinstance(judge, list) else [judge]
        sets.append(_judge_case_nos(con, names))
    if area:
        sets.append(_area_case_nos(con, area))
    if query:
        qhits = _query_case_hits(con, query)
        if semantic:
            for cn, h in semantic_case_hits(query).items():
                qhits.setdefault(cn, h)
        sets.append(set(qhits))
    if year or month:
        ym: set[str] = set()
        for cn, date in con.execute("SELECT case_no, date FROM judgements"):
            yy, mm = _year_month_of(cn, date or "")
            if year and yy != year:
                continue
            if month and mm != month:
                continue
            ym.add(cn)
        sets.append(ym)

    common = set.intersection(*sets) if sets else set()
    if not common:
        con.close()
        return []

    # One cheap full scan for display metadata (table is ~3.8k rows; this avoids
    # SQLite's bound-variable limit that a big IN-clause would hit). GROUP BY
    # collapses the rare same-case_no twin rows (two different documents of one
    # case) into a single entry — the dated row's metadata rides along with max().
    rows = con.execute(
        "SELECT case_no, max(date), parties FROM judgements GROUP BY case_no"
    ).fetchall()
    con.close()
    out = []
    for cn, d, p in rows:
        if cn not in common:
            continue
        h = qhits.get(cn)
        out.append({"case_no": cn, "date": d or "",
                    "snippet": (h.get("snippet") or (p or "")[:110]) if h else (p or "")[:110],
                    "why": h.get("why", "") if h else "", "hits": h.get("hits", 0) if h else 0,
                    "page": h.get("page") if h else None})
    if qhits:
        out.sort(key=lambda r: _rank_key(qhits[r["case_no"]]) if r["case_no"] in qhits
                 else (9, 0.0, -_date_key(r["date"])))
    else:
        out.sort(key=lambda r: r["date"], reverse=True)
    return out[:limit]


# --- citation resolution: link a precedent citation to a corpus case_no ------ #
_CN_NORM: dict[str, str] = {}            # normalized case_no -> case_no
_CN_NUMYEAR: dict[str, list[str]] = {}   # "number/year4" -> [case_no, ...]
_CN_REPORT: dict[str, list[str]] = {}    # report_cite ("68 NLR"/"SLR 1982") -> [case_no, ...]

# Role words that appear in both citations and case names but identify nobody.
_CITE_STOP = {"appellant", "respondent", "petitioner", "defendant", "plaintiff",
              "others", "another", "the", "and", "law", "reports", "volume",
              "sri", "lanka", "ceylon"}


def _party_tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]{4,}", (s or "").lower()) if t not in _CITE_STOP}


def _report_keys(cited: str) -> list[str]:
    """Law-report keys in a citation string, in the canonical report_cite form:
      '(1943) 45 N.L.R. 73'    -> ['45 NLR']
      '[1982] 1 Sri L.R. 18'   -> ['SLR 1982']"""
    out: list[str] = []
    for m in re.finditer(r"\b(\d{1,3})\s*(?:N\.?\s?L\.?\s?R\.?\b|New\s+Law\s+Reports)",
                         cited or "", re.I):
        out.append(f"{int(m.group(1))} NLR")
    if re.search(r"\bSri\.?\s?L\.?\s?R\.?\b|\bSLR\b|Sri\s+Lanka\s+Law\s+Reports",
                 cited or "", re.I):
        for y in re.findall(r"\b(19\d{2}|20\d{2})\b", cited or ""):
            out.append(f"SLR {y}")
    return out


def _norm_alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _case_signatures(s: str) -> list[tuple[str, str]]:
    """All (number, 4-digit year) pairs in a string. Handles the citation
    variants that broke naive substring matching:
      'SC/FR/272/2016'           -> [('272','2016')]
      'SC FR Application 272/2016'-> [('272','2016')]   (the 'Application' token)
      'SC FR 446 19'             -> [('446','2019')]    (space sep, 2-digit year)
    """
    out: list[tuple[str, str]] = []
    for m in re.finditer(r"(\d{1,4})\s*[/\- ]\s*(\d{2,4})", s or ""):
        num, yr = m.group(1), m.group(2)
        yr4 = yr if len(yr) == 4 else (("20" if int(yr) < 50 else "19") + yr)
        out.append((num, yr4))
    return out


def _ensure_cite_index() -> None:
    global _CN_NORM, _CN_NUMYEAR, _CN_REPORT
    if _CN_NORM:
        return
    con = sqlite3.connect(settings.sqlite_path)
    norm: dict[str, str] = {}
    numyear: dict[str, list[str]] = {}
    report: dict[str, list[str]] = {}
    for cn, rc in con.execute(
            "SELECT case_no, COALESCE(report_cite, '') FROM judgements WHERE case_no IS NOT NULL"):
        nk = _norm_alnum(cn)
        if nk:
            norm.setdefault(nk, cn)
        for num, yr4 in _case_signatures(cn):
            numyear.setdefault(f"{num}/{yr4}", []).append(cn)
        if rc:
            report.setdefault(rc, []).append(cn)
    con.close()
    _CN_NORM, _CN_NUMYEAR, _CN_REPORT = norm, numyear, report


def resolve_citation(cited: str) -> str | None:
    """Resolve a precedent citation to a corpus case_no using the CASE NUMBER —
    the only reliable identifier. Tries exact normalized match, then number/year
    (confirmed by court-prefix letters when several cases share a number), then a
    literal bare-case_no substring.

    It deliberately does NOT guess by party name: Sri Lankan surnames recur across
    many cases, so party matching yields confident *wrong* links (e.g. a
    'Tilakaratne v Ariyaratne' citation matching an unrelated case that merely
    shares both surnames). Number-less citations are better handled in the UI as a
    search the user resolves.

    Law-report citations ('45 NLR 73', '[1982] 1 Sri L.R. 18') are the exception:
    the report volume/year narrows to one listing page (~50-150 cases), inside
    which a unique party-name match is safe — that's how the lankalaw NLR/SLR
    corpus (which has no court case numbers) becomes resolvable."""
    _ensure_cite_index()
    norm = _norm_alnum(cited)
    if not norm:
        return None
    if norm in _CN_NORM:
        return _CN_NORM[norm]
    for num, yr4 in _case_signatures(cited):
        cns = _CN_NUMYEAR.get(f"{num}/{yr4}")
        if not cns:
            continue
        if len(cns) == 1:
            return cns[0]
        for cn in cns:  # several cases share this number/year -> confirm by court prefix
            pref = re.sub(r"[^a-z]", "", cn.lower())
            if pref and pref in norm:
                return cn
    for key in _report_keys(cited):  # law-report cite -> volume/year, confirm by party
        cns = _CN_REPORT.get(key)
        if not cns:
            continue
        want = _party_tokens(cited)
        if not want:
            continue
        scored = sorted(((len(want & _party_tokens(cn)), cn) for cn in cns), reverse=True)
        if scored[0][0] >= 1 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            return scored[0][1]
    for k, cn in _CN_NORM.items():  # citation literally contains a bare case_no
        if len(k) >= 9 and k in norm:
            return cn
    return None


def citation_search_terms(cited: str) -> str:
    """Party-name portion of a citation, used for a fallback library search when
    it can't be confidently resolved (the user picks the right case from results).
    Strips court tokens, the number/year, and the 'v' separator."""
    s = re.sub(r"\b(SC|CA|HC|FR|CHC|APN|APPEAL|APPLICATION|No)\b\.?", " ", cited or "", flags=re.I)
    s = re.sub(r"\d{1,4}\s*[/\- ]\s*\d{2,4}", " ", s)
    s = re.sub(r"\bv[s]?\.?\b", " ", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


# --- statute resolution: link a "Legislation Cited" string to a corpus Act ---- #
# The breakdown emits legislation as free text — "Penal Code", "Section 304 of
# the Penal Code", "Online Safety Act, No. 9 of 2024", "Constitution Article 12".
# resolve_statute() maps such a string to a corpus statute_id (the canonical Act
# label), mirroring resolve_citation() for case law: exact normalized name, then
# Act No./year, then a contained statute name (so "...of the Penal Code" resolves
# to "Penal Code"). It deliberately matches on Act identity, not loose keywords.
_ST_NAME: dict[str, str] = {}             # normalized title / short_name -> statute_id
_ST_NUMYEAR: dict[str, list[str]] = {}    # "actno/year" -> [statute_id, ...]
_ST_NAMES_BY_LEN: list[tuple[str, str]] | None = None  # (norm_name, id), longest first


def _invalidate_statute_index() -> None:
    global _ST_NAME, _ST_NUMYEAR, _ST_NAMES_BY_LEN
    _ST_NAME, _ST_NUMYEAR, _ST_NAMES_BY_LEN = {}, {}, None


def _ensure_statute_index() -> None:
    global _ST_NAME, _ST_NUMYEAR, _ST_NAMES_BY_LEN
    if _ST_NAMES_BY_LEN is not None:
        return
    name: dict[str, str] = {}
    numyear: dict[str, list[str]] = {}
    con = sqlite3.connect(settings.sqlite_path)
    try:
        rows = con.execute(
            "SELECT statute_id, title, short_name, act_no, year FROM statutes"
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    con.close()
    for sid, title, short, act_no, year in rows:
        for nm in (title, short, sid):
            nk = _norm_alnum(nm)
            if nk:
                name.setdefault(nk, sid)
        if act_no and year:
            numyear.setdefault(f"{int(act_no)}/{year}", []).append(sid)
    # Longest names first so "penal code (amendment)" wins over "penal code" when
    # both are contained in the citation.
    names_by_len = sorted(
        {(_norm_alnum(t), sid) for sid, t, *_ in
         [(r[0], r[1]) for r in rows] + [(r[0], r[2]) for r in rows] if t},
        key=lambda kv: len(kv[0]), reverse=True,
    )
    _ST_NAME, _ST_NUMYEAR, _ST_NAMES_BY_LEN = name, numyear, names_by_len


def resolve_statute(cited: str) -> str | None:
    """Resolve a legislation reference to a corpus statute_id, or None.

    Order: (1) exact normalized Act name, (2) Act No./year (confirmed when several
    Acts share a number), (3) the longest corpus Act name contained in the
    citation ('Section 304 of the Penal Code' → 'Penal Code'). Returns None when
    nothing in the corpus matches — the UI then falls back to a keyword search."""
    _ensure_statute_index()
    norm = _norm_alnum(cited)
    if not norm:
        return None
    if norm in _ST_NAME:
        return _ST_NAME[norm]
    for m in re.finditer(r"\bno\.?\s*(\d{1,3})\s+of\s+(\d{4})", (cited or ""), re.I):
        ids = _ST_NUMYEAR.get(f"{int(m.group(1))}/{m.group(2)}")
        if ids:
            return ids[0]
    for nk, sid in (_ST_NAMES_BY_LEN or []):   # contained Act name, longest first
        if len(nk) >= 6 and nk in norm:
            return sid
    return None
