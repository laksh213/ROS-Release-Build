"""Phase 4 — Retrieval + reranking.

Two stages: (1) vector recall from Chroma, (2) BGE-Reranker-v2
(`bge-reranker-v2-m3`, multilingual) re-scores the candidates. The reranker is
optional — if FlagEmbedding isn't installed yet, recall falls back to pure vector
order so the pipeline still runs.

CLI:
  python -m src.retrieve "compensation in lieu of reinstatement" -k 5
"""

from __future__ import annotations

import argparse

from .config import settings
from .store import similarity_search

_RERANKER = None


def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        from FlagEmbedding import FlagReranker  # heavy; imported lazily

        _RERANKER = FlagReranker(settings.reranker_model, use_fp16=True)
    return _RERANKER


def _rerank(query: str, hits: list[dict], k: int) -> list[dict]:
    try:
        reranker = _get_reranker()
        scores = reranker.compute_score([[query, h["text"]] for h in hits])
        for h, s in zip(hits, scores):
            h["rerank"] = float(s)
        hits = sorted(hits, key=lambda h: h["rerank"], reverse=True)
    except Exception:
        pass  # reranker unavailable -> keep vector order
    return hits[:k]


def retrieve(query: str, k: int = 8, source: str | None = None, expanded_query: str | None = None) -> list[dict]:
    """Hybrid FTS + Chroma recall -> (optional) bge-reranker-v2 -> top-k."""
    try:
        import sqlite3
        from .store import _fts_queries
        
        # 1. Fetch exact keyword matches via SQLite FTS5 first
        fts_hits = []
        con = sqlite3.connect(settings.sqlite_path)
        for _, match in _fts_queries(query):
            try:
                rows = con.execute(
                    "SELECT case_no, page, text, bm25(chunks_fts) AS score "
                    "FROM chunks_fts WHERE chunks_fts MATCH ? "
                    "LIMIT ?",
                    (match, k * 2)
                ).fetchall()
                for cn, pg, text, score in rows:
                    fts_hits.append({
                        "text": text,
                        "meta": {
                            "case_no": cn,
                            "page": pg,
                            "anchor": f"{cn} p.{pg}"
                        },
                        "distance": score
                    })
                if fts_hits:
                    break
            except Exception:
                pass
        con.close()
        
        # 2. Fetch semantic matches via Chroma vector similarity
        sem_query = expanded_query if expanded_query else query
        sem_hits = similarity_search(sem_query, k=max(k * 3, 20), source=source)
        
        # 3. Merge results, keeping FTS keyword matches first
        seen = set()
        merged = []
        
        for h in fts_hits:
            key = (h["meta"]["case_no"], h["meta"]["page"])
            if key not in seen:
                seen.add(key)
                merged.append(h)
                
        for h in sem_hits:
            cn = h["meta"].get("case_no")
            pg = h["meta"].get("page")
            if cn and pg:
                key = (cn, pg)
                if key not in seen:
                    seen.add(key)
                    merged.append(h)
                    
        # 4. Rerank if enabled, or return top-k
        if settings.use_reranker:
            return _rerank(query, merged, k)
        return merged[:k]
    except Exception as e:
        print(f"[retrieve] hybrid search failed: {e}")
        try:
            return similarity_search(query, k=k, source=source)
        except:
            return []


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Semantic search over the index.")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--source", choices=["judgment", "personal_repo"], default=None,
                    help="restrict to judgements or your notes")
    args = ap.parse_args(argv)

    hits = retrieve(args.query, args.k, source=args.source)
    if not hits:
        print("No results — is the index built? Run: python -m src.index")
        return
    for h in hits:
        m = h["meta"]
        score = h.get("rerank", -h["distance"])
        print(f"\n {m.get('anchor', '?')}   (score {score:.3f})")
        print("  " + h["text"][:260].strip().replace(chr(10), " ") + "…")


if __name__ == "__main__":
    main()
