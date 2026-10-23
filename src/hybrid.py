"""Hybrid search: vector + BM25, fused with RRF.

Why RRF and not a weighted sum of scores: cosine gives ~0.58, ts_rank ~0.09 —
the scales are incomparable and drift from query to query. RRF discards the
scores and works on positions alone:

    score(d) = Σ 1 / (k + rank_i(d))

k = 60, from the original Cormack et al. 2009 paper. It damps the difference
between top positions so that one method cannot drown out the other.

  python src/hybrid.py "有給ってすぐもらえる？"
  python src/hybrid.py "パワハラ対策？" --core
"""

import os
import pathlib
import sys

import httpx
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tokenize_ja import tokenize  # noqa: E402
from query_expand import expand  # noqa: E402
from query_llm import expand_llm, expand_combo  # noqa: E402

from env import DSN  # noqa: E402  (.env is loaded there)

MODEL = "bge-m3"
OLLAMA = "http://localhost:11434/api/embed"

# The embedding model is a choice, not a constant. Bare vector
# recall@5 on golden: bge-m3 0.68, ruri-v3-310m 0.86 (embed_compare.py).
# Each embedder has its own column; the default comes from EMBEDDER so the
# whole pipeline switches with one knob, and it goes into the run fingerprint.
EMBEDDERS = {
    "bge-m3": {"column": "embedding", "dim": 1024},
    "ruri-v3-310m": {"column": "embedding_ruri", "dim": 768,
                     "hf": "cl-nagoya/ruri-v3-310m",
                     "revision": "18b60fb8c2b9df296fb4212bb7d23ef94e579cd3",
                     # documents were encoded with "検索文書: " (embed_ruri.py)
                     "query_prefix": "検索クエリ: "},
}
# Default (sweep.py --ruri): full pipeline r@5 0.94 without
# query expansion vs 0.86 for BGE-M3 with it. EMBEDDER=bge-m3 switches back.
EMBEDDER = os.environ.get("EMBEDDER", "ruri-v3-310m")
_st = {}             # in-process models, loaded once per process


def _st_model(name):
    if name not in _st:
        from sentence_transformers import SentenceTransformer
        cfg = EMBEDDERS[name]
        print(f"loading {cfg['hf']}...", file=sys.stderr)
        _st[name] = SentenceTransformer(cfg["hf"], device="cpu",
                                        revision=cfg["revision"])
    return _st[name]

RRF_K = 60
POOL = 50          # how many we take from each method before fusion

# HNSW is an APPROXIMATE index. Measured on this corpus:
#   ef_search=40  (default) -> 第39条 does not make the top 10 at all
#   ef_search=100            -> 第39条 in 6th place (= exact numpy scan)
#   ef_search=400            -> the same, no further gain
# The default drops results SILENTLY — no error, no warning.
#
# Ruri v3 (embed_ruri.py check, golden queries, core, top-20 overlap
# with the exact scan):  ef 40 -> 0.914 (worst 0.25) · 100 -> 0.973 (worst
# 0.75) · 200 -> 0.998 · 400 -> 0.999. The value tuned on BGE-M3 loses 3% of
# Ruri's pool, up to a quarter on one question: a different model is a
# different geometry. 200 for both columns — for BGE-M3 a higher ef only
# costs latency, it cannot reduce recall.
EF_SEARCH = 200

FIELDS = ("id, chunk_id, text, law_id, law_title, law_type, core, "
          "article_title, article_caption, paragraph_num")


class Hybrid:
    def __init__(self, embedder=None):
        self.embedder = embedder or EMBEDDER
        if self.embedder not in EMBEDDERS:
            raise ValueError(f"unknown embedder: {self.embedder}")
        self.column = EMBEDDERS[self.embedder]["column"]
        # autocommit: this class only reads. Without it every SELECT opened a
        # transaction that stayed open for the life of the process (uvicorn,
        # the agent, the MCP server) — holding a lock that made ALTER TABLE
        # in embed_ruri.py wait forever, and leaving "unexpected EOF on
        # client connection with an open transaction" in the Postgres log.
        self.conn = psycopg.connect(DSN, autocommit=True)
        register_vector(self.conn)
        self.conn.execute(f"SET hnsw.ef_search = {EF_SEARCH}")
        self.http = httpx.Client()

    def embed(self, text):
        if self.embedder != "bge-m3":
            cfg = EMBEDDERS[self.embedder]
            return _st_model(self.embedder).encode(
                cfg["query_prefix"] + text, normalize_embeddings=True,
                show_progress_bar=False).astype(np.float32)
        r = self.http.post(OLLAMA, json={"model": MODEL, "input": text},
                           timeout=60)
        r.raise_for_status()
        return np.array(r.json()["embeddings"][0], dtype=np.float32)

    def vector_search(self, query, limit=POOL, core_only=False):
        """Vector search. <#> is the NEGATIVE inner product,
        so ASC returns the most similar first."""
        q = self.embed(query)
        where = "WHERE core" if core_only else ""
        col = self.column          # from EMBEDDERS, never from input
        sql = f"""
            SELECT {FIELDS}, ({col} <#> %s) * -1 AS score
            FROM chunks {where}
            ORDER BY {col} <#> %s
            LIMIT %s
        """
        return self._rows(sql, (q, q, limit))

    def bm25_search(self, query, limit=POOL, core_only=False, min_rank=0.0):
        """Lexical search via tsvector. The query is tokenised with
        THE SAME Sudachi as the corpus, or the tokens will not match."""
        toks = tokenize(query)
        if not toks:
            return []
        tsquery = " | ".join(toks)
        where = "AND core" if core_only else ""
        having = "AND ts_rank(tsv, to_tsquery('simple', %s)) >= %s" if min_rank else ""
        sql = f"""
            SELECT {FIELDS},
                   ts_rank(tsv, to_tsquery('simple', %s)) AS score
            FROM chunks
            WHERE tsv @@ to_tsquery('simple', %s) {where} {having}
            ORDER BY score DESC
            LIMIT %s
        """
        params = ([tsquery, tsquery] + ([tsquery, min_rank] if min_rank else [])
                  + [limit])
        return self._rows(sql, tuple(params))

    def _rows(self, sql, params):
        cur = self.conn.execute(sql, params)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def _fuse(self, lists, weights):
        """RRF over several rankings."""
        fused = {}
        for lst, w, tag in lists:
            for rank, row in enumerate(lst, 1):
                e = fused.setdefault(row["id"],
                                     {"row": row, "rrf": 0.0, "ranks": {}})
                e["rrf"] += w / (RRF_K + rank)
                e["ranks"][tag] = rank
        return fused

    def search(self, query, k=10, core_only=False, mode="hybrid",
               w_vec=1.0, w_bm25=1.0, pool_vec=POOL, pool_bm25=POOL,
               min_rank=0.0, expand_query=False, narrow=False,
               llm=False, combo=False):

        def _do_expand(q):
            if combo:
                return expand_combo(q, narrow=narrow)
            if llm:
                return expand_llm(q)
            return expand(q, narrow)

        # multi-query: search by BOTH query variants and fuse.
        # Solves a problem seen in the measurements: expansion fixes 7 questions
        # but breaks 3 — the ones where the original ranked first. Fusion keeps
        # such a result: it still earns 1/(60+1) from its own list.
        if mode == "multiquery":
            exp, extra = _do_expand(query)
            orig = self.vector_search(query, pool_vec, core_only)
            if not extra:
                return orig[:k]
            alt = self.vector_search(exp, pool_vec, core_only)
            fused = self._fuse([(orig, 1.0, "orig"), (alt, 1.0, "exp")], None)
            top = sorted(fused.values(), key=lambda e: -e["rrf"])[:k]
            out = []
            for e in top:
                row = dict(e["row"])
                row["score"] = e["rrf"]
                row["ranks"] = e["ranks"]
                out.append(row)
            return out

        if expand_query:
            query, _ = _do_expand(query)

        if mode == "vector":
            return self.vector_search(query, k, core_only)
        if mode == "bm25":
            return self.bm25_search(query, k, core_only, min_rank)

        vec = self.vector_search(query, pool_vec, core_only)
        lex = self.bm25_search(query, pool_bm25, core_only, min_rank)

        fused = {}
        for rank, row in enumerate(vec, 1):
            e = fused.setdefault(row["id"], {"row": row, "rrf": 0.0,
                                             "v": None, "b": None})
            e["rrf"] += w_vec / (RRF_K + rank)
            e["v"] = rank
        for rank, row in enumerate(lex, 1):
            e = fused.setdefault(row["id"], {"row": row, "rrf": 0.0,
                                             "v": None, "b": None})
            e["rrf"] += w_bm25 / (RRF_K + rank)
            e["b"] = rank

        top = sorted(fused.values(), key=lambda e: -e["rrf"])[:k]
        out = []
        for e in top:
            row = dict(e["row"])
            row["score"] = e["rrf"]
            row["rank_vec"] = e["v"]
            row["rank_bm25"] = e["b"]
            out.append(row)
        return out


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    core_only = "--core" in args
    mode = "hybrid"
    for m in ("vector", "bm25"):
        if f"--{m}" in args:
            mode = m
    query = " ".join(a for a in args if not a.startswith("--"))

    h = Hybrid()
    print(f"query: {query}")
    print(f"tokens: {tokenize(query)}")
    print(f"mode: {mode}" + (" / core" if core_only else "") + "\n")

    for i, r in enumerate(h.search(query, 10, core_only, mode), 1):
        pos = ""
        if mode == "hybrid":
            v = r["rank_vec"] or "—"
            b = r["rank_bm25"] or "—"
            pos = f"  [vec {v}, bm25 {b}]"
        head = f"{r['law_title'][:20]} {r['article_title']}{r['article_caption'] or ''}"
        print(f"{i:>2}. {r['score']:.4f}  {head}{pos}")
        print(f"      {r['text'].split(chr(10), 1)[-1][:80]}")


if __name__ == "__main__":
    main()