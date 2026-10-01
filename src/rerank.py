"""Reranking with a cross-encoder on top of vector search.

Why: the current numbers are recall@10 = 0.90 at r@1 = 0.48. What is needed
is almost always in the top ten, but at the top only half the time.

A bi-encoder (BGE-M3) encodes query and document INDEPENDENTLY, so vectors
can be precomputed and search is a single multiplication. Fast but coarse.
A cross-encoder feeds the pair in TOGETHER and sees the interaction between
words. More accurate, but nothing can be precomputed — the model runs on
every pair.

Hence the scheme: the bi-encoder pulls the top N quickly, the cross-encoder
reorders it.

⚠️ The reranker will NOT raise recall@N for the same N — it does not search
again, it rearranges what was already found. r@1, r@3 and MRR go up.

  python src/rerank.py "育休っていつまで？"
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hybrid import Hybrid  # noqa: E402
from query_llm import expand_combo  # noqa: E402
import tracing  # noqa: E402

MODEL = "BAAI/bge-reranker-v2-m3"
# Pinned commit on Hugging Face (phase 6, OWASP supply chain). Without it the
# model is "whatever main points to" on the next fresh download: new weights
# would silently move every rerank score — and the score gate threshold with
# them. Part of the run fingerprint. Change on purpose, then re-run
# score_gate.py and --eval.
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
POOL = 20          # how many candidates to hand to the reranker

_model = None


def _get():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        print(f"загрузка {MODEL}...", file=sys.stderr)
        _model = CrossEncoder(MODEL, max_length=512, revision=REVISION)
    return _model


class Reranked:
    """A wrapper over Hybrid: fetches candidates and reorders them."""

    def __init__(self, hybrid=None):
        self.h = hybrid or Hybrid()

    def search(self, query, k=10, core_only=True, pool=POOL,
               rerank_on="original", **kw):
        """rerank_on: what to feed the cross-encoder.

        'original'  — the original colloquial question
        'expanded'  — expanded by the dictionary and the LLM

        Hypothesis: the cross-encoder reads the pair as a whole and may do
        better on the original question, with the added terms acting as noise.
        But that is a hypothesis — we measure both.
        """
        # Two spans split retrieval in two: candidates (query expansion,
        # embedding via Ollama, Postgres) and the cross-encoder on the CPU.
        with tracing.span("candidates", "RETRIEVER", input=query,
                          attrs={"retrieval.pool": pool, "retrieval.core_only": core_only}) as sp:
            cands = self.h.search(query, k=pool, core_only=core_only, **kw)
            tracing.set_attrs(sp, {"retrieval.found": len(cands)})
        if not cands:
            return []

        if rerank_on == "expanded":
            rq, _ = expand_combo(query)
        else:
            rq = query

        pairs = [(rq, c["text"]) for c in cands]
        with tracing.span("rerank", "RERANKER", input=rq, attrs={
                "reranker.model_name": MODEL, "reranker.query": rq,
                "reranker.pairs": len(pairs),
                "reranker.input_chars": sum(len(t) for _, t in pairs)}):
            scores = _get().predict(pairs, show_progress_bar=False)

        for c, s, old in zip(cands, scores, range(1, len(cands) + 1)):
            c["rerank_score"] = float(s)
            c["rank_before"] = old
        cands.sort(key=lambda c: -c["rerank_score"])
        return cands[:k]

    def score(self, query, texts, rerank_on="expanded"):
        """Scores arbitrary texts against a query, without searching the database.

        Needed for the structural completeness check (verify.py): comparing
        paragraphs of one article against each other.

        rerank_on defaults to 'expanded' — the same variant as the working
        configuration (KW in generate.py). Otherwise the scores would be on a
        different scale from the ones the chunks were selected by.
        """
        if not texts:
            return []
        rq = expand_combo(query)[0] if rerank_on == "expanded" else query
        scores = _get().predict([(rq, t) for t in texts], show_progress_bar=False)
        return [float(s) for s in scores]


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    query = " ".join(a for a in args if not a.startswith("--"))

    r = Reranked()
    print(f"запрос: {query}\n")
    res = r.search(query, k=10, mode="vector", expand_query=True,
                   narrow=True, combo=True)
    for i, c in enumerate(res, 1):
        move = c["rank_before"] - i
        arrow = f"↑{move}" if move > 0 else (f"↓{-move}" if move < 0 else " —")
        head = f"{c['law_title'][:20]} {c['article_title']}{c['article_caption'] or ''}"
        print(f"{i:>2}. {c['rerank_score']:+.3f}  был {c['rank_before']:>2} {arrow}  {head}")


if __name__ == "__main__":
    main()