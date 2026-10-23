"""Do contexts with and without the source separate by reranker score?

The idea: rather than begging a 4B model for a refusal through the prompt,
move the decision outside it — onto a threshold on the cross-encoder's top-1
score. Then refusal is deterministic and tuned by a single number.

One reranking pass per question: the reranker scores the whole pool of 20 at
once, and from it we get both the ordinary top-1 and the top-1 after the gold
article is removed.

⚠️ WHAT THIS NUMBER DOES NOT PROVE.
At r@1 = 0.60 the gold article often IS the top-1. Remove it and the score
drops by construction, so the AUC on the ablation is inflated. The honest
check is the six natural misses, where the article was never there to begin
with. There are few of them, but they are not staged.

⚠️ The threshold is not tuned for a pretty number. It is set by a product
decision — how much over-refusal we are willing to accept (the 5th percentile
of the group with a source) — and only then do we look at what came out.

  uv run python src/score_gate.py
"""

import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from generate import KW, POOL_MULT, TOP_K, dedupe_by_article, load_golden  # noqa: E402
from rerank import Reranked  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "score_gate.json"


def auc(pos, neg):
    """P(a random member of pos scores above a random member of neg). 0.5 — indistinguishable."""
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def q(xs, p):
    return statistics.quantiles(sorted(xs), n=100)[p - 1] if len(xs) > 2 else min(xs)


def main():
    r = Reranked()
    pool = TOP_K * POOL_MULT
    rows = []

    for i, g in enumerate(load_golden(), 1):
        gold = {tuple(x) for x in g["relevant"]}
        cands = r.search(g["q"], k=pool, core_only=True, pool=pool, **KW)

        full = dedupe_by_article(cands, TOP_K)
        abl = dedupe_by_article(
            [c for c in cands if (c["law_id"], c["article_title"]) not in gold], TOP_K)

        hit = bool(gold & {(c["law_id"], c["article_title"]) for c in full})
        rows.append({
            "id": g["id"], "q": g["q"], "hit": hit,
            "top1_full": full[0]["rerank_score"] if full else None,
            "top1_ablated": abl[0]["rerank_score"] if abl else None,
        })
        print(f"{i:>2}/50 {'hit ' if hit else 'MISS'} "
              f"{rows[-1]['top1_full']:+.3f} → {rows[-1]['top1_ablated']:+.3f}  "
              f"{g['q'][:26]}")

    A = [x["top1_full"] for x in rows if x["hit"]]          # the source is present
    B = [x["top1_ablated"] for x in rows if x["hit"]]       # the source was removed
    C = [x["top1_full"] for x in rows if not x["hit"]]      # a natural retrieval miss

    def stat(name, xs):
        if not xs:
            return
        print(f"  {name:<28} n={len(xs):>2}  median {statistics.median(xs):+.3f}  "
              f"min {min(xs):+.3f}  max {max(xs):+.3f}")

    print("\n── top-1 score ──")
    stat("A: source in the context", A)
    stat("B: source removed", B)
    stat("C: retrieval miss", C)

    print(f"\nAUC(A vs B): {auc(A, B):.3f}   ← inflated: often the top-1 itself was removed")
    print(f"AUC(A vs C): {auc(A, C):.3f}   ← honest, but n={len(C)}")

    # The threshold is set by an over-refusal budget, not by maximising separation.
    thr = q(A, 5)
    print(f"\nthreshold = 5th percentile of A = {thr:+.3f}  (i.e. 5% over-refusal tolerated)")
    for name, xs in (("A", A), ("B", B), ("C", C)):
        if xs:
            below = sum(1 for x in xs if x < thr)
            print(f"  {name}: below the threshold {below}/{len(xs)} = {below / len(xs):.2f}")

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
