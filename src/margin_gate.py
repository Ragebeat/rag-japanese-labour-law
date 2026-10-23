"""Does the top-1/top-2 margin separate "source found" from "not found"
better than the absolute top-1 score?

The hypothesis comes from the agent runs. On two model-reformulated queries the
absolute gate ranked backwards: 労基法 第24条, the right article, came first
at 0.091 (below the 0.100 threshold -> refusal), while junk came first at
0.205 (above -> accepted). A cross-encoder score is not comparable ACROSS
queries; within one ranking the order holds (0.091 against 0.009 for #2).
A relative signal might not depend on how the query was phrased.

Two candidates, declared before measuring so the prettier one cannot be
picked afterwards:
  diff  = top1 - top2
  ratio = top1 / top2     (the agent runs suggested "ten times": that is the ratio)
The threshold for each is set by the same product rule as score_gate.py —
the 5th percentile of group A — and only then do we look at what came out.

Groups, as in score_gate.py (same reranking pass, same caveats):
  A — the gold article is in the top 5
  B — the gold article removed from the pool (ablation; inflated, see there)
  C — a natural retrieval miss (honest, but n is small)

⚠️ golden holds homogeneous colloquial questions, while the agent-run problem
appeared on reformulated ones. So a few PROBES are added: queries from
the agent runs, each labelled automatically by whether 第24条 came first.

  uv run python src/margin_gate.py
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from generate import KW, POOL_MULT, TOP_K, dedupe_by_article, load_golden  # noqa: E402
from rerank import Reranked  # noqa: E402
from score_gate import auc, q  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "margin_gate.json"
FLOOR = 1e-4          # keeps the ratio finite when top2 is ~0

# Queries from the agent runs and the rerank.py check. 第24条 first ->
# the ranking found the right article; anything else first -> junk.
PROBES = [
    "給料から勝手に天引きされた 賃金の全額払い",
    "給与の全額支払い義務",
    "賃金の全額払いの原則と、賃金から控除できる場合",
    "給与からの税金天引きの原則",
    "会社が給与から住民税を天引きすることの合法性",
]
TARGET = ("322AC0000000049", "第二十四条")      # 労働基準法 第二十四条

SIGNALS = ("top1", "diff", "ratio")


def signals(ranked):
    s1 = ranked[0]["rerank_score"] if ranked else 0.0
    s2 = ranked[1]["rerank_score"] if len(ranked) > 1 else 0.0
    return {"top1": s1, "top2": s2, "diff": s1 - s2,
            "ratio": s1 / max(s2, FLOOR)}


def main():
    r = Reranked()
    pool = TOP_K * POOL_MULT

    def ranked(query, drop=frozenset(), cands=None):
        # One reranking pass is enough for both views: the reranker scores
        # the whole pool regardless (same as score_gate.py).
        if cands is None:
            cands = r.search(query, k=pool, core_only=True, pool=pool, **KW)
        cands = [c for c in cands if (c["law_id"], c["article_title"]) not in drop]
        # Deduped by article: the second chunk of the SAME article is not a
        # competitor, and would make every margin look tiny.
        return dedupe_by_article(cands, TOP_K)

    rows = []
    for i, g in enumerate(load_golden(), 1):
        gold = {tuple(x) for x in g["relevant"]}
        cands = r.search(g["q"], k=pool, core_only=True, pool=pool, **KW)
        full = ranked(g["q"], cands=cands)
        abl = ranked(g["q"], drop=gold, cands=cands)
        hit = bool(gold & {(c["law_id"], c["article_title"]) for c in full})
        rows.append({"id": g["id"], "q": g["q"], "hit": hit,
                     "full": signals(full), "ablated": signals(abl)})
        f = rows[-1]["full"]
        print(f"{i:>2}/50 {'hit ' if hit else 'MISS'}  top1 {f['top1']:.3f}  "
              f"top2 {f['top2']:.3f}  diff {f['diff']:+.3f}  ratio {f['ratio']:>7.1f}")

    groups = {
        "A": [x["full"] for x in rows if x["hit"]],
        "B": [x["ablated"] for x in rows if x["hit"]],
        "C": [x["full"] for x in rows if not x["hit"]],
    }

    print(f"\n{'signal':<7} {'AUC A↔B':>8} {'AUC A↔C':>8}   {'thresh':>8}  "
          f"{'A below':>7} {'B below':>7} {'C below':>7}")
    summary = {}
    for s in SIGNALS:
        A = [g[s] for g in groups["A"]]
        B = [g[s] for g in groups["B"]]
        C = [g[s] for g in groups["C"]]
        thr = q(A, 5)
        below = {k: sum(1 for x in v if x < thr) / len(v) if v else float("nan")
                 for k, v in (("A", A), ("B", B), ("C", C))}
        summary[s] = {"auc_ab": auc(A, B), "auc_ac": auc(A, C),
                      "threshold": thr, "below": below}
        print(f"{s:<7} {auc(A, B):>8.3f} {auc(A, C):>8.3f}   {thr:>8.3f}  "
              f"{below['A']:>7.2f} {below['B']:>7.2f} {below['C']:>7.2f}")
    print(f"  n: A={len(groups['A'])}  B={len(groups['B'])}  C={len(groups['C'])}"
          f"   ← AUC A↔B is inflated (often the top-1 itself was removed); A↔C is honest, but n is small")
    print("  \"B below\", \"C below\" — the share the gate correctly cuts off; "
          "\"A below\" ≈ 0.05 by construction (the cost: over-refusal)")

    print("\n── probes from the agent runs (reformulated queries) ──")
    probes = []
    for p in PROBES:
        rk = ranked(p)
        right = bool(rk) and (rk[0]["law_id"], rk[0]["article_title"]) == TARGET
        sg = signals(rk)
        passes = {s: sg[s] >= summary[s]["threshold"] for s in SIGNALS}
        probes.append({"q": p, "right": right, **sg, "passes": passes})
        head = f"{rk[0]['law_title'][:10]} {rk[0]['article_title']}" if rk else "—"
        verdict = "  ".join(f"{s}:{'pass' if passes[s] else 'refuse'}" for s in SIGNALS)
        print(f"  {'✓ 第24条' if right else '✗ junk  '}  {head:<22} "
              f"top1 {sg['top1']:.3f} diff {sg['diff']:+.3f} ratio {sg['ratio']:>6.1f}  "
              f"| {verdict}")
    print("  a good signal: ✓ → pass, ✗ → refuse")

    OUT.write_text(json.dumps({"rows": rows, "summary": summary, "probes": probes},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
