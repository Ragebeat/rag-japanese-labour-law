"""Оценка трёх режимов поиска на golden dataset.

База (вектор на numpy, точный перебор): recall@5 = 0.68 core, 0.60 all.

  python src/eval_hybrid.py
"""

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hybrid import Hybrid  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden.jsonl"
OUT = ROOT / "data" / "eval_hybrid.json"

KS = (1, 3, 5, 10)
MAXK = max(KS)
BASELINE = {"core": 0.680, "all": 0.600}      # recall@5 наивного векторного


def first_hit(results, relevant):
    want = {(law, art) for law, art in relevant}
    for i, r in enumerate(results, 1):
        if (r["law_id"], r["article_title"]) in want:
            return i
    return None


def run(h, golden, mode, core_only):
    rows, t0 = [], time.time()
    for g in golden:
        res = h.search(g["q"], k=MAXK, core_only=core_only, mode=mode)
        rows.append({
            "id": g["id"], "q": g["q"],
            "pos": first_hit(res, g["relevant"]),
            "top1": (f"{res[0]['law_title'][:16]} {res[0]['article_title']}"
                     if res else "—"),
        })
    n = len(rows)
    m = {f"recall@{k}": sum(1 for r in rows if r["pos"] and r["pos"] <= k) / n
         for k in KS}
    m["MRR"] = sum(1 / r["pos"] for r in rows if r["pos"]) / n
    m["sec_per_query"] = (time.time() - t0) / n
    return m, rows


def main():
    golden = [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]
    h = Hybrid()
    print(f"вопросов: {len(golden)}\n")

    report = {}
    for scope, core_only in (("core", True), ("all", False)):
        print(f"{'='*54}\nСРЕЗ: {scope}\n{'='*54}")
        print(f"{'режим':<10} {'r@1':>6} {'r@3':>6} {'r@5':>6} {'r@10':>6} "
              f"{'MRR':>6} {'сек/запр':>9}")
        for mode in ("vector", "bm25", "hybrid"):
            m, rows = run(h, golden, mode, core_only)
            report[f"{scope}/{mode}"] = {"metrics": m, "rows": rows}
            print(f"{mode:<10} {m['recall@1']:>6.3f} {m['recall@3']:>6.3f} "
                  f"{m['recall@5']:>6.3f} {m['recall@10']:>6.3f} "
                  f"{m['MRR']:>6.3f} {m['sec_per_query']:>9.3f}")

        base = BASELINE[scope]
        best = report[f"{scope}/hybrid"]["metrics"]["recall@5"]
        print(f"\nбаза (naive vector, numpy) recall@5 = {base:.3f}")
        print(f"гибрид                         = {best:.3f}  "
              f"({best - base:+.3f})\n")

    # что починилось и что сломалось относительно чистого вектора
    v = {r["id"]: r["pos"] for r in report["core/vector"]["rows"]}
    b = {r["id"]: r["pos"] for r in report["core/bm25"]["rows"]}
    hy = {r["id"]: r["pos"] for r in report["core/hybrid"]["rows"]}
    qs = {r["id"]: r["q"] for r in report["core/hybrid"]["rows"]}

    def ok(p):
        return p is not None and p <= 5

    fixed = [i for i in qs if not ok(v[i]) and ok(hy[i])]
    broke = [i for i in qs if ok(v[i]) and not ok(hy[i])]
    only_bm = [i for i in qs if not ok(v[i]) and ok(b[i])]
    nobody = [i for i in qs if not ok(v[i]) and not ok(b[i]) and not ok(hy[i])]

    print(f"{'='*54}\nРАЗБОР (core, recall@5)\n{'='*54}")
    print(f"починил гибрид: {len(fixed)}")
    for i in fixed:
        print(f"   #{i:>2} {qs[i][:32]:<34} vec {v[i]} → hyb {hy[i]}")
    print(f"сломал гибрид: {len(broke)}")
    for i in broke:
        print(f"   #{i:>2} {qs[i][:32]:<34} vec {v[i]} → hyb {hy[i]}")
    print(f"вытянул только BM25: {len(only_bm)}")
    for i in only_bm:
        print(f"   #{i:>2} {qs[i][:32]:<34} bm25 {b[i]}")
    print(f"не вытянул никто: {len(nobody)}")
    for i in nobody:
        print(f"   #{i:>2} {qs[i][:40]}")

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()