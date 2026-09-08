"""Sweeping hybrid search configurations over the golden dataset.

⚠️ The golden set is only 50 questions. Turning knobs until the number looks
good is fitting to those 50, not improving search. So this file holds FIVE
meaningful variants rather than a grid. A difference below 0.02 is noise,
not an improvement.

  python src/sweep.py
"""

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hybrid import Hybrid  # noqa: E402
from rerank import Reranked  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden.jsonl"
OUT = ROOT / "data" / "sweep.json"

BASELINE = 0.820     # recall@5, dictionary + LLM fallback

BEST = dict(mode="vector", expand_query=True, narrow=True, combo=True)

CONFIGS = [
    ("vector", dict(mode="vector"),
     "исходная база 0.68"),
    ("словарь+LLM", BEST,
     "текущий лидер 0.82"),
    ("+rerank orig", dict(**BEST, rerank=True, rerank_on="original"),
     "cross-encoder по ИСХОДНОМУ вопросу"),
    ("+rerank exp", dict(**BEST, rerank=True, rerank_on="expanded"),
     "cross-encoder по расширенному"),
    ("+rerank pool40", dict(**BEST, rerank=True, rerank_on="original",
                            pool=40),
     "40 кандидатов вместо 20"),
]


def first_hit(results, relevant):
    want = {(law, art) for law, art in relevant}
    for i, r in enumerate(results, 1):
        if (r["law_id"], r["article_title"]) in want:
            return i
    return None


def run(h, rr, golden, kw):
    kw = dict(kw)
    use_rerank = kw.pop("rerank", False)
    rows, t0 = [], time.time()
    for g in golden:
        if use_rerank:
            res = rr.search(g["q"], k=10, core_only=True, **kw)
        else:
            res = h.search(g["q"], k=10, core_only=True, **kw)
        rows.append({"id": g["id"], "q": g["q"],
                     "pos": first_hit(res, g["relevant"])})
    n = len(rows)
    return {
        "recall@1": sum(1 for r in rows if r["pos"] == 1) / n,
        "recall@3": sum(1 for r in rows if r["pos"] and r["pos"] <= 3) / n,
        "recall@5": sum(1 for r in rows if r["pos"] and r["pos"] <= 5) / n,
        "recall@10": sum(1 for r in rows if r["pos"]) / n,
        "MRR": sum(1 / r["pos"] for r in rows if r["pos"]) / n,
        "sec": (time.time() - t0) / n,
    }, rows


def main():
    golden = [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]
    h = Hybrid()
    rr = Reranked(h)
    print(f"вопросов: {len(golden)}, срез core, база recall@5 = {BASELINE}\n")
    print(f"{'конфиг':<16} {'r@1':>6} {'r@3':>6} {'r@5':>6} {'r@10':>6} "
          f"{'MRR':>6} {'сек':>6}  {'вs базы':>8}")
    print("-" * 74)

    report = {}
    for name, kw, note in CONFIGS:
        m, rows = run(h, rr, golden, kw)
        report[name] = {"params": {k: str(v) for k, v in kw.items()},
                        "note": note, "metrics": m, "rows": rows}
        delta = m["recall@5"] - BASELINE
        mark = "" if abs(delta) >= 0.02 else "  (шум)"
        print(f"{name:<16} {m['recall@1']:>6.3f} {m['recall@3']:>6.3f} "
              f"{m['recall@5']:>6.3f} {m['recall@10']:>6.3f} "
              f"{m['MRR']:>6.3f} {m['sec']:>6.3f}  {delta:>+8.3f}{mark}")
        print(f"{'':16} └ {note}")

    best = max(report.items(), key=lambda kv: kv[1]["metrics"]["recall@5"])
    print(f"\nлучший: {best[0]} — recall@5 = "
          f"{best[1]['metrics']['recall@5']:.3f}")
    if best[1]["metrics"]["recall@5"] - BASELINE < 0.02:
        print("⚠️ выигрыш меньше 0.02 — в пределах шума на 50 вопросах")

    # what exactly the query expansion fixed
    def ok(p):
        return p is not None and p <= 5

    for pair in (("словарь+LLM", "+rerank orig"),
                 ("+rerank orig", "+rerank exp"),
                 ("+rerank orig", "+rerank pool40")):
        if pair[0] not in report or pair[1] not in report:
            continue
        a = {r["id"]: r["pos"] for r in report[pair[0]]["rows"]}
        b = {r["id"]: r["pos"] for r in report[pair[1]]["rows"]}
        qs = {r["id"]: r["q"] for r in report[pair[0]]["rows"]}
        fixed = [i for i in qs if not ok(a[i]) and ok(b[i])]
        broke = [i for i in qs if ok(a[i]) and not ok(b[i])]
        print(f"\n{pair[0]} → {pair[1]}:  починил {len(fixed)}, сломал {len(broke)}")
        for i in fixed:
            print(f"   + #{i:>2} {qs[i][:34]:<36} {a[i]} → {b[i]}")
        for i in broke:
            print(f"   − #{i:>2} {qs[i][:34]:<36} {a[i]} → {b[i]}")

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()