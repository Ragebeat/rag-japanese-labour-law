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
    # Phase 6: the trace showed the reranker takes ~88% of retrieval (about
    # 0.4-0.6 s per pair on the CPU). Phase 4 only tried a LARGER pool; the
    # cheap lever is a smaller one — if recall holds.
    ("+rerank exp pool15", dict(**BEST, rerank=True, rerank_on="expanded",
                                pool=15),
     "15 кандидатов — реранк на четверть дешевле"),
    ("+rerank exp pool10", dict(**BEST, rerank=True, rerank_on="expanded",
                                pool=10),
     "10 кандидатов — реранк вдвое дешевле"),
]

# --pool runs only the working configuration against its smaller pools, and
# writes its own report: sweep.json is phase 4's record.
POOL_SET = ("+rerank exp", "+rerank exp pool15", "+rerank exp pool10")
OUT_POOL = ROOT / "data" / "sweep_pool.json"

# --ruri (phase 7): level 2 of the embedding comparison — Ruri in the full
# pipeline. The expansion dictionary was written against BGE-M3's misses, so
# the question is not only "how high" but "is expansion still worth an LLM
# call": compare "ruri +rerank orig" (no expansion anywhere) with
# "ruri +rerank exp" (the working configuration). BGE-M3's working
# configuration runs alongside as the reference — same session, same Ollama.
RURI = "ruri-v3-310m"
RURI_CONFIGS = [
    ("bge +rerank exp", dict(**BEST, rerank=True, rerank_on="expanded"),
     "рабочая конфигурация на BGE-M3 — отсчёт", "bge-m3"),
    ("ruri vector", dict(mode="vector"),
     "голая Ruri, должна дать 0.86 (как embed_compare)", RURI),
    ("ruri +словарь+LLM", BEST,
     "расширение без реранкера", RURI),
    ("ruri +rerank orig", dict(mode="vector", rerank=True, rerank_on="original"),
     "реранкер без расширения — ни одного вызова LLM", RURI),
    ("ruri +rerank exp", dict(**BEST, rerank=True, rerank_on="expanded"),
     "рабочая конфигурация на Ruri", RURI),
]
OUT_RURI = ROOT / "data" / "sweep_ruri.json"


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
    for j, g in enumerate(golden, 1):
        if use_rerank:
            res = rr.search(g["q"], k=10, core_only=True, **kw)
        else:
            res = h.search(g["q"], k=10, core_only=True, **kw)
        rows.append({"id": g["id"], "q": g["q"],
                     "pos": first_hit(res, g["relevant"])})
        # A reranked config takes ~10 minutes with no output otherwise —
        # indistinguishable from a hang.
        print(f"  {j}/{len(golden)}  {time.time() - t0:.0f}с", end="\r",
              file=sys.stderr, flush=True)
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
    only_pool = "--pool" in sys.argv
    ruri = "--ruri" in sys.argv
    golden = [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]
    if ruri:
        # One Hybrid per embedder; the config's 4th field picks which.
        engines = {e: Hybrid(embedder=e) for e in ("bge-m3", RURI)}
        configs = [(n, kw, note, engines[e]) for n, kw, note, e in RURI_CONFIGS]
        out = OUT_RURI
    else:
        h = Hybrid()
        base = [c for c in CONFIGS if c[0] in POOL_SET] if only_pool else CONFIGS
        configs = [(n, kw, note, h) for n, kw, note in base]
        out = OUT_POOL if only_pool else OUT
    print(f"вопросов: {len(golden)}, срез core, база recall@5 = {BASELINE}\n")
    print(f"{'конфиг':<20} {'r@1':>6} {'r@3':>6} {'r@5':>6} {'r@10':>6} "
          f"{'MRR':>6} {'сек':>6}  {'вs базы':>8}")
    print("-" * 78)

    report = {}
    for name, kw, note, eng in configs:
        m, rows = run(eng, Reranked(eng), golden, kw)
        report[name] = {"params": {k: str(v) for k, v in kw.items()},
                        "embedder": eng.embedder,
                        "note": note, "metrics": m, "rows": rows}
        delta = m["recall@5"] - BASELINE
        mark = "" if abs(delta) >= 0.02 else "  (шум)"
        print(f"{name:<20} {m['recall@1']:>6.3f} {m['recall@3']:>6.3f} "
              f"{m['recall@5']:>6.3f} {m['recall@10']:>6.3f} "
              f"{m['MRR']:>6.3f} {m['sec']:>6.3f}  {delta:>+8.3f}{mark}")
        print(f"{'':20} └ {note}")

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
                 ("+rerank orig", "+rerank pool40"),
                 ("+rerank exp", "+rerank exp pool15"),
                 ("+rerank exp", "+rerank exp pool10"),
                 ("bge +rerank exp", "ruri +rerank exp"),
                 ("ruri +rerank orig", "ruri +rerank exp")):
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

    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"→ {out}")


if __name__ == "__main__":
    main()