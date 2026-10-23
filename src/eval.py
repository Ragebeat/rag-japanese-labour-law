"""Retrieval evaluation on the golden dataset.

recall@k — the share of questions where AT LEAST ONE relevant article made
the top k. The user needs an answer, not an exhaustive digest, so one is
enough. MRR — the mean of 1/position of the first hit.

  python src/eval.py
"""

import json
import pathlib

from search import Index

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden.jsonl"
OUT = ROOT / "data" / "eval_report.json"

KS = (1, 3, 5, 10)
MAXK = max(KS)


def first_hit(results, relevant):
    """Position (1-based) of the first hit, or None."""
    want = {(law, art) for law, art in relevant}
    for i, (_score, c) in enumerate(results, 1):
        if (c["law_id"], c["article_title"]) in want:
            return i
    return None


def run(idx, golden, core_only):
    rows = []
    for g in golden:
        res = idx.search(g["q"], k=MAXK, core_only=core_only)
        pos = first_hit(res, g["relevant"])
        rows.append({
            "id": g["id"],
            "q": g["q"],
            "pos": pos,
            "top1": f"{res[0][1]['law_title'][:18]} {res[0][1]['article_title']}",
        })

    n = len(rows)
    metrics = {f"recall@{k}": sum(1 for r in rows if r["pos"] and r["pos"] <= k) / n
               for k in KS}
    metrics["MRR"] = sum(1 / r["pos"] for r in rows if r["pos"]) / n
    return metrics, rows


def main():
    golden = [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]
    idx = Index()
    print(f"questions: {len(golden)}   chunks: {len(idx.meta)} "
          f"(core {int(idx.core_mask.sum())})\n")

    report = {}
    for mode, core_only in (("all", False), ("core", True)):
        metrics, rows = run(idx, golden, core_only)
        report[mode] = {"metrics": metrics, "rows": rows}

        print(f"=== mode {mode} ===")
        for k, v in metrics.items():
            print(f"  {k:<10} {v:.3f}")

        miss = [r for r in rows if r["pos"] is None]
        far = [r for r in rows if r["pos"] and r["pos"] > 5]
        print(f"  not in the top 10: {len(miss)}")
        for r in miss:
            print(f"     #{r['id']:>2} {r['q'][:34]:<34} → {r['top1']}")
        if far:
            print(f"  found below rank 5: {len(far)}")
            for r in far:
                print(f"     #{r['id']:>2} rank {r['pos']:>2}  {r['q'][:30]}")
        print()

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()