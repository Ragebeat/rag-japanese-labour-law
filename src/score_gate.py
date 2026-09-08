"""Разделяются ли контексты с источником и без него по score реранкера?

Идея: решение об отказе не выпрашивать у 4B-модели промптом, а вынести
за её пределы — на порог по top-1 score cross-encoder'а. Тогда отказ
детерминирован и настраивается одним числом.

Одна rerank-прогонка на вопрос: реранкер скорит весь пул из 20 сразу,
из него получаем и обычный топ-1, и топ-1 после удаления золотой статьи.

⚠️ ЧЕГО ЭТА ЦИФРА НЕ ДОКАЗЫВАЕТ.
При r@1 = 0.60 золотая статья часто и ЕСТЬ топ-1. Убрали её — балл упал
по построению, и AUC на абляции завышена. Честная проверка — шесть
естественных промахов, где статьи не было изначально. Их мало, но они
не подстроены.

⚠️ Порог не подбирается ради красивой цифры. Он задаётся продуктовым
решением «сколько over-refusal мы готовы терпеть» (5-й перцентиль
группы с источником), и уже потом смотрим, что получилось.

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
    """P(случайный из pos выше случайного из neg). 0.5 — неразличимы."""
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

    A = [x["top1_full"] for x in rows if x["hit"]]          # источник есть
    B = [x["top1_ablated"] for x in rows if x["hit"]]       # источник удалён
    C = [x["top1_full"] for x in rows if not x["hit"]]      # промах естественный

    def stat(name, xs):
        if not xs:
            return
        print(f"  {name:<28} n={len(xs):>2}  медиана {statistics.median(xs):+.3f}  "
              f"мин {min(xs):+.3f}  макс {max(xs):+.3f}")

    print("\n── top-1 score ──")
    stat("A: источник в контексте", A)
    stat("B: источник удалён", B)
    stat("C: промах retrieval", C)

    print(f"\nAUC(A против B): {auc(A, B):.3f}   ← завышена: удаляли часто сам топ-1")
    print(f"AUC(A против C): {auc(A, C):.3f}   ← честная, но n={len(C)}")

    # Порог задан бюджетом на over-refusal, а не максимизацией разделения.
    thr = q(A, 5)
    print(f"\nпорог = 5-й перцентиль A = {thr:+.3f}  (то есть терпим 5% over-refusal)")
    for name, xs in (("A", A), ("B", B), ("C", C)):
        if xs:
            below = sum(1 for x in xs if x < thr)
            print(f"  {name}: ниже порога {below}/{len(xs)} = {below / len(xs):.2f}")

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
