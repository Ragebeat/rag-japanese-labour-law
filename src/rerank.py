"""Reranking через cross-encoder поверх векторного поиска.

Зачем: текущие цифры recall@10 = 0.90 при r@1 = 0.48. Нужное почти
всегда в десятке, но наверху лишь в половине случаев.

Bi-encoder (BGE-M3) кодирует запрос и документ НЕЗАВИСИМО → векторы
предпосчитаны, поиск = одно умножение. Быстро, но грубо.
Cross-encoder подаёт пару ВМЕСТЕ на вход → видит взаимодействие слов.
Точнее, но нельзя предпосчитать — модель гоняется на каждую пару.

Отсюда схема: bi-encoder достаёт топ-N быстро, cross-encoder переупорядочивает.

⚠️ Реранкер НЕ поднимет recall@N при том же N — он не ищет заново,
а переставляет уже найденное. Растут r@1, r@3, MRR.

  python src/rerank.py "育休っていつまで？"
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hybrid import Hybrid  # noqa: E402
from query_llm import expand_combo  # noqa: E402

MODEL = "BAAI/bge-reranker-v2-m3"
POOL = 20          # сколько кандидатов отдать реранкеру

_model = None


def _get():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        print(f"загрузка {MODEL}...", file=sys.stderr)
        _model = CrossEncoder(MODEL, max_length=512)
    return _model


class Reranked:
    """Обёртка над Hybrid: достаёт кандидатов и переупорядочивает."""

    def __init__(self, hybrid=None):
        self.h = hybrid or Hybrid()

    def search(self, query, k=10, core_only=True, pool=POOL,
               rerank_on="original", **kw):
        """rerank_on: что подавать cross-encoder'у.

        'original'  — исходный разговорный вопрос
        'expanded'  — расширенный словарём/LLM

        Гипотеза: cross-encoder читает пару целиком и может лучше
        понять оригинальный вопрос, а добавленные термины ему шум.
        Но это гипотеза — меряем оба варианта.
        """
        cands = self.h.search(query, k=pool, core_only=core_only, **kw)
        if not cands:
            return []

        if rerank_on == "expanded":
            rq, _ = expand_combo(query)
        else:
            rq = query

        pairs = [(rq, c["text"]) for c in cands]
        scores = _get().predict(pairs, show_progress_bar=False)

        for c, s, old in zip(cands, scores, range(1, len(cands) + 1)):
            c["rerank_score"] = float(s)
            c["rank_before"] = old
        cands.sort(key=lambda c: -c["rerank_score"])
        return cands[:k]

    def score(self, query, texts, rerank_on="expanded"):
        """Скорит произвольные тексты против запроса, без поиска в БД.

        Нужно для структурной проверки полноты (verify.py): сравнить
        параграфы одной статьи между собой.

        rerank_on по умолчанию 'expanded' — тот же вариант, что в рабочей
        конфигурации (KW в generate.py). Иначе баллы окажутся в другой
        шкале, чем те, по которым чанки отбирались.
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