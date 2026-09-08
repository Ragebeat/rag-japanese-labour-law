"""Гибридный поиск: вектор + BM25, объединение через RRF.

Почему RRF, а не взвешенная сумма оценок: косинус даёт ~0.58,
ts_rank ~0.09 — шкалы несопоставимы и плавают от запроса к запросу.
RRF выбрасывает оценки и работает только с позициями:

    score(d) = Σ 1 / (k + rank_i(d))

k = 60 — из оригинальной статьи Cormack et al. 2009. Гасит разницу
между верхними позициями, чтобы один метод не забивал второй.

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

def _load_env():
    """Читает .env в корне проекта, не затирая уже заданные переменные.

    Раньше .env подгружала IDE — из голого терминала падало на
    fe_sendauth: no password supplied. Зависеть от того, откуда запущено,
    для воспроизводимости замеров негодится.

    Без python-dotenv: одна зависимость ради десяти строк не нужна.
    """
    path = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip("\"'"))


_load_env()

DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://rag:{}@localhost:5433/ragdb".format(
        os.environ.get("POSTGRES_PASSWORD", "")
    ),
)

MODEL = "bge-m3"
OLLAMA = "http://localhost:11434/api/embed"
RRF_K = 60
POOL = 50          # сколько берём из каждого метода до слияния

# HNSW — ПРИБЛИЖЁННЫЙ индекс. Замерено на нашем корпусе:
#   ef_search=40  (дефолт) → 第39条 НЕ попадает в топ-10 вообще
#   ef_search=100            → 第39条 на 6-м месте (= точный перебор numpy)
#   ef_search=400            → то же самое, выигрыша нет
# Дефолт теряет результаты МОЛЧА — ни ошибки, ни предупреждения.
EF_SEARCH = 100

FIELDS = ("id, chunk_id, text, law_id, law_title, law_type, core, "
          "article_title, article_caption, paragraph_num")


class Hybrid:
    def __init__(self):
        self.conn = psycopg.connect(DSN)
        register_vector(self.conn)
        self.conn.execute(f"SET hnsw.ef_search = {EF_SEARCH}")
        self.http = httpx.Client()

    def embed(self, text):
        r = self.http.post(OLLAMA, json={"model": MODEL, "input": text},
                           timeout=60)
        r.raise_for_status()
        return np.array(r.json()["embeddings"][0], dtype=np.float32)

    def vector_search(self, query, limit=POOL, core_only=False):
        """Векторный. <#> — ОТРИЦАТЕЛЬНОЕ скалярное произведение,
        поэтому ASC даёт самые похожие первыми."""
        q = self.embed(query)
        where = "WHERE core" if core_only else ""
        sql = f"""
            SELECT {FIELDS}, (embedding <#> %s) * -1 AS score
            FROM chunks {where}
            ORDER BY embedding <#> %s
            LIMIT %s
        """
        return self._rows(sql, (q, q, limit))

    def bm25_search(self, query, limit=POOL, core_only=False, min_rank=0.0):
        """Лексический через tsvector. Запрос токенизируется
        ТЕМ ЖЕ Sudachi, что и корпус — иначе токены не совпадут."""
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
        """RRF по нескольким ранжированиям."""
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

        # multi-query: ищем ПО ОБОИМ вариантам запроса и сливаем.
        # Решает проблему из замера: расширение чинит 7 вопросов,
        # но ломает 3 — те, где оригинал был первым. При слиянии
        # такой результат не теряется — он получает 1/(60+1) из своего списка.
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
    print(f"запрос: {query}")
    print(f"токены: {tokenize(query)}")
    print(f"режим: {mode}" + (" / core" if core_only else "") + "\n")

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