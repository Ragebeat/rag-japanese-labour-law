"""Query expansion через локальную LLM — альтернатива словарю.

Гипотеза не «LLM лучше словаря», а уже: **LLM найдёт замены там,
где словарь молчит**. Пример: 1日何時間まで働かせていいの？ — ни одного
разговорного термина, словарь не срабатывает.

Решения из фазы 3:
- инструкция и примеры в system, вопрос пользователя в user
- structured output через format=json
- **схема допускает ПУСТОЙ список** — иначе модель выдумает термины
  там, где их нет (разбирали в фазе 3: схема без поля для отказа
  → выдумка вместо «не знаю»)

Кэш на диск: 50 вопросов × несколько прогонов, а Ollama ещё и свопает
модели (эмбеддер + генератор не влезают в 6 ГБ одновременно).

  python src/query_llm.py            — прогнать по тестовым вопросам
  python src/query_llm.py --warm     — прогреть кэш по всему golden
"""

import hashlib
import json
import os
import pathlib
import sys
import time

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from query_expand import expand  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "llm_expand_cache.json"
GOLDEN = ROOT / "data" / "golden.jsonl"

MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")
URL = "http://localhost:11434/api/chat"
MAX_TERMS = 4

# Правки после первого замера (vector+llm 0.740 против словаря 0.780):
#
# 1. Фраза из примера 休憩時間を除き一週間について四十時間 прилипла и попала
#    в СЕМЬ посторонних вопросов (生理休暇, 育児時間, バイトの有給...).
#    → примеры теперь короткие и без длинных цитат из статей
#
# 2. Галлюцинации цифр: 最長180日 (育休 на самом деле до года),
#    最大180日 (介護休業 — 93 дня), 6か月 (на деле 12 за 2 года).
#    → числа и сроки запрещены явно
#
# 3. Бытовая лексика: 不当解雇, 性的ハラスメント, 平等の原則,
#    正規の雇用形態 — все отсутствуют в тексте законов.
#    → правило усилено примерами ПЛОХИХ терминов

SYSTEM = """日本の労働法令の検索用に、口語の質問から法令用語を抽出してください。

【必ず守ること】
1. 法律の条文にそのまま登場する単語だけ。一般的な言い方は使わない
2. 数字・日数・期間・金額は絶対に含めない
3. 短い単語（漢字数文字）。条文の長い引用は不可
4. 自信がなければ含めない。空の配列でもよい
5. 最大4語。JSONのみ出力
6. 【重要】日本語の漢字のみ。中国語の簡体字は絶対に使わない
   × 最低工资、劳动基准法、劳动者、休息
   ○ 最低賃金、労働基準法、労働者、休憩

【使ってはいけない語の例】
× 不当解雇、ハラスメント、平等の原則、正規雇用、受診義務
（これらは条文に存在しない表現）

【出力形式】
{"terms": ["用語1", "用語2"]}

【例】
質問: クビになったらすぐ出ていくの？
{"terms": ["解雇", "予告"]}

質問: 今日の天気は？
{"terms": []}"""

SCHEMA = {
    "type": "object",
    "properties": {
        # ПУСТОЙ СПИСОК РАЗРЕШЁН — minItems не ставим сознательно
        "terms": {"type": "array", "items": {"type": "string"},
                  "maxItems": MAX_TERMS},
    },
    "required": ["terms"],
}

_cache = None


# Упрощённые китайские иероглифы, которых нет в японском.
# qwen3.5 выдавала 最低工资 и 劳动基准法 — в японском корпусе
# такие токены не найдутся никогда. Промпт не всегда удерживает —
# нужен постфильтр. Список неполный, но покрывает частотное.
SIMPLIFIED = set("资动劳务单岁权应长时间关术员类问题说语边达过还进运远连这那们个为从优产价业乐习书买卖发号只叶团国园图坚报择挣据无旧显机权条来构标样权级经络统编考计记论证该请质走车转边适选都销际预领频题")


def _looks_chinese(t):
    return any(ch in SIMPLIFIED for ch in t)


def _load_cache():
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE.read_text(encoding="utf-8")) \
            if CACHE.exists() else {}
    return _cache


def _save_cache():
    CACHE.write_text(json.dumps(_cache, ensure_ascii=False, indent=1),
                     encoding="utf-8")


def _key(q):
    """Ключ включает САМ ПРОМПТ — иначе после его правки
    подтянутся старые ответы и замер ничего не покажет."""
    h = hashlib.sha1(f"{MODEL}|{SYSTEM}|{q}".encode()).hexdigest()
    return h[:16]


def llm_terms(query, client=None):
    """Список юридических терминов по вопросу. Может быть пустым."""
    cache = _load_cache()
    k = _key(query)
    if k in cache:
        return cache[k]

    own = client is None
    client = client or httpx.Client()
    try:
        r = client.post(URL, json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": query},
            ],
            "format": SCHEMA,
            "stream": False,
            "think": False,
            "options": {"temperature": 0},
        }, timeout=300)
        r.raise_for_status()
        raw = r.json()["message"]["content"]
        terms = json.loads(raw).get("terms", [])
        # постфильтр: отбрасываем термины с цифрами и слишком длинные —
        # модель игнорирует инструкцию примерно в трёти случаев
        clean = []
        for t in terms:
            t = (t or "").strip()
            if not t or len(t) > 12:
                continue
            if any(ch.isdigit() for ch in t):
                continue
            if any(ch in t for ch in "一二三四五六七八九十百"):
                continue
            if _looks_chinese(t):
                continue
            clean.append(t)
        terms = clean[:MAX_TERMS]
    except Exception as e:
        print(f"    LLM ошибка на '{query[:20]}': {e}")
        terms = []
    finally:
        if own:
            client.close()

    cache[k] = terms
    _save_cache()
    return terms


def expand_llm(query, client=None):
    """Добавляет термины К запросу — тот же механизм, что у словаря."""
    terms = llm_terms(query, client)
    if not terms:
        return query, []
    return query + " " + " ".join(terms), terms


def expand_combo(query, client=None, narrow=True):
    """Словарь для частотного, LLM для остального.

    Словарь точнее (человек проверил каждую пару), но покрывает
    только предусмотренное и не масштабируется без юриста.
    LLM шире, но путает близкие институты (育児時間 → 育児休業).

    Замер показал их равными (0.78 против 0.76, разница = 1 вопрос),
    но golden состоит из вопросов того же домена, под который писался
    словарь — на незнакомых формулировках он просто промолчит.
    """
    _, dict_terms = expand(query, narrow=narrow)
    if dict_terms:
        return query + " " + " ".join(dict_terms), dict_terms
    terms = llm_terms(query, client)
    if not terms:
        return query, []
    return query + " " + " ".join(terms), terms


def warm():
    """Прогреть кэш по всему golden — чтобы замер потом был быстрым."""
    golden = [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]
    t0 = time.time()
    empty = 0
    with httpx.Client() as c:
        for i, g in enumerate(golden, 1):
            terms = llm_terms(g["q"], c)
            if not terms:
                empty += 1
            print(f"{i:>2}/{len(golden)} {g['q'][:26]:<28} → {terms}")
    print(f"\nготово за {(time.time()-t0)/60:.1f} мин, пустых ответов: {empty}")


if __name__ == "__main__":
    if "--warm" in sys.argv:
        warm()
    else:
        for q in [
            "パワハラ対策は会社の義務？",
            "1日何時間まで働かせていいの？",
            "育休っていつまで取れる？",
            "5年働いたら正社員になれる？",
            "週に何日休みが必要？",
            "今日の天気は？",
        ]:
            _, terms = expand_llm(q)
            print(f"{q}\n   → {terms}")