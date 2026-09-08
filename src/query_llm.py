"""Query expansion with a local LLM — an alternative to the dictionary.

The hypothesis is not "an LLM beats a dictionary" but the narrower one: **an
LLM will find substitutions where the dictionary is silent**. Example:
1日何時間まで働かせていいの？ — not a single colloquial term, so the dictionary
does not fire.

Decisions carried over from phase 3:
- instructions and examples in system, the user question in user
- structured output via format=json
- **the schema allows an EMPTY list** — otherwise the model invents terms
  where there are none (covered in phase 3: a schema with no field for
  "nothing found" produces fabrication instead of "I do not know")

Cached to disk: 50 questions times several runs, and Ollama swaps models on
top of that (the embedder and the generator do not fit in 6 GB together).

  python src/query_llm.py            — run over the test questions
  python src/query_llm.py --warm     — warm the cache over the whole golden set
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

# Fixes after the first measurement (vector+llm 0.740 against dictionary 0.780):
#
# 1. The phrase 休憩時間を除き一週間について四十時間 from an example stuck and
#    turned up in SEVEN unrelated questions (生理休暇, 育児時間, バイトの有給...).
#    -> examples are now short and carry no long quotations from articles
#
# 2. Fabricated figures: 最長180日 (育休 is in fact up to a year),
#    最大180日 (介護休業 is 93 days), 6か月 (in fact 12 over 2 years).
#    -> numbers and periods are now forbidden explicitly
#
# 3. Everyday vocabulary: 不当解雇, 性的ハラスメント, 平等の原則,
#    正規の雇用形態 — none of these appear in the statutory text.
#    -> the rule is reinforced with examples of BAD terms

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
        # AN EMPTY LIST IS ALLOWED — minItems is deliberately omitted
        "terms": {"type": "array", "items": {"type": "string"},
                  "maxItems": MAX_TERMS},
    },
    "required": ["terms"],
}

_cache = None


# Simplified Chinese characters that do not exist in Japanese.
# qwen3.5 produced 最低工资 and 劳动基准法 — tokens like that will never
# be found in a Japanese corpus. The prompt does not always hold, so a
# post-filter is needed. The list is incomplete but covers the frequent cases.
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
    """The key includes THE PROMPT ITSELF — otherwise, after editing it,
    stale answers would be pulled in and the measurement would show nothing."""
    h = hashlib.sha1(f"{MODEL}|{SYSTEM}|{q}".encode()).hexdigest()
    return h[:16]


def llm_terms(query, client=None):
    """Legal terms for the question. May be empty."""
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
        # post-filter: drop terms containing digits and overly long ones —
        # the model ignores the instruction in roughly a third of cases
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
    """Adds terms TO the query — the same mechanism as the dictionary."""
    terms = llm_terms(query, client)
    if not terms:
        return query, []
    return query + " " + " ".join(terms), terms


def expand_combo(query, client=None, narrow=True):
    """The dictionary for the frequent cases, the LLM for the rest.

    The dictionary is more precise (a human checked every pair) but covers
    only what was anticipated and does not scale without a lawyer. The LLM is
    broader but confuses neighbouring institutions (育児時間 -> 育児休業).

    Measurement put them level (0.78 against 0.76, a difference of one
    question), but the golden set consists of questions from the same domain
    the dictionary was written for — on unfamiliar phrasings it simply stays
    silent.
    """
    _, dict_terms = expand(query, narrow=narrow)
    if dict_terms:
        return query + " " + " ".join(dict_terms), dict_terms
    terms = llm_terms(query, client)
    if not terms:
        return query, []
    return query + " " + " ".join(terms), terms


def warm():
    """Warm the cache over the whole golden set, so the measurement runs fast."""
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