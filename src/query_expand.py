"""Расширение запроса словарём разговорных → юридических терминов.

Словарь лежит в `data/synonyms.json` — пополняется без правки кода.

Диагноз из занятия 3: гибридный поиск провалился из-за терминологического
разрыва: パワハラ и 育休 в тексте законов ОТСУТСТВУЮТ вообще.

⚠️ Словарь составлен ПО ЗНАНИЮ ДОМЕНА, а НЕ по списку проваленных
вопросов golden. Иначе это было бы вписывание ответов в словарь,
а не улучшение поиска.

⚠️ Главное ограничение: словарь покрывает только предусмотренное.
На незнакомом термине он молчит. Отсюда связка с LLM-fallback
(см. query_llm.py) — словарь для частотного, модель для остального.

  python src/query_expand.py "育休っていつまで？"
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DICT_PATH = ROOT / "data" / "synonyms.json"

_syn = None


def synonyms():
    """Словарь из JSON. Ключи на `_` — комментарии, пропускаются."""
    global _syn
    if _syn is None:
        raw = json.loads(DICT_PATH.read_text(encoding="utf-8"))
        _syn = {k: v for k, v in raw.items()
                if not k.startswith("_") and isinstance(v, list) and v}
    return _syn


def expand(query, narrow=False):
    """Добавляет юридические эквиваленты К запросу, не заменяя его.

    Заменять было бы рискованно: если словарь ошибётся, теряется
    исходный смысл. Добавление только расширяет покрытие.

    narrow=True — только первый термин из списка. На замере видно,
    что длинный хвост размывает: 労災 → 業務災害 通勤災害 補償
    тянет к статьям про 通勤, хотя спрашивали про 休業補償.
    """
    extra = []
    for slang, legal in synonyms().items():
        if slang not in query:
            continue
        terms = legal[:1] if narrow else legal
        extra.extend(t for t in terms if t not in query)
    if not extra:
        return query, []
    return query + " " + " ".join(extra), extra


if __name__ == "__main__":
    tests = sys.argv[1:] or [
        "育休っていつまで取れる？",
        "パワハラ対策は会社の義務？",
        "36協定って何？",
        "通勤中のケガも労災？",
        "1日何時間まで働かせていいの？",
    ]
    print(f"словарь: {len(synonyms())} пар → {DICT_PATH}\n")
    for q in tests:
        _, extra = expand(q, narrow=True)
        print(f"{q}\n   narrow: {extra or '—'}")