"""Японские числа и ссылки на статьи — общий разбор.

Вынесено из generate.py: тем же разбором пользуется expand_golden.py,
а тянуть туда весь конвейер поиска ради двух функций незачем.

Номера статей в корпусе всегда кандзи (第三十九条), в тексте модели и в
запросах — как придётся (第39条 / 第３９条). Сравнивать надо канонический
вид с обеих сторон.
"""

import re

_ZEN = str.maketrans("０１２３４５６７８９", "0123456789")
_DIG = {"〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_UNIT = {"十": 10, "百": 100, "千": 1000}


def kanji_int(s):
    """三十九 → 39, 百三十 → 130. None, если не кандзи-число."""
    total = cur = 0
    for ch in s:
        if ch in _DIG:
            cur = _DIG[ch]
        elif ch in _UNIT:
            total += (cur or 1) * _UNIT[ch]
            cur = 0
        else:
            return None
    return total + cur


def num(s):
    s = s.translate(_ZEN)
    return int(s) if s.isdigit() else kanji_int(s)


NUM = "[0-9０-９〇一二三四五六七八九十百千]+"

# Любое упоминание статьи: 第39条, 第九条の二
ART_RE = re.compile(rf"第({NUM})条(?:の({NUM}))?")

# Ссылка на статью РОДИТЕЛЬСКОГО закона из подзаконного акта:
# 「法第三十九条第七項の厚生労働省令で定める…」. В 施行規則 конструкция 法第N条
# по соглашению всегда указывает на свой закон, другого «法» там нет.
REF_RE = re.compile(rf"法第({NUM})条(?:の({NUM}))?")


def _keys(rx, text):
    out = set()
    for main, sub in rx.findall(text):
        m = num(main)
        if m is not None:
            out.add((m, num(sub) if sub else None))
    return out


def art_key(text):
    """Все упомянутые статьи как канонические пары: 第39条の2 → (39, 2)."""
    return _keys(ART_RE, text)


def ref_key(text):
    """Только ссылки вида 法第N条 — на родительский закон."""
    return _keys(REF_RE, text)


def fmt(keys):
    return sorted(f"第{m}条" + (f"の{s}" if s else "") for m, s in keys)
