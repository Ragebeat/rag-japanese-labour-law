"""Japanese numerals and article references — shared parsing.

Lifted out of generate.py: expand_golden.py uses the same parsing, and
pulling the whole search pipeline in there for two functions makes no sense.

Article numbers in the corpus are always kanji (第三十九条); in model output
and in queries they come however they come (第39条 / 第３９条). Comparison has
to happen on a canonical form on both sides.
"""

import re

_ZEN = str.maketrans("０１２３４５６７８９", "0123456789")
_DIG = {"〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_UNIT = {"十": 10, "百": 100, "千": 1000}


def kanji_int(s):
    """三十九 -> 39, 百三十 -> 130. None if this is not a kanji numeral."""
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

# Any mention of an article: 第39条, 第九条の二
ART_RE = re.compile(rf"第({NUM})条(?:の({NUM}))?")

# A reference to an article of the PARENT statute from an ordinance:
# 「法第三十九条第七項の厚生労働省令で定める…」. Inside a 施行規則 the form 法第N条
# always points at its own statute by convention; there is no other 「法」 there.
REF_RE = re.compile(rf"法第({NUM})条(?:の({NUM}))?")


def _keys(rx, text):
    out = set()
    for main, sub in rx.findall(text):
        m = num(main)
        if m is not None:
            out.add((m, num(sub) if sub else None))
    return out


def art_key(text):
    """Every article mentioned, as canonical pairs: 第39条の2 -> (39, 2)."""
    return _keys(ART_RE, text)


def ref_key(text):
    """Only references of the form 法第N条 — to the parent statute."""
    return _keys(REF_RE, text)


def fmt(keys):
    return sorted(f"第{m}条" + (f"の{s}" if s else "") for m, s in keys)
