"""Query robustness: do full-width / half-width / ASCII-punctuation variants of
the golden questions still retrieve the gold article?

Retrieval only, no generation. Baseline hits come from the last --eval report
(retrieval is deterministic, so no need to recompute them).
Only questions the variant actually changes are re-run (reranking is ~10 s each).

  uv run python src\\perturb_test.py
"""
import json
import unicodedata

import generate as g
from rerank import Reranked


def zen(q):
    """ASCII digits/letters -> full-width: 36協定 -> ３６協定."""
    return "".join(chr(ord(c) + 0xFEE0) if c.isascii() and c.isalnum() else c
                   for c in q)


def _hankaku_map():
    m = {}
    for cp in range(0xFF61, 0xFFA0):            # half-width block
        h = chr(cp)
        f = unicodedata.normalize("NFKC", h)
        if f != h:
            m.setdefault(f, h)
    for cp in range(0x30A1, 0x30FB):            # ガ -> ｶ + ﾞ, パ -> ﾊ + ﾟ
        f = chr(cp)
        d = unicodedata.normalize("NFD", f)
        if len(d) == 2 and d[1] in "\u3099\u309a" and d[0] in m:
            m[f] = m[d[0]] + ("\uff9e" if d[1] == "\u3099" else "\uff9f")
    return m


_HK = _hankaku_map()


def kana(q):
    """Katakana -> half-width: パワハラ -> ﾊﾟﾜﾊﾗ."""
    out = "".join(_HK.get(c, c) for c in q)
    assert unicodedata.normalize("NFKC", out) == unicodedata.normalize("NFKC", q), q
    return out


def asciiq(q):
    """？ -> ? plus a trailing space: how people type on a phone/keyboard."""
    return q.replace("？", "?").replace("！", "!") + " "


VARIANTS = {"zen": zen, "kana": kana, "asciiq": asciiq}

gold = g.load_golden()
data = json.load(open(g.ROOT / "data" / "gen_report_quotes_whole.json",
                      encoding="utf-8"))
rows = data if isinstance(data, list) else data.get("rows", data)
base = {x["id"]: x["hit"] for x in rows}

r = Reranked()
for name, fn in VARIANTS.items():
    changed = [x for x in gold if fn(x["q"]) != x["q"]]
    lost, gained = [], []
    for x in changed:
        ch = g.retrieve(r, fn(x["q"]), g.DEFAULT_MODE)
        want = {tuple(a) for a in x["relevant"]}
        got = {(c["law_id"], c["article_title"]) for c in ch}
        hit = bool(want & got)
        if base[x["id"]] and not hit:
            lost.append(x["id"])
        if not base[x["id"]] and hit:
            gained.append(x["id"])
        print(f"  {name} #{x['id']:<3} {'hit ' if hit else 'miss'}", end="\r", flush=True)
    print(f"{name:7s} вопросов изменено {len(changed):>2}   потеряно {lost}   появилось {gained}")
