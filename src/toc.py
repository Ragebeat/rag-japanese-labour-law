"""Просмотр оглавления закона и текста статьи — для проверки golden dataset.

  python src/toc.py 育児                 — оглавление закона по куску названия
  python src/toc.py 育児 第五条           — текст конкретной статьи
  python src/toc.py --find 看護休暇       — в каких статьях встречается слово
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "chunks.jsonl"


def load():
    with CHUNKS.open(encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def match_law(c, law_key):
    """law_key — точный law_id либо кусок названия."""
    return law_key == c["law_id"] or law_key in c["law_title"]


def toc(law_key):
    seen, title = set(), None
    for c in load():
        if not match_law(c, law_key):
            continue
        title = c["law_title"]
        k = (c["article_num"], c["article_title"], c["article_caption"])
        if k not in seen:
            seen.add(k)
            print(f"{k[1]}\t{k[2]}")
    if title is None:
        print(f"не найдено: {law_key}")
    else:
        print(f"\n— {title}, статей: {len(seen)}")


def article(law_key, art_title):
    found = False
    for c in load():
        if match_law(c, law_key) and c["article_title"] == art_title:
            found = True
            print(f"[{c['law_title'][:30]}] {c['article_title']} {c['article_caption']}")
            print(c["text"][:400])
            print("-" * 60)
    if not found:
        print(f"не найдено: {law_key} / {art_title}")


def find(word, law_key=None):
    seen = set()
    for c in load():
        if law_key and not match_law(c, law_key):
            continue
        if word in c["text"]:
            k = (c["law_title"], c["article_title"], c["article_caption"])
            if k not in seen:
                seen.add(k)
                print(f"{k[0][:24]:<24} {k[1]}\t{k[2]}")
    if not seen:
        print(f"нигде не встречается: {word}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
    elif a[0] == "--find":
        find(a[1], a[2] if len(a) > 2 else None)
    elif len(a) == 1:
        toc(a[0])
    else:
        article(a[0], a[1])