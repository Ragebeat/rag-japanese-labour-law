"""Query expansion with a colloquial-to-legal term dictionary.

The dictionary lives in `data/synonyms.json` and is extended without touching
the code.

The diagnosis from session 3: hybrid search failed because of a terminology
gap — パワハラ and 育休 are ABSENT from the statutory text entirely.

⚠️ The dictionary was written FROM DOMAIN KNOWLEDGE, not from the list of
failed golden questions. Otherwise it would be writing the answers into the
dictionary rather than improving search.

⚠️ Its main limitation: it covers only what was anticipated. On an unfamiliar
term it stays silent. Hence the pairing with an LLM fallback (see
query_llm.py) — the dictionary for the frequent cases, the model for the rest.

  python src/query_expand.py "育休っていつまで？"
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DICT_PATH = ROOT / "data" / "synonyms.json"

_syn = None


def synonyms():
    """The dictionary from JSON. Keys starting with `_` are comments and skipped."""
    global _syn
    if _syn is None:
        raw = json.loads(DICT_PATH.read_text(encoding="utf-8"))
        _syn = {k: v for k, v in raw.items()
                if not k.startswith("_") and isinstance(v, list) and v}
    return _syn


def expand(query, narrow=False):
    """Adds legal equivalents TO the query instead of replacing it.

    Replacing would be risky: if the dictionary is wrong, the original meaning
    is lost. Adding only widens coverage.

    narrow=True — only the first term of each list. The measurements show that
    a long tail dilutes: 労災 -> 業務災害 通勤災害 補償 pulls towards articles
    about 通勤 when the question was about 休業補償.
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