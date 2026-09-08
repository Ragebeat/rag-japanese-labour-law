"""Japanese tokenisation for BM25 via Sudachi.

The decisions come from measurements, not from a tutorial:

1. Mode A (short units). In mode C 年次有給休暇 is ONE token, and a query for
   有給 would never match it. The low selectivity of frequent tokens (労働) is
   handled by BM25 itself through IDF.

2. We index surface(), NOT normalized_form(). Sudachi's normalisation is a
   canonical orthography, not a dictionary form: する->為る, その->其の, 十->10.
   Those spellings appear neither in the statute nor in the query.

3. We filter ON normalized_form — that way し / する / 為る are dropped by one
   rule while the surface form is what gets indexed.

4. We do NOT discard 接尾辞: that is where 者 lives (労働者, 使用者), the key
   words of this corpus.

Known in advance to be beyond BM25: パワハラ (the statute says 優越的な関係…),
育休 (the statute says 育児休業), 36協定. Those are for query transformation.
"""

from sudachipy import Dictionary, SplitMode

DROP_POS = {"助詞", "助動詞", "補助記号", "空白", "連体詞", "接続詞"}
DROP_NORM = {"為る", "有る", "居る", "成る", "事", "物", "此れ", "其の", "為", "出来る"}

_tokenizer = None


def _get():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Dictionary().create()
    return _tokenizer


def _split_long(text, limit=10000):
    """Sudachi crashes on very long strings."""
    if len(text) <= limit:
        return [text]
    return [text[i:i + limit] for i in range(0, len(text), limit)]


def tokenize(text):
    """The list of tokens for BM25."""
    t = _get()
    out = []
    for part in _split_long(text):
        for m in t.tokenize(part, SplitMode.A):
            if m.part_of_speech()[0] in DROP_POS:
                continue
            if m.normalized_form() in DROP_NORM:
                continue
            out.append(m.surface())
    return out


def to_tsvector_text(text):
    """Tokens joined by spaces — for Postgres to_tsvector('simple', ...)."""
    return " ".join(tokenize(text))


if __name__ == "__main__":
    law = tokenize(
        "使用者は、その雇入れの日から起算して六箇月間継続勤務し全労働日の"
        "八割以上出勤した労働者に対して、十労働日の有給休暇を与えなければならない。"
    )
    print(f"第39条 ({len(law)}): {' '.join(law)}\n")
    for q in [
        "有給って入社してすぐもらえるの？",
        "パワハラ対策は会社の義務？",
        "育休っていつまで取れる？",
    ]:
        qt = tokenize(q)
        hit = sorted(set(qt) & set(law))
        print(f"{q}\n   {qt}\n   ∩ 第39条: {hit}")