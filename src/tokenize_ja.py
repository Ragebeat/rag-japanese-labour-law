"""Токенизация японского для BM25 через Sudachi.

Решения выведены замерами, не взяты из туториала:

1. Режим A (короткие единицы). В режиме C 年次有給休暇 — ОДИН токен,
   и запрос про 有給 не найдёт его никогда. Низкая избирательность
   частых токенов (労働) лечится самим BM25 через IDF.

2. Индексируем surface(), НЕ normalized_form(). Нормализация Sudachi —
   это каноническое написание, а не словарная форма: する→為る,
   その→其の, 十→10. Таких написаний нет ни в законе, ни в запросе.

3. Фильтруем ПО normalized_form — так し / する / 為る отсеиваются
   одним правилом, а индексируется поверхностная форма.

4. 接尾辞 НЕ выбрасываем — там живёт 者 (労働者, 使用者), ключевые
   слова корпуса.

Знаем заранее, что BM25 не вытянет: パワハラ (в законе 優越的な関係…),
育休 (в законе 育児休業), 36協定. Это для query transformation.
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
    """Sudachi падает на очень длинных строках."""
    if len(text) <= limit:
        return [text]
    return [text[i:i + limit] for i in range(0, len(text), limit)]


def tokenize(text):
    """Список токенов для BM25."""
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
    """Токены через пробел — для Postgres to_tsvector('simple', ...)."""
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