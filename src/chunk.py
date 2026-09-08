"""Chunking the Japanese labour law corpus.

Rules:
  1. The base unit is a Paragraph (median 91 tokens, p90 = 291).
  2. A paragraph longer than MAX_CHARS is split by Item.
  3. An Item longer than MAX_CHARS is split by sentence, with overlap.
Dropped: SupplProvision (附則, 37.6%), 削除 articles, 読替え articles.
"""

import json
import pathlib
import re
import xml.etree.ElementTree as ET
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.json"
OUT = ROOT / "data" / "chunks.jsonl"

MAX_CHARS = 800          # threshold past which we split deeper
OVERLAP_SENT = 1         # sentences of overlap at level 3
SKIP_CAPTION = ("読替え", "読み替え")


def norm(s):
    """In Japanese legal text whitespace is nothing but XML indentation."""
    return re.sub(r"\s+", "", s or "")


def text_of(el):
    return norm("".join(el.itertext())) if el is not None else ""


def split_sentences(text):
    return [p for p in re.split(r"(?<=。)", text) if p]


PAIRS = [("（", "）"), ("「", "」"), ("(", ")"), ("『", "』")]


def balanced(s):
    """The 。 full stop occurs inside brackets — no splitting there."""
    return all(s.count(o) == s.count(c) for o, c in PAIRS)


def pack_sentences(text, limit=MAX_CHARS, overlap=OVERLAP_SENT):
    sents = split_sentences(text)
    out, cur, fresh = [], [], False
    for s in sents:
        cur.append(s)
        fresh = True
        joined = "".join(cur)
        if len(joined) >= limit and balanced(joined):
            out.append(joined)
            cur = cur[-overlap:] if overlap else []
            fresh = False
    if fresh:
        out.append("".join(cur))
    return out or [text]


def make(law, label, body, art_num, art_title, caption, p_num, i_num, part, level):
    cid = f"{law['law_id']}_a{art_num}_p{p_num}"
    if i_num:
        cid += f"_i{i_num}"
    if part is not None:
        cid += f"_s{part}"
    return {
        "chunk_id": cid,
        "text": f"{label}\n{body}",
        "law_id": law["law_id"],
        "law_title": law["law_title"],
        "law_type": law["law_type"],
        "abbrev": law.get("abbrev"),
        "core": law.get("core", False),
        "enforcement_date": law.get("enforcement_date"),
        "article_num": art_num,
        "article_title": art_title,
        "article_caption": caption,
        "paragraph_num": p_num,
        "item_num": i_num,
        "level": level,
        "n_chars": len(body),
    }


def article_chunks(art, law, stats):
    art_num = art.get("Num", "")
    art_title = norm(art.findtext("ArticleTitle", ""))
    caption = norm(art.findtext("ArticleCaption", ""))

    whole = text_of(art)
    if "削除" in whole and len(whole) < len(art_title) + 20:
        stats["skip_deleted"] += 1
        return
    if any(k in caption for k in SKIP_CAPTION):
        stats["skip_yomikae"] += 1
        return

    head = f"{law['law_title']} {art_title}" + (caption if caption else "")

    for para in art.findall("Paragraph"):
        p_num = para.get("Num", "1")
        p_text = text_of(para)
        if not p_text:
            continue
        label = f"{head} 第{p_num}項"

        if len(p_text) <= MAX_CHARS:
            stats["lvl1"] += 1
            yield make(law, label, p_text, art_num, art_title, caption,
                       p_num, None, None, 1)
            continue

        items = para.findall("Item")
        if not items:
            for k, piece in enumerate(pack_sentences(p_text)):
                stats["lvl3"] += 1
                yield make(law, label, piece, art_num, art_title, caption,
                           p_num, None, k, 3)
            continue

        lead = text_of(para.find("ParagraphSentence"))
        for it in items:
            i_num = it.get("Num", "")
            body = lead + text_of(it)
            ilabel = f"{label} 第{i_num}号"
            if len(body) <= MAX_CHARS:
                stats["lvl2"] += 1
                yield make(law, ilabel, body, art_num, art_title, caption,
                           p_num, i_num, None, 2)
            else:
                for k, piece in enumerate(pack_sentences(body)):
                    stats["lvl3"] += 1
                    yield make(law, ilabel, piece, art_num, art_title, caption,
                               p_num, i_num, k, 3)


def main():
    laws = {r["law_id"]: r for r in json.loads(MANIFEST.read_text(encoding="utf-8"))}
    stats = Counter()
    sizes = []
    n_core = 0

    with OUT.open("w", encoding="utf-8") as fh:
        for f in sorted(RAW.glob("*.xml")):
            law = laws.get(f.stem)
            if law is None:
                stats["no_manifest"] += 1
                continue
            root = ET.parse(f).getroot()
            for mp in root.iter("MainProvision"):
                for art in mp.iter("Article"):
                    for ch in article_chunks(art, law, stats):
                        fh.write(json.dumps(ch, ensure_ascii=False) + "\n")
                        sizes.append(ch["n_chars"])
                        if ch["core"]:
                            n_core += 1

    def pct(p):
        s = sorted(sizes)
        return s[min(int(len(s) * p / 100), len(s) - 1)]

    print(f"чанков всего: {len(sizes)}   из них core: {n_core}")
    print(f"  уровень 1 (параграф):   {stats['lvl1']}")
    print(f"  уровень 2 (по Item):    {stats['lvl2']}")
    print(f"  уровень 3 (предложения):{stats['lvl3']}")
    print(f"пропущено: 削除 {stats['skip_deleted']}, 読替え {stats['skip_yomikae']}")
    if sizes:
        print(f"размер чанка, симв: min {min(sizes)}  p50 {pct(50)}  "
              f"p90 {pct(90)}  p99 {pct(99)}  max {max(sizes)}")
        print(f"суммарно символов: {sum(sizes)}")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
