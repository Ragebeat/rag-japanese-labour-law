"""Corpus statistics: documents by type, article and paragraph length, token estimates.

The chunking rules in chunk.py were derived from these distributions.

  uv run python src/analyze.py
"""

import json
import pathlib
import re
import statistics as st
import xml.etree.ElementTree as ET

import tiktoken

RAW = pathlib.Path("data/raw")
MANIFEST = json.loads(pathlib.Path("data/manifest.json").read_text(encoding="utf-8"))
META = {r["law_id"]: r for r in MANIFEST}
ENC = tiktoken.encoding_for_model("gpt-4o")


def text_of(el):
    return re.sub(r"\s+", "", "".join(el.itertext()))


def pct(values, p):
    if not values:
        return 0
    s = sorted(values)
    i = min(int(len(s) * p / 100), len(s) - 1)
    return s[i]


def describe(name, values):
    if not values:
        print(f"{name}: empty")
        return
    print(f"\n{name}  (n={len(values)})")
    print(
        f"  min {min(values):>6}   p25 {pct(values, 25):>6}   p50 {pct(values, 50):>6}"
        f"   p75 {pct(values, 75):>6}   p90 {pct(values, 90):>6}"
        f"   p99 {pct(values, 99):>6}   max {max(values):>7}"
    )
    print(f"  mean {st.mean(values):.0f}   sum {sum(values)}")


def main():
    files = sorted(RAW.glob("*.xml"))
    doc_sizes = {}
    by_type = {}
    art_chars, par_chars = [], []
    art_deleted = 0
    suppl_chars = main_chars = 0
    big_articles = []
    par_samples = []  # for the token measurement

    for f in files:
        law_id = f.stem
        raw = f.read_text(encoding="utf-8")
        doc_sizes[law_id] = len(raw)
        lt = META.get(law_id, {}).get("law_type", "?")
        d = by_type.setdefault(lt, {"n": 0, "chars": 0})
        d["n"] += 1
        d["chars"] += len(raw)

        root = ET.fromstring(raw)

        for sp in root.iter("SupplProvision"):
            suppl_chars += len(text_of(sp))
        for mp in root.iter("MainProvision"):
            main_chars += len(text_of(mp))

        # articles from the main body only
        for mp in root.iter("MainProvision"):
            for art in mp.iter("Article"):
                t = text_of(art)
                art_chars.append(len(t))
                if "削除" in t and len(t) < 30:
                    art_deleted += 1
                    continue
                num = art.get("Num", "?")
                title = art.findtext("ArticleTitle", "")
                cap = art.findtext("ArticleCaption", "")
                big_articles.append((len(t), law_id, num, title, cap))
                for par in art.iter("Paragraph"):
                    pt = text_of(par)
                    if pt:
                        par_chars.append(len(pt))
                        if len(par_samples) < 300:
                            par_samples.append(pt)

    print(f"documents: {len(files)}   characters total: {sum(doc_sizes.values())}")

    print("\nby type:")
    for lt, d in sorted(by_type.items(), key=lambda x: -x[1]["chars"]):
        print(f"  {lt:<22} {d['n']:>4} docs {d['chars']:>9} chars")

    print("\nlargest documents:")
    for law_id, n in sorted(doc_sizes.items(), key=lambda x: -x[1])[:10]:
        title = META.get(law_id, {}).get("law_title", "?")
        print(f"  {n:>8}  {law_id}  {title[:40]}")

    describe("ARTICLES, characters", art_chars)
    describe("PARAGRAPHS, characters", par_chars)

    print(f"\n削除 articles: {art_deleted} of {len(art_chars)} ({art_deleted / max(len(art_chars), 1) * 100:.1f}%)")
    total = main_chars + suppl_chars
    print(f"本則 {main_chars} chars / 附則 {suppl_chars} chars  →附則 = {suppl_chars / max(total, 1) * 100:.1f}%")

    print("\nlongest articles:")
    for n, law_id, num, title, cap in sorted(big_articles, reverse=True)[:10]:
        print(f"  {n:>6}  {law_id}  {title}{cap}")

    # characters -> tokens on the real corpus
    sample = "".join(par_samples)
    tok = len(ENC.encode(sample))
    print(f"\ntokenising a sample ({len(par_samples)} paragraphs, {len(sample)} chars):")
    print(f"  tokens: {tok}   chars per token: {len(sample) / tok:.2f}")
    print(f"  whole corpus estimate (本則): ~{int(main_chars / (len(sample) / tok))} tokens")

    ratio = tok / len(sample)
    print("\nparagraphs in TOKENS (estimated via the ratio):")
    est = [int(c * ratio) for c in par_chars]
    describe("PARAGRAPHS, tokens", est)


if __name__ == "__main__":
    main()
