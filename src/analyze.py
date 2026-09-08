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
        print(f"{name}: пусто")
        return
    print(f"\n{name}  (n={len(values)})")
    print(
        f"  min {min(values):>6}   p25 {pct(values, 25):>6}   p50 {pct(values, 50):>6}"
        f"   p75 {pct(values, 75):>6}   p90 {pct(values, 90):>6}"
        f"   p99 {pct(values, 99):>6}   max {max(values):>7}"
    )
    print(f"  mean {st.mean(values):.0f}   сумма {sum(values)}")


def main():
    files = sorted(RAW.glob("*.xml"))
    doc_sizes = {}
    by_type = {}
    art_chars, par_chars = [], []
    art_deleted = 0
    suppl_chars = main_chars = 0
    big_articles = []
    par_samples = []  # для замера токенов

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

        # статьи только из основной части
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

    print(f"документов: {len(files)}   символов всего: {sum(doc_sizes.values())}")

    print("\nпо типам:")
    for lt, d in sorted(by_type.items(), key=lambda x: -x[1]["chars"]):
        print(f"  {lt:<22} {d['n']:>4} шт   {d['chars']:>9} симв")

    print("\nсамые большие документы:")
    for law_id, n in sorted(doc_sizes.items(), key=lambda x: -x[1])[:10]:
        title = META.get(law_id, {}).get("law_title", "?")
        print(f"  {n:>8}  {law_id}  {title[:40]}")

    describe("СТАТЬИ, символов", art_chars)
    describe("ПАРАГРАФЫ, символов", par_chars)

    print(f"\nстатей-削除: {art_deleted} из {len(art_chars)} ({art_deleted / max(len(art_chars), 1) * 100:.1f}%)")
    total = main_chars + suppl_chars
    print(f"本則 {main_chars} симв / 附則 {suppl_chars} симв  →附則 = {suppl_chars / max(total, 1) * 100:.1f}%")

    print("\nсамые длинные статьи:")
    for n, law_id, num, title, cap in sorted(big_articles, reverse=True)[:10]:
        print(f"  {n:>6}  {law_id}  {title}{cap}")

    # символы → токены на реальном корпусе
    sample = "".join(par_samples)
    tok = len(ENC.encode(sample))
    print(f"\nтокенизация выборки ({len(par_samples)} параграфов, {len(sample)} симв):")
    print(f"  токенов: {tok}   символов на токен: {len(sample) / tok:.2f}")
    print(f"  оценка всего корпуса (本則): ~{int(main_chars / (len(sample) / tok))} токенов")

    ratio = tok / len(sample)
    print("\nпараграфы в ТОКЕНАХ (оценка через коэффициент):")
    est = [int(c * ratio) for c in par_chars]
    describe("ПАРАГРАФЫ, токенов", est)


if __name__ == "__main__":
    main()
