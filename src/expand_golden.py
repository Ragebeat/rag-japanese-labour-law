"""Expanding the golden set under rule A — a proposal, not an application.

RULE A (written down BEFORE looking at the result, applied to all 50):
    an article of a 施行規則 / 施行令 counts alongside the statutory article
    in `relevant` if it explicitly references it as 「法第N条…」.

REFINEMENT after the first run (193 articles across 35 questions — dilution):
    only if the ordinance article references no more than MAX_PARENTS
    parent articles.

Why: 第六条の二 (selecting the majority representative) references 法第18条,
第24条, 第32条の2, 第39条 and more at once — and landed in five different
questions. That is procedural plumbing, not the implementation of a norm.
Referencing one article means implementing it; referencing eight means it is
a general mechanism.

⚠️ The "caption versus body" axis turned out to be the wrong one: 第六条の二
lists its parent articles right in the caption and passes the strict mode too.

⚠️ The script OVERWRITES NOTHING. It writes golden_proposed.jsonl and
golden_review.md alongside. Every candidate has to be reviewed by eye: the
rule sets the search boundary, it does not replace judgement on the merits —
just as the formal rule for selecting the core corpus needed manual exceptions.

  uv run python src/expand_golden.py              # MAX_PARENTS=1
  uv run python src/expand_golden.py --parents 2  # looser
  uv run python src/expand_golden.py --caption    # reference in the caption only
"""

import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from jnum import art_key, ref_key  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHUNKS = DATA / "chunks.jsonl"
GOLDEN = DATA / "golden.jsonl"
PROPOSED = DATA / "golden_proposed.jsonl"
REVIEW = DATA / "golden_review.md"

# Questions where the mismatch is not "statute versus ordinance" but two
# different legal institutions. Rule A does not take them, and should not:
# covering them would need a rule B, which legitimises any retrofitted
# expansion. We report them on a separate line instead of tuning the metric.
DISPUTED = {
    20: "育介法17条 (制限) против 育介則76条; сам вопрос спорен — в 育休 не работают",
    24: "労契法18条 даёт 無期転換 ≠ 正社員; パート有期法13条 про 転換措置",
    38: "派遣法40条の6 = みなし при незаконном派遣; 40条の4 = обычный путь",
}

# Ordinances the "X施行規則" pattern does not catch.
EXTRA_SUB = {
    "労働安全衛生法": ["労働安全衛生規則"],
}


def load_chunks():
    laws = {}                                    # law_id -> law_title
    arts = collections.defaultdict(dict)         # law_id -> article_title -> text
    for line in CHUNKS.open(encoding="utf-8"):
        c = json.loads(line)
        laws[c["law_id"]] = c["law_title"]
        a = arts[c["law_id"]].setdefault(
            c["article_title"], {"caption": c["article_caption"] or "", "body": []})
        a["body"].append(c["text"])
    return laws, arts


def sub_laws(laws):
    """parent_law_id -> [sub_law_id] for 施行令 / 施行規則."""
    by_title = {}
    for lid, title in laws.items():
        by_title.setdefault(title, lid)
    out = collections.defaultdict(list)
    for lid, title in laws.items():
        for suffix in ("施行令", "施行規則"):
            if title.endswith(suffix):
                parent = by_title.get(title[: -len(suffix)])
                if parent:
                    out[parent].append(lid)
        for parent_title, names in EXTRA_SUB.items():
            if title in names and parent_title in by_title:
                out[by_title[parent_title]].append(lid)
    return out


def main():
    caption_only = "--caption" in sys.argv
    max_parents = 1
    if "--parents" in sys.argv:
        max_parents = int(sys.argv[sys.argv.index("--parents") + 1])

    laws, arts = load_chunks()
    subs = sub_laws(laws)

    golden = [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]
    review, added_total, hubs = [], 0, {}

    for g in golden:
        additions = []
        for law_id, art in g["relevant"]:
            keys = art_key(art)
            for sub_id in subs.get(law_id, []):
                for sub_art, data in arts[sub_id].items():
                    haystack = data["caption"] if caption_only \
                        else data["caption"] + "\n".join(data["body"])
                    refs = ref_key(haystack)
                    if not (keys & refs):
                        continue
                    if len(refs) > max_parents:
                        # plumbing: implements many norms at once rather than this one
                        hubs[f"{laws[sub_id]} {sub_art}"] = len(refs)
                        continue
                    pair = [sub_id, sub_art]
                    if pair in g["relevant"] or any(p == pair for p, _, _ in additions):
                        continue
                    additions.append((pair, len(refs), data["caption"]))

        if g["id"] in DISPUTED:
            g["disputed"] = DISPUTED[g["id"]]

        if additions:
            added_total += len(additions)
            review.append(f"### #{g['id']} {g['q']}\n")
            review.append(f"было: {g['relevant']}\n")
            for pair, n_ref, cap in additions:
                review.append(
                    f"- [ ] `{laws[pair[0]]} {pair[1]}` родителей:{n_ref}  {cap}")
            review.append("")
            g["relevant"] = g["relevant"] + [p for p, _, _ in additions]

    with PROPOSED.open("w", encoding="utf-8") as f:
        for g in golden:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    touched = sum(1 for r in review if r.startswith("### "))
    head = [
        "# Предложение по расширению golden (правило A)",
        "",
        f"Режим: {'ссылка только в заголовке' if caption_only else 'заголовок + тело'}",
        f"Максимум родительских ссылок: {max_parents}",
        f"Затронуто вопросов: {touched} из {len(golden)}",
        f"Добавлено статей: {added_total}",
        "",
        "Отметить галочкой те, что по существу отвечают на вопрос. "
        "Правило задаёт границу поиска, не отменяет проверку.",
        "",
        f"Спорные (правилу A не подчиняются, метрика отдельной строкой): "
        f"{sorted(DISPUTED)}",
        "",
    ]
    if hubs:
        head += ["## Отсеяно как втулки (ссылаются слишком на многое)", ""]
        head += [f"- {k} — родителей: {v}"
                 for k, v in sorted(hubs.items(), key=lambda kv: -kv[1])[:15]]
        head += [""]
    REVIEW.write_text("\n".join(head + review), encoding="utf-8")

    print(f"вопросов затронуто: {touched}/{len(golden)}")
    print(f"статей добавлено:   {added_total}")
    print(f"отсеяно втулок:     {len(hubs)}")
    print(f"→ {REVIEW}")
    print(f"→ {PROPOSED}   (golden.jsonl НЕ тронут)")


if __name__ == "__main__":
    main()
