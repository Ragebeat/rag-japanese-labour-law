"""A form for manual annotation of answers — the reference set for an LLM judge.

Why: #31 (the right article was in the context, the model did not take it) is
invisible to citation_validity, to the verdict and to the score gate. Cases
like that are only caught by a human. Twenty annotated questions become the
set on which an LLM judge is checked BEFORE it is trusted — otherwise the
judge is as unverified an instrument as HNSW with its default ef_search.

SAMPLING (the rule was written before generating):
    from the questions that hit — a systematic sample of every second one by
    id, first 20. Not random and not "the interesting ones": the golden set is
    ordered by statute, so a step over ids preserves domain coverage, while
    picking "interesting" cases would fit the calibration set to known problems.

THREE FLAGS, independent of one another:
    substance  — does it answer the question: full / partial / none
    distortion — does it distort the norm (wrong period, wrong amount, a duty
                 presented as a recommendation): yes / no
    extra      — does it drag in unrelated material from junk chunks: yes / no

Why three rather than one score: an answer can be substantively right and
still carry junk (#4, #5), or cite the article correctly and get the period
wrong. These are different defects with different costs; averaging them is
not allowed.

  uv run python src/make_review.py
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORT = ROOT / "data" / "gen_report_whole.json"
OUT = ROOT / "data" / "manual_review.md"

N = 20
STEP = 2

HEAD = """# Ручная разметка ответов — эталон для судьи

Заполнить каждую строку `grade:`, заменив `?`. Ничего больше не трогать —
файл разбирается регексом.

- **substance** = `full` (ответил на вопрос) · `partial` (по теме, но мимо сути)
  · `none` (не ответил)
- **distortion** = `yes` / `no` — искажено ли содержание нормы: неверный срок,
  сумма, обязанность подана как рекомендация, норма распространена не на тех
- **extra** = `yes` / `no` — притащено ли постороннее из мусорных чанков

Оценивать **ответ против вопроса**, а не против golden. Golden узок
(занятие 7), и ответ по 施行規則 может быть верным.

Если сомневаешься между `full` и `partial` — ставь `partial`.

---

"""


def main():
    if not REPORT.exists():
        sys.exit(f"нет {REPORT.name} — сначала generate.py --eval")
    if OUT.exists() and "--force" not in sys.argv:
        sys.exit(f"{OUT.name} уже есть — в нём разметка, перезапись её сотрёт.\n"
                 f"Посмотреть: --parse · пересобрать с нуля: --force")

    rows = [r for r in json.loads(REPORT.read_text(encoding="utf-8")) if r["hit"]]
    rows.sort(key=lambda r: r["id"])
    picked = rows[::STEP][:N]

    if len(picked) < N:
        print(f"⚠️ доступно только {len(picked)} вопросов при шаге {STEP}")

    parts = [HEAD]
    for r in picked:
        parts.append(f"## #{r['id']}  {r['q']}\n")
        parts.append("**Контекст:**\n")
        for i, c in enumerate(r["context"], 1):
            parts.append(f"{i}. `{c}`")
        parts.append("\n**Ответ:**\n")
        parts.append("> " + r["answer"].replace("\n", "\n> "))
        parts.append("")
        parts.append("`grade: substance=?  distortion=?  extra=?`")
        parts.append("")
        parts.append("заметка:")
        parts.append("\n---\n")

    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"вопросов в бланке: {len(picked)}  (id: "
          f"{', '.join(str(r['id']) for r in picked)})")
    print(f"→ {OUT}")


def parse():
    """Reads the filled-in form. Called from the future comparison with the judge."""
    text = OUT.read_text(encoding="utf-8")
    out = {}
    for block in text.split("## #")[1:]:
        qid = int(block.split()[0])
        m = re.search(r"grade:\s*substance=(\w+)\s+distortion=(\w+)\s+extra=(\w+)",
                      block)
        if m and "?" not in m.group(0):
            out[qid] = dict(substance=m.group(1), distortion=m.group(2),
                            extra=m.group(3))
    return out


if __name__ == "__main__":
    if "--parse" in sys.argv:
        g = parse()
        print(f"заполнено: {len(g)}")
        for k, v in sorted(g.items()):
            print(f"  #{k}: {v}")
    else:
        main()
