"""Бланк для ручной разметки ответов — эталон под будущего судью-LLM.

Зачем: #31 (нужная статья в контексте, модель её не взяла) не виден ни
citation_validity, ни вердикту, ни гейту по score. Такие случаи ловятся
только человеком. Двадцать размеченных вопросов станут набором, на котором
проверяется LLM-судья ДО того, как ему поверят — иначе судья будет таким же
непроверенным инструментом, как HNSW с дефолтным ef_search.

ВЫБОРКА (правило записано до генерации):
    из вопросов с hit — систематическая выборка каждого второго по id,
    первые 20. Не случайная и не «интересные»: golden упорядочен по законам,
    поэтому шаг по id сохраняет покрытие доменов, а отбор «интересных» был бы
    подгонкой калибровочного набора под известные проблемы.

ТРИ ФЛАГА, независимые друг от друга:
    substance  — ответил ли по существу вопроса: full / partial / none
    distortion — исказил ли содержание нормы (неверный срок, сумма,
                 обязанность подана как рекомендация): yes / no
    extra      — притащил ли постороннее из мусорных чанков: yes / no

Почему три, а не одна оценка: ответ может быть по существу верным и при этом
тащить мусор (#4, #5), а может верно цитировать статью и переврать срок —
это разные дефекты с разной ценой, усреднять их нельзя.

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
    """Читает заполненный бланк. Вызывается из будущего сравнения с судьёй."""
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
