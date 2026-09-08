"""Выгрузка комментариев и докстрингов на перевод и вставка обратно.

Зачем скриптом, а не руками: правок под две сотни, и любая опечатка в
отступе докстринга — синтаксическая ошибка в файле, который вчера работал.
Машина переставляет байты точнее человека.

    uv run python tools/comments.py extract    # -> tools/comments.json
    uv run python tools/comments.py inject     # <- tools/comments.en.json
    uv run python tools/comments.py check      # что переведено, что нет

ДВА ФАЙЛА. comments.json — выгрузка (id, файл, строка, исходный текст),
машинная, руками не правится. comments.en.json — плоский словарь
{id: "перевод"}, только его и заполняем. Так оригиналы не переписываются
дважды и не разъезжаются.

ПРОВЕРКА НА ДРЕЙФ. При вставке скрипт ищет в файле сохранённый оригинал.
Если между extract и inject файл правился — фрагмент пропускается с
предупреждением, а не портится молча.

Замены идут С КОНЦА ФАЙЛА К НАЧАЛУ: иначе первая же замена сдвинет
координаты всех последующих.

ЧТО КЛАДЁМ В ПЕРЕВОД. Для докстрингов оригинал включает кавычки и
внутренние отступы — перевод должен быть таким же самодостаточным куском
исходника. Для комментариев включает решётку. Чем заменяем, тем и
вставляем, без досборки на месте.
"""

import ast
import io
import json
import pathlib
import sys
import tokenize

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "tools" / "comments.json"
EN = ROOT / "tools" / "comments.en.json"

# Только Python. Фронт (web/src/*.js*) переводится отдельно — там пять
# файлов и десяток комментариев, скрипт под это городить незачем.
TARGETS = ["src/*.py", "tools/*.py"]
SKIP = {"_"}          # файлы на подчёркивание: разовая диагностика

CYR = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")


def _cyr(s):
    return any(c in CYR for c in s.lower())


def files():
    out = []
    for pat in TARGETS:
        for p in sorted(ROOT.glob(pat)):
            if p.name[0] in SKIP or p.name == "comments.py":
                continue
            out.append(p)
    return out


def _offsets(text):
    """Смещение начала каждой строки, чтобы переводить (row, col) в индекс."""
    out, pos = [0], 0
    for line in text.splitlines(keepends=True):
        pos += len(line)
        out.append(pos)
    return out


def _charcol(lines, row, bytecol):
    """ast отдаёт col_offset в БАЙТАХ utf-8, tokenize — в символах.

    На строках с кириллицей и японским это разные числа, и граница
    докстринга уезжала внутрь кода ниже. При inject это затёрло бы
    рабочие строки переводом.
    """
    raw = lines[row - 1].encode("utf-8")
    return len(raw[:bytecol].decode("utf-8", "ignore"))


def extract():
    items = []
    for path in files():
        text = path.read_text(encoding="utf-8")
        offs = _offsets(text)
        lines = text.splitlines(keepends=True)
        rel = path.relative_to(ROOT).as_posix()

        def span(start, end):
            return offs[start[0] - 1] + start[1], offs[end[0] - 1] + end[1]

        # Докстринги: модуль, классы, функции
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, (ast.Module, ast.ClassDef,
                                     ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if not (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                continue
            a, b = span(
                (first.lineno, _charcol(lines, first.lineno, first.col_offset)),
                (first.end_lineno,
                 _charcol(lines, first.end_lineno, first.end_col_offset)))
            items.append({"id": f"{rel}#{first.lineno}", "file": rel,
                          "kind": "docstring", "line": first.lineno,
                          "src": text[a:b]})

        # Комментарии
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type != tokenize.COMMENT:
                continue
            a, b = span(tok.start, tok.end)
            items.append({"id": f"{rel}#{tok.start[0]}c", "file": rel,
                          "kind": "comment", "line": tok.start[0],
                          "src": text[a:b]})

    items.sort(key=lambda i: (i["file"], i["line"]))
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    if not EN.exists():
        EN.write_text("{}\n", encoding="utf-8")

    print(f"файлов: {len(files())}  фрагментов: {len(items)}  "
          f"с кириллицей: {sum(1 for i in items if _cyr(i['src']))}")
    print(f"→ {OUT}")
    print(f"перевод класть в {EN} — плоский словарь id → текст")


def inject():
    if not OUT.exists() or not EN.exists():
        sys.exit("нужны tools/comments.json и tools/comments.en.json")
    items = json.loads(OUT.read_text(encoding="utf-8"))
    en = json.loads(EN.read_text(encoding="utf-8"))

    by_file = {}
    for it in items:
        t = en.get(it["id"], "")
        if t.strip():
            by_file.setdefault(it["file"], []).append(dict(it, en=t))

    total, skipped = 0, []
    for rel, group in by_file.items():
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        offs = _offsets(text)

        spans = []
        for it in group:
            start = max(0, offs[it["line"] - 1] - 200)
            idx = text.find(it["src"], start)
            if idx == -1:
                skipped.append(it["id"])
                continue
            spans.append((idx, idx + len(it["src"]), it["en"]))

        for a, b, new in sorted(spans, key=lambda s: -s[0]):
            text = text[:a] + new + text[b:]
            total += 1

        path.write_text(text, encoding="utf-8")
        print(f"{rel}: {len(spans)}")

    print(f"\nзаменено: {total}")
    if skipped:
        print(f"пропущено (не нашлось в файле): {len(skipped)}")
        for s in skipped:
            print(f"  {s}")


def check():
    if not OUT.exists():
        sys.exit("нет tools/comments.json")
    items = json.loads(OUT.read_text(encoding="utf-8"))
    en = json.loads(EN.read_text(encoding="utf-8")) if EN.exists() else {}
    cyr = [i for i in items if _cyr(i["src"])]
    left = [i for i in cyr if not en.get(i["id"], "").strip()]

    print(f"всего фрагментов: {len(items)}")
    print(f"с кириллицей:     {len(cyr)}")
    print(f"переведено:       {len(cyr) - len(left)}")
    if left:
        print(f"осталось:         {len(left)}")
        for i in left[:30]:
            print(f"  {i['id']}  {i['src'].splitlines()[0][:60]}")


def dump():
    """Печатает оригиналы по списку id — чтобы не пересылать весь файл.

        uv run python tools/comments.py dump src/api.py#66 src/toc.py#23
    """
    items = {i["id"]: i for i in json.loads(OUT.read_text(encoding="utf-8"))}
    for key in sys.argv[2:]:
        it = items.get(key)
        print(f"===== {key} =====")
        print(it["src"] if it else "(нет такого id)")
        print()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    {"extract": extract, "inject": inject, "check": check, "dump": dump}[cmd]()
