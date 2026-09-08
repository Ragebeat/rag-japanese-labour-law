"""Второй проход: проверка ответа против контекста отдельной моделью.

ЗАЧЕМ НЕ ПРОСТО faithfulness. Проверено на наших худших случаях — она их
не ловит:
  #7  «девяносто дней» взяты из 労組法15条, которая ЛЕЖАЛА в контексте
  #43 «десять дней» взяты из 39条1項, которая тоже лежала
Оба утверждения из контекста следуют. Согласуется с замерами: выдуманных
ссылок 0 из 50, distortion 3 из 20 — того, что ловит faithfulness, у нас
почти нет.

Поэтому три проверки, из них полезны две:
  適用 — применима ли процитированная статья к спрошенному
         (ловит подмену института: 16 случаев из 30 в абляции)
  欠落 — было ли в контексте положение, меняющее ответ, и не взято ли оно
         (ловит #43, #22, #19, #31)
  矛盾 — следует ли сказанное из контекста (классическая faithfulness)

СУДЬЯ — ДРУГОГО СЕМЕЙСТВА, чем генератор. Тот же gemma был бы слеп там же,
где генератор, и мы измерили бы согласие модели с самой собой.

Судья выдаёт контролируемые токены, а не прозу — приём из занятия 7,
сработавший дважды (маркер [N], вердикт 判定).

  uv run python src/verify.py --calibrate   # 20 вопросов против ручной разметки
  uv run python src/verify.py               # весь отчёт текущего режима
"""

import difflib
import json
import os
import pathlib
import re
import sys

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from generate import (DEFAULT_MODE, Reranked, build_context,  # noqa: E402
                      report_path, retrieve, _by_article)
from make_review import parse as parse_manual  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
URL = "http://localhost:11434/api/chat"
# Судья должен быть ДРУГОГО семейства, чем генератор (gemma):
# тот же был бы слеп там же, где генератор.
# Меняется переменной среды, чтобы сравнение судей было воспроизводимым,
# а не правкой константы между прогонами.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen3:8b")

# ⚠️ В примерах формата нет реальных статей — урок занятия 5.
SYSTEM = """あなたは法令回答の検証者です。回答そのものは書きません。

【入力】質問・提供された条文・回答・回答が引用した番号
【出力】次の3行だけ。説明も前置きも書かない。

適用: [1]=可 [2]=不可 …
欠落: なし
矛盾: なし

【判定基準】
適用 — 「回答が引用した番号」に挙がったものだけを判定する。
       引用されていない番号は一切評価しない。提供された条文に無関係な
       ものが含まれていても、回答が使っていなければ問題ではない。
       引用が正確でも、別の制度の規定で質問された事柄を規律していなければ
       「不可」とする。
欠落 — 提供された条文の中に、質問者にとって答えが変わる規定
       （適用範囲・例外・別の項）があり、回答がそれを使っていない場合、
       その箇所を「[N] 第◯条第◯項」の形で挙げる。なければ「なし」。
矛盾 — 回答の中の、提供された条文から導けない【文】をそのまま抜き出して
       書く。条文名や条番号だけを書いてはならない。なければ「なし」。

番号 [N] は提供された条文の範囲内のものだけを使う。存在しない番号を書かない。
条文に書かれていることだけで判定する。自分の知識で補わない。"""

APPLY_RE = re.compile(r"\[(\d+)\]\s*=\s*(可|不可)")
LINE_RE = {k: re.compile(rf"{k}[:：]\s*(.+)") for k in ("適用", "欠落", "矛盾")}


def judge(question, chunks, answer, client=None):
    own = client is None
    client = client or httpx.Client()
    try:
        # Список цитат даём явно. Без него судья оценивал все пять чанков
        # контекста, то есть переделывал работу retrieval вместо проверки
        # ответа — отсюда 13 блокировок из 20 в первой калибровке.
        cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)
                        if 1 <= int(m) <= len(chunks)})
        cited_str = " ".join(f"[{i}]" for i in cited) or "なし"
        user = (f"【質問】\n{question}\n\n"
                f"【条文】\n{build_context(chunks)}\n\n"
                f"【回答】\n{answer}\n\n"
                f"【回答が引用した番号】\n{cited_str}")
        r = client.post(URL, json={
            "model": JUDGE_MODEL,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}],
            "stream": False, "think": False,
            "options": {"temperature": 0},
        }, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    finally:
        if own:
            client.close()


def parse_verdict(text, n_chunks):
    """Разбирает три строки судьи. Ничего не усредняет."""
    def line(key):
        m = LINE_RE[key].search(text)
        return m.group(1).strip() if m else None

    applic = {int(i): v for i, v in APPLY_RE.findall(line("適用") or "")}
    bad = sorted(i for i, v in applic.items() if v == "不可" and 1 <= i <= n_chunks)

    def filled(key):
        v = line(key)
        if v is None:
            return None
        if v in ("なし", "無し", "None"):
            return ""
        return v

    missing, contra = filled("欠落"), filled("矛盾")

    # Судья ссылался на [4] при трёх чанках — чистим и здесь тоже.
    if missing:
        idx = [int(m) for m in re.findall(r"\[(\d+)\]", missing)]
        if idx and not all(1 <= i <= n_chunks for i in idx):
            missing = ""
    # На месте противоречия должно быть УТВЕРЖДЕНИЕ, а не название
    # статьи. Судья вписывал 「労働者災害補償保険法 第十三条」 — слот заполнен,
    # регекс доволен, содержания нет.
    if contra and re.fullmatch(r"[^\u3002]{0,40}第[^\u3002]{1,12}条[^\u3002]{0,20}", contra):
        contra = ""
    return {
        "raw": text,
        "parsed": bool(applic) or missing is not None,
        "inapplicable": bad,
        "missing": missing or "",
        "contradiction": contra or "",
        # блокируем: неприменимая ссылка ИЛИ пропущенное положение
        # (решение зафиксировано в занятии 8: omission блокирует наравне)
        "block": bool(bad) or bool(missing) or bool(contra),
    }


def run(rows, mode, client, r):
    out = []
    for i, row in enumerate(rows, 1):
        ch = retrieve(r, row["q"], mode)
        raw = judge(row["q"], ch, row["answer"], client)
        v = parse_verdict(raw, len(ch))
        out.append({"id": row["id"], "q": row["q"], **v})
        mark = "БЛОК" if v["block"] else ("  ?  " if not v["parsed"] else "    ")
        print(f"{i:>2}/{len(rows)} {mark} {row['q'][:26]}")
    return out


def cmd_calibrate():
    """Судья против ручной разметки. Согласие по каждому флагу отдельно.

    ⚠️ Метки в manual_review.md проставлены агентом (Claude), не человеком.
    Это согласие двух моделей, а не согласие с истиной. Записано явно,
    чтобы через месяц не принять одно за другое.

    Разметка делалась на режиме dedupe — судим те же ответы в том же
    режиме, иначе сравнивать нечего.
    """
    manual = parse_manual()
    if not manual:
        sys.exit("manual_review.md не заполнен")

    path = report_path("dedupe")
    if not path.exists():
        sys.exit(f"нужен {path.name} (переименуй старый gen_report.json)")
    by_id = {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}
    rows = [by_id[i] for i in sorted(manual) if i in by_id]

    r = Reranked()
    with httpx.Client() as cl:
        res = run(rows, "dedupe", cl, r)

    # Сопоставление флагов судьи и ручных меток
    pairs = {
        "欠落 ~ partial": (lambda v: bool(v["missing"]),
                          lambda m: m["substance"] == "partial"),
        "矛盾 ~ distortion": (lambda v: bool(v["contradiction"]),
                             lambda m: m["distortion"] == "yes"),
        "不可 ~ extra": (lambda v: bool(v["inapplicable"]),
                        lambda m: m["extra"] == "yes"),
    }
    print(f"\nсудья: {JUDGE_MODEL}   вопросов: {len(res)}")
    print(f"не разобрано: {sum(1 for v in res if not v['parsed'])}")
    for name, (jf, hf) in pairs.items():
        both = only_j = only_h = neither = 0
        for v in res:
            j, h = jf(v), hf(manual[v["id"]])
            both += j and h
            only_j += j and not h
            only_h += h and not j
            neither += not j and not h
        agree = (both + neither) / len(res)
        print(f"  {name:<20} согласие {agree:.2f}  "
              f"оба {both} · только судья {only_j} · только разметка {only_h}")

    out = ROOT / "data" / f"verify_calibrate_{JUDGE_MODEL.replace(':', '-')}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}")


def cmd_full(mode=DEFAULT_MODE):
    path = report_path(mode)
    if not path.exists():
        sys.exit(f"нужен {path.name} — сначала generate.py --eval")
    rows = json.loads(path.read_text(encoding="utf-8"))

    r = Reranked()
    with httpx.Client() as cl:
        res = run(rows, mode, cl, r)

    n = len(res)
    print(f"\nсудья: {JUDGE_MODEL}   режим: {mode}   вопросов: {n}")
    print(f"не разобрано:        {sum(1 for v in res if not v['parsed'])}")
    print(f"неприменимая ссылка: {sum(1 for v in res if v['inapplicable'])}")
    print(f"пропущено положение: {sum(1 for v in res if v['missing'])}")
    print(f"противоречие:        {sum(1 for v in res if v['contradiction'])}")
    blocked = sum(1 for v in res if v["block"])
    print(f"ЗАБЛОКИРОВАНО:       {blocked} = {blocked / n:.2f}")

    out = ROOT / "data" / f"verify_{mode}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}")


# ────────────── структурная проверка полноты ──────────────
# Четыре судьи (8B … 120B) провалили 欠落: лучший результат 3 из 4 при
# шести ложных, и качество не растёт с размером. Значит класс надо ловить
# не моделью, а структурой.
#
# Правило без единого настраиваемого числа:
#   параграф статьи считается пропущенным, если он НЕ использован в ответе,
#   а реранкер оценил его НЕ НИЖЕ, чем тот, который использован.
#
# Порога нет → подганять на двадцати точках нечего.

USED_MIN = 10   # символов общей подстроки — параграф считается использованным


def _used(paragraph, answer):
    """Взят ли параграф в ответ.

    Юридические ответы цитируют почти дословно (六箇月間継続勤務し全労働日の
    八割以上出勤), поэтому длинная общая подстрока — надёжный признак.
    """
    m = difflib.SequenceMatcher(None, paragraph, answer, autojunk=False)
    return m.find_longest_match(0, len(paragraph), 0, len(answer)).size >= USED_MIN


def structural_omission(question, chunks, answer, rr):
    """Статьи, где ответ взял один параграф, а соседний не хуже по скору.

    ⚠️ Имеет смысл ТОЛЬКО в режиме whole. На dedupe в контексте лежит один
    параграф, и флаг «не взял 2項» означал бы, что модели его не давали —
    то же заблуждение, что разобрано на #22.
    """
    arts = _by_article()
    cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)
                    if 1 <= int(m) <= len(chunks)})
    flags = []
    for i in cited:
        c = chunks[i - 1]
        parts = arts.get((c["law_id"], c["article_title"]), [])
        if len(parts) < 2:
            continue
        bodies = [p["text"].split("\n", 1)[-1] for p in parts]
        scores = rr.score(question, bodies)
        used = [j for j, b in enumerate(bodies) if _used(b, answer)]
        if not used:
            continue
        best_used = max(scores[j] for j in used)
        for j, s in enumerate(scores):
            if j not in used and s >= best_used:
                flags.append({
                    "marker": i,
                    "article": f"{c['law_title']} {c['article_title']}",
                    "paragraph": parts[j]["paragraph_num"],
                    "score": round(float(s), 3),
                    "best_used": round(float(best_used), 3),
                })
    return flags


def cmd_structural(mode=DEFAULT_MODE):
    path = report_path(mode)
    if not path.exists():
        sys.exit(f"нужен {path.name}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    rr = Reranked()

    res = []
    for i, row in enumerate(rows, 1):
        ch = retrieve(rr, row["q"], mode)
        f = structural_omission(row["q"], ch, row["answer"], rr)
        res.append({"id": row["id"], "q": row["q"], "omission": f})
        print(f"{i:>2}/{len(rows)} {'ФЛАГ' if f else '    '} {row['q'][:26]}")

    n = len(res)
    hit = [r for r in res if r["omission"]]
    print(f"\nрежим: {mode}   вопросов: {n}")
    print(f"помечено: {len(hit)} = {len(hit) / n:.2f}")

    manual = parse_manual()
    if manual:
        common = [r for r in res if r["id"] in manual]
        both = sum(1 for r in common
                   if bool(r["omission"]) and manual[r["id"]]["substance"] == "partial")
        only_s = sum(1 for r in common
                     if r["omission"] and manual[r["id"]]["substance"] != "partial")
        only_m = sum(1 for r in common
                     if not r["omission"] and manual[r["id"]]["substance"] == "partial")
        print(f"против разметки ({len(common)}): оба {both} · только структура {only_s}"
              f" · только разметка {only_m}")

    out = ROOT / "data" / f"omission_{mode}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}")


if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        cmd_calibrate()
    elif "--structural" in sys.argv:
        cmd_structural()
    else:
        cmd_full()
