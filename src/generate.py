"""Генерация ответа по найденным статьям, с цитированием.

Ключевое решение: модель отвечает ТОЛЬКО по поданным чанкам, а ссылку
даёт маркером [N] — номером чанка, который мы сами проставили в контекст.
Если ответа в контексте нет — обязана отказаться.

Почему маркер, а не номер статьи: номер статьи приходится выковыривать
регексом из свободного текста, и модель пишет его как хочет (第39条 /
第三十九条 / 第３９条). Маркер [N] мы контролируем сами — сопоставление
ответ→чанк становится точным, а не эвристикой. Номер статьи модель тоже
пишет (для человека) и он проверяется вторым, независимым способом.

  python src/generate.py "有給って入社してすぐもらえるの？"
  python src/generate.py --sample     # 5 вопросов сырьём, смотреть глазами
  python src/generate.py --eval       # прогон по golden
"""

import collections
import json
import os
import pathlib
import re
import sys

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rerank import Reranked  # noqa: E402
from jnum import art_key, fmt as _fmt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden.jsonl"
CHUNKS = ROOT / "data" / "chunks.jsonl"

# Режим сборки контекста по умолчанию. whole выбран по замерам:
# recall@5 такой же, как у dedupe (0.88), over-refusal 2 → 0, обрезанных
# статей нет. Не хуже ни по одному показателю и лучше по двум.
DEFAULT_MODE = "whole"


def report_path(mode, kind=""):
    """Отчёты именуются по режиму сборки — иначе через месяц не вспомнить,
    какой конфигурацией получены цифры. Цена путаницы уже известна:
    всё занятие 8 было смерено на dedupe и потребовало перемера.
    """
    return ROOT / "data" / f"gen_report{kind}_{mode}.json"

MODEL = os.environ.get("GEN_MODEL", "gemma4:e4b")
URL = "http://localhost:11434/api/chat"
TOP_K = 5

REFUSAL = "該当する規定が見つかりません"

# ⚠️ В примерах формата НЕТ реальных статей и терминов из golden.
# Урок занятия 5: пример в промпте становится шаблоном.
#
# Строка 「判定:」 добавлена по итогам первого прогона: отказ в прозе
# модель выражала десятью разными формулировками и ни разу — чисто.
# Тот же приём, что с маркером [N]: пусть модель выдаёт токен, который
# мы контролируем, вместо выковыривания смысла из свободного текста.
SYSTEM = f"""あなたは日本の労働法令の検索結果を要約するアシスタントです。

【出力形式】
1行目に必ず「判定: 回答可能」または「判定: 回答不可」とだけ書く。
2行目以降に本文を書く。

【厳守】
1. 提供された条文だけを根拠にする。自分の知識で補わない
2. 根拠にした条文を必ず [1] のような番号で示す。番号の直後に法令名と
   条番号も書く。形式:「[N] ◯◯法 第N条によると…」
3. 質問に直接答える条文が提供されていない場合は「判定: 回答不可」とし、
   本文には「{REFUSAL}」とだけ書く。
   関連しそうな条文を列挙したり、前置きを書いたりしない
4. 数字・日数・金額は条文にある通りに引用する。推測しない
5. 本文は3〜5文で簡潔に"""


# ─────────────────────────── нормализация номеров ───────────────────────────
# Сама нормализация номеров живёт в jnum.py: тем же разбором
# пользуется expand_golden.py, а две копии — верный способ
# починить только одну из них.

MARK_RE = re.compile(r"\[(\d+)\]")
VERDICT_RE = re.compile(r"判定[:：]\s*(回答可能|回答不可)")

# Формулировки уклонения, собранные с первого прогона глазами.
# Нужны НЕ для детекта отказа (для этого теперь есть 判定), а чтобы
# посчитать перестраховку: сказала 回答可能 и тут же оговорилась.
HEDGE = ("見つかりません", "できませんでした", "確認できません",
         "判断できません", "記述はありません", "直接的な規定は",
         "ことはできません")


def build_context(chunks):
    parts = []
    for i, c in enumerate(chunks, 1):
        head = f"{c['law_title']} {c['article_title']}{c['article_caption'] or ''}"
        body = c["text"].split("\n", 1)[-1]
        parts.append(f"[{i}] {head}\n{body}")
    return "\n\n".join(parts)


def generate(query, chunks, client=None):
    own = client is None
    client = client or httpx.Client()
    try:
        r = client.post(URL, json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": f"【条文】\n{build_context(chunks)}\n\n【質問】\n{query}"},
            ],
            "stream": False, "think": False,
            "options": {"temperature": 0},
        }, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    finally:
        if own:
            client.close()


# ────────────────────────────── разбор ответа ───────────────────────────────

def context_arts(chunks):
    """Два множества статей, встречающихся в контексте.

    source    — статьи, которые мы реально подали (article_title)
    mentioned — плюс всё, что упомянуто в заголовках и телах чанков

    Разница не формальная. Японское право сплошь построено на перекрёстных
    отсылках (「法第二十五条第一項の…」, 「労働基準法第八十九条及び第九十条の
    定めるところによる」). Названная так статья — не выдумка, но и не
    источник: её текста у модели не было, проверить содержание нечем.

    Первый прогон считал только source — и дал 6 ложных галлюцинаций
    из 6 найденных. Регекс был проверен, эталонное множество — нет.
    """
    source, extra = set(), set()
    for c in chunks:
        source |= art_key(c["article_title"])
        extra |= art_key(c["article_caption"] or "")
        extra |= art_key(c["text"])
    return source, source | extra


def inspect(answer, chunks):
    """Разбирает ответ на проверяемые факты. Ничего не усредняет.

    ⚠️ Это НЕ faithfulness. Проверяется только источник: сослалась
    ли модель на то, что ей дали. Случай «номер верный, срок переврала»
    этим не ловится, и случай «нужная статья была в контексте, но модель
    её не взяла» — тоже. Для них нужна разметка человеком.
    """
    lines = [l for l in answer.splitlines() if l.strip()]
    head = lines[0] if lines else ""

    m = VERDICT_RE.search(head)
    misplaced = False
    if not m:
        m = VERDICT_RE.search(answer)
        misplaced = bool(m)          # вердикт есть, но не на первой строке
    verdict = {"回答可能": "ok", "回答不可": "no"}.get(m.group(1)) if m else None

    body = VERDICT_RE.sub("", answer).strip()

    marks = {int(x) for x in MARK_RE.findall(body)}
    bad_marks = {x for x in marks if not 1 <= x <= len(chunks)}

    cited = art_key(body)
    source, mentioned = context_arts(chunks)
    bad_arts = cited - mentioned               # выдумка: в контексте нет нигде
    transitive = (cited & mentioned) - source  # отсылка из тела чанка

    return {
        "verdict": verdict,
        "verdict_misplaced": misplaced,
        # 回答不可 и больше ничего — чего не было ни разу в первом прогоне
        "clean_refusal": verdict == "no" and not marks and not cited,
        "partial_refusal": verdict == "no" and bool(marks or cited),
        # сказала «могу ответить» и тут же оговорилась — перестраховка
        "hedged": any(h in body for h in HEDGE),
        "marks": sorted(marks),
        "bad_marks": sorted(bad_marks),
        "cited": _fmt(cited),
        "transitive": _fmt(transitive),
        "bad_arts": _fmt(bad_arts),
        # сторож от регресса, а не показатель качества: ожидается 1.0
        "citation_validity": None if not cited
        else (len(cited) - len(bad_arts)) / len(cited),
        "len": len(body),
    }


# ──────────────────────────────────── cli ────────────────────────────────────

KW = dict(mode="vector", expand_query=True, narrow=True, combo=True,
          rerank_on="expanded")

POOL_MULT = 4     # сколько кандидатов брать до схлопывания: TOP_K * POOL_MULT
ARTICLE_CAP = 3000  # символов; длиннее — статью целиком не собираем


def dedupe_by_article(chunks, k):
    """Один чанк на статью: лучший параграф, остальные отбрасываем.

    Список уже отсортирован реранкером, поэтому первое вхождение и есть
    лучшее — достаточно запомнить виденное.

    Ключ — (law_id, article_title), а не одно название: 第五条 есть почти
    в каждом законе корпуса, по одному номеру схлопнутся разные статьи.

    ⚠️ Это меняет retrieval, а не только показ. В топ-k попадает больше
    РАЗНЫХ статей → recall@k и MRR сдвинутся. Мерить обе конфигурации.
    """
    seen, out = set(), []
    for c in chunks:
        key = (c["law_id"], c["article_title"])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) == k:
            break
    return out


_ARTS = None


def _by_article():
    """(law_id, article_title) -> все чанки статьи в порядке параграфов."""
    global _ARTS
    if _ARTS is None:
        _ARTS = collections.defaultdict(list)
        for line in CHUNKS.open(encoding="utf-8"):
            c = json.loads(line)
            _ARTS[(c["law_id"], c["article_title"])].append(c)
        for v in _ARTS.values():
            v.sort(key=lambda c: (int(c["paragraph_num"] or 0), c["chunk_id"]))
    return _ARTS


def whole_article(c, arts):
    """Заменяет один параграф статьёй целиком.

    Зачем: дедуп оставляет лучший параграф по мнению реранкера, а ответ на
    вопрос может лежать в соседнем пункте той же статьи. На #22 реранкер
    выбрал 13条 1項 (12 месяцев), а для большинства спрашивающих верно 2項
    (6 месяцев) — модель его не видела и не могла взять.

    ⚠️ Уровень 3 чанкинга режет с overlap — соседние куски частично
    перекрываются. Точные дубли выкидываем, частичные останутся. Это 1.6%
    корпуса, но если всплывёт в ответах — чинить здесь.
    """
    parts = arts.get((c["law_id"], c["article_title"]), [c])
    total = sum(p["n_chars"] for p in parts)
    if len(parts) == 1 or total > ARTICLE_CAP:
        return c
    seen, bodies = set(), []
    for p in parts:
        body = p["text"].split("\n", 1)[-1]
        if body in seen:
            continue
        seen.add(body)
        bodies.append(body)
    out = dict(c)
    out["text"] = c["text"].split("\n", 1)[0] + "\n" + "\n".join(bodies)
    out["n_chars"] = total
    out["n_paragraphs"] = len(bodies)
    return out


def retrieve(r, query, mode="dedupe", k=TOP_K, exclude=None):
    """Сборка контекста. Три режима:

    raw    — топ-k параграфов как есть (базовая линия; дубли едят слоты)
    dedupe — один лучший параграф на статью (теряет соседние пункты)
    whole  — k разных статей, каждая целиком

    Реранкеру всегда отдаём pool = k * POOL_MULT и берём столько же: он всё
    равно скорит весь пул, так что схлопывание и отсев бесплатны по времени.

    exclude — множество (law_id, article_title), выбрасываемых из пула до
    отбора топ-k. Нужно для абляции.
    """
    pool = k * POOL_MULT
    cands = r.search(query, k=pool, core_only=True, pool=pool, **KW)
    if exclude:
        cands = [c for c in cands
                 if (c["law_id"], c["article_title"]) not in exclude]
    if mode == "raw":
        return cands[:k]
    picked = dedupe_by_article(cands, k)
    if mode == "whole":
        arts = _by_article()
        picked = [whole_article(c, arts) for c in picked]
    return picked


def load_golden():
    return [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]


def cmd_sample(n=5, mode=DEFAULT_MODE):
    """Сырые ответы глазами. Цифр здесь нет и не должно быть."""
    r = Reranked()
    with httpx.Client() as cl:
        for g in load_golden()[:n]:
            ch = retrieve(r, g["q"], mode)
            ans = generate(g["q"], ch, cl)
            info = inspect(ans, ch)

            print("=" * 72)
            print(f"#{g['id']}  {g['q']}")
            print("— контекст —")
            for i, c in enumerate(ch, 1):
                print(f"  [{i}] {c['law_title'][:22]} {c['article_title']}"
                      f"{c['article_caption'] or ''}")
            print("— ответ —")
            print(ans)
            print("— разбор —")
            v = {"ok": "回答可能", "no": "回答不可",
                 None: "НЕТ ВЕРДИКТА"}[info["verdict"]]
            print(f"  判定: {v}"
                  f"{' (не на первой строке)' if info['verdict_misplaced'] else ''}")
            print(f"  маркеры {info['marks']}  статьи {info['cited']}")
            if info["transitive"]:
                print(f"  ↳ транзитивные отсылки: {info['transitive']}")
            if info["bad_marks"]:
                print(f"  ❌ маркер вне диапазона: {info['bad_marks']}")
            if info["bad_arts"]:
                print(f"  ❌ статья не из контекста: {info['bad_arts']}")
            if info["partial_refusal"]:
                print("  ⚠ отказ неполный: 回答不可, а ссылки есть")
            if info["hedged"] and info["verdict"] == "ok":
                print("  ⚠ перестраховка: 回答可能, но в тексте оговорка")
            print()


def cmd_eval(mode=DEFAULT_MODE):
    golden = load_golden()
    r = Reranked()
    rows = []
    print(f"сборка контекста: {mode}\n")
    with httpx.Client() as cl:
        for i, g in enumerate(golden, 1):
            ch = retrieve(r, g["q"], mode)
            ans = generate(g["q"], ch, cl)
            info = inspect(ans, ch)

            # Попал ли retrieval: есть ли хотя бы одна золотая статья в контексте.
            # Это разделяет два разных вопроса, которые общая цифра смешивает:
            # «нашёл ли поиск» и «воспользовалась ли модель тем, что нашлось».
            gold = {tuple(x) for x in g["relevant"]}
            got = {(c["law_id"], c["article_title"]) for c in ch}
            hit = bool(gold & got)

            rows.append({"id": g["id"], "q": g["q"], "answer": ans,
                         "hit": hit,
                         "n_context": len(ch),
                         "context": [f"{c['law_id']} {c['article_title']}" for c in ch],
                         **info})
            flag = {"no": "НЕТ", "ok": "   ", None: " ? "}[info["verdict"]]
            if info["bad_arts"] or info["bad_marks"]:
                flag = flag[:2] + "❌"
            print(f"{i:>2}/{len(golden)} {'hit ' if hit else 'MISS'} {flag} {g['q'][:26]}")

    n = len(rows)
    novrd = [r_ for r_ in rows if r_["verdict"] is None]
    refus = [r_ for r_ in rows if r_["verdict"] == "no"]
    clean = [r_ for r_ in refus if r_["clean_refusal"]]
    hedge = [r_ for r_ in rows if r_["verdict"] == "ok" and r_["hedged"]]
    no_mark = [r_ for r_ in rows if r_["verdict"] == "ok" and not r_["marks"]]
    bad = [r_ for r_ in rows if r_["bad_arts"] or r_["bad_marks"]]
    trans = [r_ for r_ in rows if r_["transitive"]]
    with_cit = [r_ for r_ in rows if r_["citation_validity"] is not None]

    print(f"\nвсего:                 {n}")
    print(f"без вердикта:         {len(novrd):>3}   ← контракт формата сломан")
    print(f"回答不可 (отказ):       {len(refus):>3}   из них чистых {len(clean)}")
    print(f"перестраховка:         {len(hedge):>3}   回答可能 + оговорка в тексте")
    print(f"ответов без маркера:   {len(no_mark):>3}")
    print(f"транзитивных отсылок: {len(trans):>3}   статья из тела чанка, не источник")
    print(f"выдуманных ссылок:    {len(bad):>3}")
    if with_cit:
        print(f"citation_validity:    "
              f"{sum(r_['citation_validity'] for r_ in with_cit) / len(with_cit):.3f}"
              f"  (сторож, ожидается 1.000)")

    # ─── retrieval против генерации (§4.5) ───
    # Общая частота отказов ничего не говорит: отказ при промахе поиска —
    # правильное поведение, отказ при попадании — потеря. Две ошибки,
    # а не одна, и оптимизировать их надо в разные стороны.
    def q(hit, verdict):
        return [r_ for r_ in rows if r_["hit"] is hit and r_["verdict"] == verdict]

    hits = [r_ for r_ in rows if r_["hit"]]
    miss = [r_ for r_ in rows if not r_["hit"]]
    over = q(True, "no")
    blind = q(False, "ok")

    print(f"\nretrieval recall@{TOP_K}: {len(hits)}/{n} = {len(hits) / n:.2f}")
    print("                     回答可能   回答不可")
    print(f"  A: статья найдена   {len(q(True, 'ok')):>5}   {len(over):>5}  ← over-refusal")
    print(f"  B: статьи нет      {len(blind):>5}   {len(q(False, 'no')):>5}")
    print(f"                     ↑ ответ мимо источника")
    if miss:
        print(f"  отказ при промахе:  {len(q(False, 'no')) / len(miss):.2f}   (выше — лучше)")
    if hits:
        print(f"  отказ при попадании:{len(over) / len(hits):.2f}   (ниже — лучше)")
    for label, group in (("over-refusal", over), ("ответ без источника", blind)):
        if group:
            print(f"  {label}: " + ", ".join(f"#{r_['id']}" for r_ in group))

    out = report_path(mode)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}")


def cmd_ablate(mode=DEFAULT_MODE):
    """Абляция: у вопросов с попаданием убираем золотую статью из контекста.

    Зачем: в обычном прогоне промахов retrieval всего 6, и отказ 0.17
    против 0.05 на такой выборке неотличим от случайности. Здесь получаем
    44 контролируемых промаха с реалистичными соседними дистракторами.

    Парный дизайн: те же вопросы, разница ровно в наличии источника.
    Объяснение «просто вопросы разной трудности» исключено по построению.

    Исходный прогон (--eval) должен быть сделан: из него берём, какие
    вопросы были hit, и вердикт для парного сравнения.
    """
    if not report_path(mode).exists():
        sys.exit(f"сначала --eval в режиме {mode}: нужен "
                 f"{report_path(mode).name} для парного сравнения")
    base = {b["id"]: b for b in
            json.loads(report_path(mode).read_text(encoding="utf-8"))}
    targets = [g for g in load_golden() if base.get(g["id"], {}).get("hit")]

    r = Reranked()
    rows = []
    print(f"абляция ({mode}): убираем золотую статью у {len(targets)} вопросов\n")
    with httpx.Client() as cl:
        for i, g in enumerate(targets, 1):
            gold = {tuple(x) for x in g["relevant"]}
            ch = retrieve(r, g["q"], mode, exclude=gold)
            ans = generate(g["q"], ch, cl)
            info = inspect(ans, ch)

            # Санитарная проверка: золотой статьи точно нет в контексте.
            # Без неё весь эксперимент может молча измерять не то.
            assert not (gold & {(c["law_id"], c["article_title"]) for c in ch})

            rows.append({"id": g["id"], "q": g["q"], "answer": ans,
                         "verdict_base": base[g["id"]]["verdict"],
                         "n_context": len(ch),
                         "context": [f"{c['law_id']} {c['article_title']}" for c in ch],
                         **info})
            mark = {"no": "ОТКАЗ", "ok": "ответ", None: "  ?  "}[info["verdict"]]
            print(f"{i:>2}/{len(targets)} {mark} {g['q'][:26]}")

    n = len(rows)
    refused = [x for x in rows if x["verdict"] == "no"]
    answered = [x for x in rows if x["verdict"] == "ok"]
    substituted = [x for x in answered if x["marks"]]

    print(f"\nаблировано вопросов: {n}")
    print(f"отказалась:          {len(refused):>3}  = {len(refused) / n:.2f}  (выше — лучше)")
    print(f"ответила по соседней: {len(substituted):>3}  = {len(substituted) / n:.2f}")

    # Парная таблица: тот же вопрос с источником и без него.
    # Если модель чувствует наличие источника, вес должен уйти
    # в клетку «было 回答可能 → стало 回答不可».
    def cell(b, a):
        return sum(1 for x in rows if x["verdict_base"] == b and x["verdict"] == a)

    print("\nпарное сравнение (строки — с источником, столбцы — без)")
    print("              回答可能   回答不可")
    print(f"  回答可能      {cell('ok', 'ok'):>5}   {cell('ok', 'no'):>5}  ← заметила пропажу")
    print(f"  回答不可      {cell('no', 'ok'):>5}   {cell('no', 'no'):>5}")
    print(f"              ↑ не заметила")

    out = report_path(mode, "_ablate")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}")


def main():
    args = sys.argv[1:]
    mode = ("raw" if "--nodedupe" in args else
            "dedupe" if "--dedupe" in args else
            "whole" if "--whole" in args else DEFAULT_MODE)
    if "--ablate" in args:
        return cmd_ablate(mode)
    if "--sample" in args:
        return cmd_sample(mode=mode)
    if "--eval" in args:
        return cmd_eval(mode=mode)

    query = " ".join(a for a in args if not a.startswith("--")) or \
        "有給って入社してすぐもらえるの？"
    r = Reranked()
    ch = retrieve(r, query, mode)
    print(f"\n{generate(query, ch)}\n")
    print("— источники —")
    for i, c in enumerate(ch, 1):
        print(f"[{i}] {c['law_title'][:24]} {c['article_title']}"
              f"{c['article_caption'] or ''}")


if __name__ == "__main__":
    main()
