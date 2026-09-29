"""Generating an answer from the retrieved articles, with citations.

The key decision: the model answers ONLY from the supplied chunks, and cites
by an [N] marker — the chunk number we put into the context ourselves. If the
answer is not in the context, it must refuse.

Why a marker rather than an article number: an article number has to be dug
out of free text with a regex, and the model writes it however it likes
(第39条 / 第三十九条 / 第３９条). The [N] marker is ours, so mapping answer to
chunk becomes exact instead of heuristic. The model also writes the article
number for human readers, and that is verified a second, independent way.

  python src/generate.py "有給って入社してすぐもらえるの？"
  python src/generate.py --sample     # 5 questions raw, to read by eye
  python src/generate.py --eval       # run over the golden set
  python src/generate.py --eval --base     # the phase 4 prompt, without verbatim quotes
"""

import collections
import json
import os
import pathlib
import re
import sys
import unicodedata

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rerank import Reranked  # noqa: E402
from jnum import art_key, fmt as _fmt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden.jsonl"
CHUNKS = ROOT / "data" / "chunks.jsonl"

# Default context assembly mode. `whole` was chosen on measurements:
# recall@5 equals dedupe (0.88), over-refusal 2 -> 0, and no truncated
# articles. Not worse on any metric and better on two.
DEFAULT_MODE = "whole"


def report_path(mode, kind=""):
    """Reports are named after the assembly mode — otherwise, a month later,
    there is no telling which configuration produced which numbers. The cost
    of that confusion is known: all of session 8 was measured on dedupe and
    had to be re-measured.
    """
    return ROOT / "data" / f"gen_report{kind}_{mode}.json"

MODEL = os.environ.get("GEN_MODEL", "gemma4:e4b")
URL = "http://localhost:11434/api/chat"
TOP_K = 5

REFUSAL = "該当する規定が見つかりません"

# ⚠️ The format examples contain NO real articles or terms from the golden set.
# Lesson from session 5: an example in the prompt becomes a template.
#
# The 「判定:」 line was added after the first run: in prose the model
# expressed refusal in ten different phrasings and never once cleanly.
# Same technique as the [N] marker: have the model emit a token we control
# instead of digging meaning out of free text.
SYSTEM_BASE = f"""あなたは日本の労働法令の検索結果を要約するアシスタントです。

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

# Phase 6: the verbatim-quote variant. Five answers in phase 5 had the right
# article and a wrong detail; the one class this makes CHECKABLE is a quote
# that is not really a quote (two exceptions spliced into one, text put under
# another article's number). The check sees only what is inside 「」:
# paragraph numbers, paraphrase and omissions stay invisible to it.
# Rule 7 is a separate, cheaper net: a number outside quotes is unverified.
SYSTEM_QUOTES = SYSTEM_BASE + """
6. 答えの核心（数字・期間・要件・例外）は、条文から一字一句そのまま「」で引用し、
   直後にその条文の番号を付ける。形式:「引用文」[N]
   「」の中は言い換えない・要約しない・別々の文をつなぎ合わせない。
   別の文を引用するときは、別の「」にする
7. 数字・日数・期間は「」の引用の中にだけ書く"""

# The default since phase 6 (29.09), measured on golden against the base
# prompt: claim coverage 0.09 -> 0.68, answers with numbers outside quotes
# 22 -> 11, over-refusal 0 -> 0, refusal on retrieval miss 0.17 -> 0.17 (the
# same five blind answers). One suspected regression: #26 dropped the key
# article (労契法 第十条) — one case, not separable from run-to-run noise.
# SYSTEM_BASE stays: the phase 4 ablation is paired with a base-prompt report.
SYSTEM = SYSTEM_QUOTES

# Generation options — ONE copy, imported by api.py. The first num_ctx fix
# (phase 6) patched generate() and missed api.py's two own requests: three
# copies of the same dict, one of them updated.
#
# num_ctx is explicit on purpose. check_ctx.py: prompts run up to ~7 000
# tokens and nothing was truncated — but only because the Ollama app on this
# machine is set to 32k. That setting lives outside the repo; on a stock
# install below 24 GiB VRAM the default is 4096, and longer prompts would
# lose their start silently. Same value as agent.py: a different window
# reloads the model on every switch between the two.
GEN_OPTIONS = {"temperature": 0, "num_ctx": 32_768}


# ────────────────────────── article number parsing ──────────────────────────
# The normalisation itself lives in jnum.py: expand_golden.py uses the same
# parsing, and two copies are a reliable way to fix only one of them.
#

MARK_RE = re.compile(r"\[(\d+)\]")
VERDICT_RE = re.compile(r"判定[:：]\s*(回答可能|回答不可)")

# Hedging phrases collected by eye from the first run.
# NOT used to detect refusal (判定 does that now), but to count
# over-hedging: said 回答可能 and immediately qualified it.
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


def generate(query, chunks, client=None, system=SYSTEM):
    own = client is None
    client = client or httpx.Client()
    try:
        r = client.post(URL, json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",
                 "content": f"【条文】\n{build_context(chunks)}\n\n【質問】\n{query}"},
            ],
            "stream": False, "think": False,
            "options": GEN_OPTIONS,
        }, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    finally:
        if own:
            client.close()


# ───────────────────────────── answer analysis ──────────────────────────────

def context_arts(chunks):
    """Two sets of articles occurring in the context.

    source    — articles we actually supplied (article_title)
    mentioned — plus everything named in chunk captions and bodies

    The distinction is not formal. Japanese law is built on cross-references
    throughout (「法第二十五条第一項の…」, 「労働基準法第八十九条及び第九十条の
    定めるところによる」). An article named that way is not a fabrication, but
    it is not a source either: the model never saw its text, so there is
    nothing to verify the content against.

    The first run counted `source` only — and produced 6 false hallucinations
    out of 6 found. The regex had been verified; the reference set had not.
    """
    source, extra = set(), set()
    for c in chunks:
        source |= art_key(c["article_title"])
        extra |= art_key(c["article_caption"] or "")
        extra |= art_key(c["text"])
    return source, source | extra


def inspect(answer, chunks):
    """Breaks the answer down into checkable facts. Averages nothing.

    ⚠️ This is NOT faithfulness. It checks the source only: whether the model
    cited what it was given. The case of "right number, wrong duration" is not
    caught here, and neither is "the right article was in the context but the
    model did not use it". Those need human annotation.
    """
    lines = [l for l in answer.splitlines() if l.strip()]
    head = lines[0] if lines else ""

    m = VERDICT_RE.search(head)
    misplaced = False
    if not m:
        m = VERDICT_RE.search(answer)
        misplaced = bool(m)          # the verdict exists, but not on the first line
    verdict = {"回答可能": "ok", "回答不可": "no"}.get(m.group(1)) if m else None

    body = VERDICT_RE.sub("", answer).strip()

    marks = {int(x) for x in MARK_RE.findall(body)}
    bad_marks = {x for x in marks if not 1 <= x <= len(chunks)}

    cited = art_key(body)
    source, mentioned = context_arts(chunks)
    bad_arts = cited - mentioned               # fabrication: nowhere in the context
    transitive = (cited & mentioned) - source  # cross-reference from a chunk body

    return {
        "verdict": verdict,
        "verdict_misplaced": misplaced,
        # 回答不可 and nothing else — never seen once in the first run
        "clean_refusal": verdict == "no" and not marks and not cited,
        "partial_refusal": verdict == "no" and bool(marks or cited),
        # said "I can answer" and hedged in the same breath — over-hedging
        "hedged": any(h in body for h in HEDGE),
        "marks": sorted(marks),
        "bad_marks": sorted(bad_marks),
        "cited": _fmt(cited),
        "transitive": _fmt(transitive),
        "bad_arts": _fmt(bad_arts),
        # a regression guard, not a quality indicator: 1.0 is expected
        "citation_validity": None if not cited
        else (len(cited) - len(bad_arts)) / len(cited),
        **check_quotes(body, chunks),
        "len": len(body),
    }


# ── verbatim quote check (phase 6) ──
# Minimum 4 characters: a shorter 「」 is a term the answer brackets for
# emphasis (「労働者」), not a quote.
# ANY_QUOTE_RE only cuts quotes out when looking for numbers. Quotes
# themselves come from _quotes(), which tracks nesting: statute text quotes
# its own terms, and a long quote in #23 contained 「支給日数」 inside.
ANY_QUOTE_RE = re.compile(r"「([^「」]{4,})」")
MIN_QUOTE = 4
ELLIPSIS_RE = re.compile(r"…+|\.{3,}|・{3,}")
# A number with a unit. 条/項/号 are deliberately absent: article references
# legitimately live outside quotes.
NUM_RE = re.compile(r"[0-9〇一二三四五六七八九十百千]+"
                    r"(?:日|箇月|か月|ヶ月|カ月|ケ月|時間|年|週|割|円|%|人)")


def _norm(s):
    """Width and whitespace only. Numerals are NOT unified: 六箇月 against
    6か月 is exactly the kind of drift a verbatim check exists to catch."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


def _verbatim(quote, text):
    """quote occurs in text; fragments split by an ellipsis must occur in order."""
    pos = 0
    for frag in ELLIPSIS_RE.split(quote):
        frag = _norm(frag)
        if not frag:
            continue
        i = text.find(frag, pos)
        if i < 0:
            return False
        pos = i + len(frag)
    return True


def _sentences(body):
    """Split on 。 and newlines, but never inside 「」."""
    out, cur, depth = [], [], 0
    for ch in body:
        if ch == "「":
            depth += 1
        elif ch == "」" and depth:
            depth -= 1
        if ch in "。\n" and depth == 0:
            if "".join(cur).strip():
                out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur))
    return out


def _quotes(s):
    """Top-level 「…」 spans; nested brackets stay inside their parent."""
    qs, depth, start = [], 0, 0
    for i, ch in enumerate(s):
        if ch == "「":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == "」" and depth:
            depth -= 1
            if depth == 0:
                qs.append(s[start:i])
    return qs


def check_quotes(body, chunks):
    """A quote must be a substring of a chunk cited IN ITS SENTENCE, as the
    model saw that chunk.

    First version demanded 「…」[N] back to back and missed 71 of 102
    quotes: the model writes natural Japanese — 「…」を支払わなければ
    なりません[1] — quote mid-sentence, marker at the end.

    Checked against the sentence's own [N] chunks, not the whole context:
    otherwise text of one article quoted under another's number would pass
    (phase 5, case 5). found_in says where the text really is — non-empty
    means "real text, wrong number", empty means "not a real quote" (case 4).
    """
    texts = [_norm(c["text"].split("\n", 1)[-1]) for c in chunks]

    def where(q):
        return [i + 1 for i, t in enumerate(texts) if _verbatim(q, t)]

    # Two coverages, because verbatim-ness alone rewards NOT quoting. The
    # first test answer quoted the undisputed sentence and paraphrased the
    # exceptions — the phase 5 splice again, outside the quotes — while
    # reporting "0 non-verbatim". And sentence coverage alone rewards a
    # token quote: 「妊娠したこと」 in brackets, the claim itself paraphrased.
    # So: claims_quoted (sentences with a verbatim quote) AND quoted_chars
    # (share of claim text that is verbatim).
    bad, loose = [], []
    n_quotes = claims = claims_quoted = claim_chars = quoted_chars = 0
    for s in _sentences(body):
        src = [int(x) for x in MARK_RE.findall(s) if 1 <= int(x) <= len(texts)]
        ok_chars = 0
        for q in _quotes(s):
            if len(_norm(q)) < MIN_QUOTE:
                continue
            if not src:
                loose.append({"quote": q, "found_in": where(q)})
                continue
            n_quotes += 1
            if any(_verbatim(q, texts[n - 1]) for n in src):
                ok_chars += len(_norm(q))
            else:
                bad.append({"quote": q, "n": src, "found_in": where(q)})
        if src:
            claims += 1
            claims_quoted += ok_chars > 0
            claim_chars += len(_norm(MARK_RE.sub("", s)))
            quoted_chars += ok_chars

    outside = unicodedata.normalize("NFKC", ANY_QUOTE_RE.sub("", body))
    return {"quotes": n_quotes, "quote_bad": bad, "quote_loose": loose,
            "num_unquoted": NUM_RE.findall(outside),
            "claims": claims, "claims_quoted": claims_quoted,
            "claim_chars": claim_chars, "quoted_chars": quoted_chars}


# ──────────────────────────────────── cli ────────────────────────────────────

KW = dict(mode="vector", expand_query=True, narrow=True, combo=True,
          rerank_on="expanded")

POOL_MULT = 4     # how many candidates to take before collapsing: TOP_K * POOL_MULT
ARTICLE_CAP = 3000  # characters; beyond this we do not assemble the whole article


def dedupe_by_article(chunks, k):
    """One chunk per article: keep the best paragraph, drop the rest.

    The list is already sorted by the reranker, so the first occurrence is the
    best one — remembering what has been seen is enough.

    The key is (law_id, article_title), not the title alone: 第五条 exists in
    almost every statute in the corpus, so one number would collapse different
    articles together.

    ⚠️ This changes retrieval, not just presentation. More DISTINCT articles
    reach the top k, so recall@k and MRR shift. Measure both configurations.
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
    """(law_id, article_title) -> every chunk of the article, in paragraph order."""
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
    """Replaces a single paragraph with the whole article.

    Why: dedupe keeps the paragraph the reranker rates best, but the answer to
    the question may sit in a neighbouring paragraph of the same article. On
    #22 the reranker picked 13条 1項 (twelve months), while for most people
    asking, 2項 (six months) is the correct one — the model never saw it and
    could not have used it.

    ⚠️ Level 3 of the chunker splits with overlap, so adjacent pieces partly
    repeat. Exact duplicates are dropped, partial ones remain. That is 1.6% of
    the corpus, but if it ever surfaces in answers, fix it here.
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
    """Context assembly. Three modes:

    raw    — top-k paragraphs as they come (baseline; duplicates eat slots)
    dedupe — one best paragraph per article (loses neighbouring paragraphs)
    whole  — k distinct articles, each in full

    The reranker always gets pool = k * POOL_MULT and we take the same number:
    it scores the entire pool regardless, so collapsing and filtering are free
    in time terms.

    exclude — a set of (law_id, article_title) removed from the pool before
    the top-k selection. Needed for the ablation.
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


def cmd_sample(n=5, mode=DEFAULT_MODE, quotes=True):
    """Raw answers, to be read by eye. No numbers here, and there should not be."""
    r = Reranked()
    system = SYSTEM_QUOTES if quotes else SYSTEM_BASE
    with httpx.Client() as cl:
        for g in load_golden()[:n]:
            ch = retrieve(r, g["q"], mode)
            ans = generate(g["q"], ch, cl, system=system)
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
            print_quotes(info)
            print()


def print_quotes(info):
    if info["claims"]:
        print(f"  утверждений с [N]: {info['claims']}, с дословной цитатой: "
              f"{info['claims_quoted']}  ← остальные непроверяемы")
    if info["quotes"] or info["quote_loose"]:
        print(f"  цитат с [N]: {info['quotes']}, из них не дословных: "
              f"{len(info['quote_bad'])}; без [N]: {len(info['quote_loose'])}")
    for b in info["quote_bad"]:
        real = (f" → текст на самом деле в [{','.join(map(str, b['found_in']))}]"
                if b["found_in"] else " → ни в одной статье")
        print(f"  ❌ [{','.join(map(str, b['n']))}]「{b['quote'][:50]}」{real}")
    if info["num_unquoted"]:
        print(f"  ⚠ числа вне цитат: {info['num_unquoted']}")


def quote_summary(rows):
    """Quote metrics over report rows. Shared by --eval and rescore_quotes.py:
    the metric can change without regenerating a single answer."""
    answered = [r_ for r_ in rows if r_["verdict"] == "ok"]
    if not answered:
        return
    nq = sum(r_["quotes"] for r_ in answered)
    bads = [(r_, b) for r_ in answered for b in r_["quote_bad"]]
    moved = [x for x in bads if x[1]["found_in"]]
    nc = sum(r_["claims"] for r_ in answered)
    ncq = sum(r_["claims_quoted"] for r_ in answered)
    cc = sum(r_["claim_chars"] for r_ in answered)
    qc = sum(r_["quoted_chars"] for r_ in answered)
    print(f"\nцитаты: {nq} в {sum(1 for r_ in answered if r_['quotes'])}/{len(answered)} ответах")
    if nc:
        # Read coverage BEFORE verbatim-ness: high verbatim-ness with low
        # coverage means the model quotes only what is easy to quote.
        print(f"  покрытие утверждений: {ncq}/{nc} = {ncq / nc:.2f}  "
              f"предложение с [N] и хотя бы одной дословной цитатой")
        print(f"  покрытие текста:      {qc}/{cc} = {qc / cc:.2f}  "
              f"доля символов утверждений внутри дословных цитат")
    if nq:
        print(f"  дословных:            {nq - len(bads)}/{nq} = {(nq - len(bads)) / nq:.2f}")
    print(f"  не из своей [N]:      {len(bads)}  (текст есть под другим номером: {len(moved)})")
    print(f"  цитат в предложениях без [N]: {sum(len(r_['quote_loose']) for r_ in answered)}")
    print(f"  ответов с числом вне цитат: {sum(1 for r_ in answered if r_['num_unquoted'])}")
    for r_, b in bads:
        real = (f" → в [{','.join(map(str, b['found_in']))}]" if b["found_in"]
                else " → нигде")
        print(f"  #{r_['id']} [{','.join(map(str, b['n']))}]{real}  「{b['quote'][:40]}」")


def cmd_eval(mode=DEFAULT_MODE, quotes=True):
    golden = load_golden()
    r = Reranked()
    rows = []
    system = SYSTEM_QUOTES if quotes else SYSTEM_BASE
    print(f"сборка контекста: {mode}{' · промпт с цитатами' if quotes else ''}\n")
    with httpx.Client() as cl:
        for i, g in enumerate(golden, 1):
            ch = retrieve(r, g["q"], mode)
            ans = generate(g["q"], ch, cl, system=system)
            info = inspect(ans, ch)

            # Did retrieval hit: is at least one gold article present in the context.
            # This separates two different questions that a single number conflates:
            # "did search find it" and "did the model use what was found".
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

    quote_summary(rows)

    # ─── retrieval versus generation (§4.5) ───
    # The overall refusal rate says nothing: refusing when search missed is
    # correct behaviour, refusing when it hit is a loss. Two error rates,
    # not one, and they must be optimised in opposite directions.
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

    # A separate report: the quote prompt changes generation, and the baseline
    # gen_report_whole.json must stay — the ablation is paired against it.
    out = report_path(mode, "_quotes" if quotes else "")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}")


def cmd_ablate(mode=DEFAULT_MODE):
    """Ablation: for questions that hit, remove the gold article from the context.

    Why: an ordinary run has only 6 retrieval misses, and 0.17 refusal against
    0.05 on a sample that size is indistinguishable from chance. This gives 44
    controlled misses with realistic neighbouring distractors.

    Paired design: the same questions, differing only in whether the source is
    present. "Some questions are simply harder" is excluded by construction.

    The baseline run (--eval) must exist: it tells us which questions were a
    hit, and supplies the verdict for the paired comparison.
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
            ans = generate(g["q"], ch, cl, system=SYSTEM_BASE)  # paired with the base-prompt report
            info = inspect(ans, ch)

            # Sanity check: the gold article is definitely absent from the context.
            # Without it the whole experiment could silently measure something else.
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

    # Paired table: the same question with the source and without it.
    # If the model senses the presence of a source, the mass should move
    # into the "was 回答可能 -> became 回答不可" cell.
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
    # The quote prompt is the default; --base is the phase 4 prompt.
    quotes = "--base" not in args
    if "--ablate" in args:
        return cmd_ablate(mode)
    if "--sample" in args:
        return cmd_sample(mode=mode, quotes=quotes)
    if "--eval" in args:
        return cmd_eval(mode=mode, quotes=quotes)

    query = " ".join(a for a in args if not a.startswith("--")) or \
        "有給って入社してすぐもらえるの？"
    r = Reranked()
    ch = retrieve(r, query, mode)
    ans = generate(query, ch, system=SYSTEM_QUOTES if quotes else SYSTEM_BASE)
    print(f"\n{ans}\n")
    print("— источники —")
    for i, c in enumerate(ch, 1):
        print(f"[{i}] {c['law_title'][:24]} {c['article_title']}"
              f"{c['article_caption'] or ''}")
    print("— цитаты —")
    print_quotes(inspect(ans, ch))


if __name__ == "__main__":
    main()
