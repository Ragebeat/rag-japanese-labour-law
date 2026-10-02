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

Every --eval is also logged to data/runs.jsonl with a configuration
fingerprint; src/compare.py diffs any two runs.
"""

import collections
import difflib
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import unicodedata
from datetime import datetime

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rerank import Reranked  # noqa: E402
import hybrid  # noqa: E402
from jnum import NUM, art_key, fmt as _fmt, num  # noqa: E402
import tracing  # noqa: E402

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
7. 数字・日数・期間は「」の引用の中にだけ書く
8. 条文に（第N項）の印がある場合、根拠の項まで書く。形式:「[N] ◯◯法 第N条第N項」
   項の番号は必ず（第N項）の印から取る。段落を数えて推測しない"""

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

# ── run registry (phase 6) ──
# On 29.09 every --eval needed a manual Copy-Item first, or the new report
# overwrote the one to compare against — and the comparison itself was done
# by reading two JSON files. Now each run is kept under a fingerprint of
# everything that shapes the answer, and logged with its metrics.
#
# Bump CONTEXT_FORMAT by hand whenever build_context() / whole_article()
# change what the model sees: code is not part of the fingerprint, so a
# format change would otherwise hide under the same hash.
CONTEXT_FORMAT = "whole + 「（第N項）」 labels, 2026-09-29"
RUNS_DIR = ROOT / "data" / "runs"
RUNS_LOG = ROOT / "data" / "runs.jsonl"


def run_config(system, mode):
    """Everything that shapes an answer, read from the modules themselves —
    not retyped here, or this would become a fourth copy that drifts.
    The expander model is in: switching it (qwen → gemma) moved reranker
    scores enough to break a phase 5 "finding"."""
    import query_llm
    import rerank
    return {
        "generator": MODEL,
        "gen_options": GEN_OPTIONS,
        "system_sha": hashlib.sha256(system.encode()).hexdigest()[:12],
        "context_format": CONTEXT_FORMAT,
        "mode": mode,
        "retrieval": {"kw": KW, "top_k": TOP_K, "pool_mult": POOL_MULT,
                      "article_cap": ARTICLE_CAP},
        "expander": getattr(query_llm, "MODEL", None),
        "reranker": getattr(rerank, "MODEL", None),
        "reranker_revision": getattr(rerank, "REVISION", None),
        "embedder": hybrid.EMBEDDER,
    }


def fingerprint(config):
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def git_state():
    """Short commit, "+dirty" if src/ has uncommitted changes — then the
    commit alone does not reproduce the run."""
    try:
        run = lambda *a: subprocess.run(["git", *a], capture_output=True,  # noqa: E731
                                        text=True, cwd=ROOT, timeout=5).stdout.strip()
        head = run("rev-parse", "--short", "HEAD")
        return head + ("+dirty" if run("status", "--porcelain", "--", "src") else "")
    except Exception:                               # noqa: BLE001
        return None


def run_metrics(rows):
    """One summary per run, the same definitions for every run — that is
    what makes two runs comparable. Old reports without quote fields get
    None there, not zeros."""
    hit = [r for r in rows if r["hit"]]
    miss = [r for r in rows if not r["hit"]]
    ans = [r for r in rows if r["verdict"] == "ok"]
    cv = [r["citation_validity"] for r in rows if r.get("citation_validity") is not None]
    has_q = all("claims" in r for r in ans)

    def s(k):
        return sum(r.get(k, 0) for r in ans)

    def ratio(a, b):
        return round(a / b, 3) if b else None

    return {
        "n": len(rows),
        "recall": ratio(len(hit), len(rows)),
        "over_refusal": sum(1 for r in hit if r["verdict"] == "no"),
        "refusal_at_miss": ratio(sum(1 for r in miss if r["verdict"] == "no"), len(miss)),
        "no_verdict": sum(1 for r in rows if r["verdict"] not in ("ok", "no")),
        "bad_refs": sum(len(r.get("bad_arts", [])) for r in rows),
        "blocked_links": sum(len(r.get("blocked_links", [])) for r in rows),
        "citation_validity": ratio(sum(cv), len(cv)),
        "claim_coverage": ratio(s("claims_quoted"), s("claims")) if has_q else None,
        "text_coverage": ratio(s("quoted_chars"), s("claim_chars")) if has_q else None,
        "verbatim_coverage": ratio(s("verbatim_chars"), s("content_chars")) if has_q else None,
        "quotes": s("quotes") if has_q else None,
        "quote_bad": sum(len(r.get("quote_bad", [])) for r in ans) if has_q else None,
        "para_claims": s("para_claims") if has_q else None,
        "para_bad": sum(len(r.get("para_bad", [])) for r in ans) if has_q else None,
        "num_unquoted_answers": sum(1 for r in ans if r.get("num_unquoted")) if has_q else None,
    }


def log_run(rows, system, mode, label):
    """Keep this run's report under its own name and append a line to
    runs.jsonl. Two runs with the same fingerprint are a repeat — exactly
    what is needed to measure run-to-run noise."""
    config = run_config(system, mode)
    fp = fingerprint(config)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"{ts}_{label}_{fp}_{mode}.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    entry = {"ts": ts, "label": label, "fingerprint": fp, "git": git_state(),
             "report": str(path.relative_to(ROOT)), "config": config,
             "metrics": run_metrics(rows)}
    with RUNS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"→ прогон записан: {label} · отпечаток {fp} · {entry['git']} · {path.name}")
    return entry


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
        # A single paragraph (an article over ARTICLE_CAP, or dedupe/raw mode):
        # its number lives only in the stripped chunk header, so it goes into
        # the head. Assembled articles carry per-paragraph labels in the body.
        if "paragraphs" not in c and c.get("paragraph_num"):
            head += f" 第{c['paragraph_num']}項"
        body = c["text"].split("\n", 1)[-1]
        parts.append(f"[{i}] {head}\n{body}")
    return "\n\n".join(parts)


def generate(query, chunks, client=None, system=SYSTEM, builder=None):
    # builder: an alternative context format (phase 6 injection test wraps
    # documents in <document> tags). Default: build_context.
    own = client is None
    client = client or httpx.Client()
    try:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",
             "content": f"【条文】\n{(builder or build_context)(chunks)}\n\n【質問】\n{query}"},
        ]
        with tracing.span("generate", "LLM") as sp:
            tracing.llm_request(sp, MODEL, messages, GEN_OPTIONS)
            r = client.post(URL, json={
                "model": MODEL, "messages": messages,
                "stream": False, "think": False,
                "options": GEN_OPTIONS,
            }, timeout=300)
            if r.is_success:
                tracing.llm_response(sp, r.json())
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
        "blocked_links": guard_links(body)[1],
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
# Bracket-free verbatim coverage: a run of 10+ identical characters between
# the claim and its cited article is copying, not chance.
MIN_RUN = 10
ELLIPSIS_RE = re.compile(r"…+|\.{3,}|・{3,}")
# A number with a unit. 条/項/号 are deliberately absent: article references
# legitimately live outside quotes.
NUM_RE = re.compile(r"[0-9〇一二三四五六七八九十百千]+"
                    r"(?:日|箇月|か月|ヶ月|カ月|ケ月|時間|年|週|割|円|%|人)")
# A paragraph reference in the answer: 第五項 / 第5項 / 第５項.
PARA_RE = re.compile(rf"第({NUM})項")


def _outside(s):
    """The sentence with top-level 「…」 removed: a 第N項 inside a quote is
    the statute talking about itself (「第三項において同じ」), not the answer
    naming its source."""
    out, depth = [], 0
    for ch in s:
        if ch == "「":
            depth += 1
        elif ch == "」" and depth:
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out)


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


def _covered(content, sources):
    """How many characters of content lie in runs of MIN_RUN+ characters
    copied verbatim from any of the sources — brackets or not.

    Why: after the paragraph rule the model kept copying statute text word
    for word but dropped the 「」 (#42, #50: whole article sentences, even the
    old kana 代つて). Bracket-based coverage called that paraphrase. The
    question is whether the claim rests on the source text, not how it is
    typeset."""
    hit = [False] * len(content)
    for t in sources:
        sm = difflib.SequenceMatcher(None, content, t, autojunk=False)
        for b in sm.get_matching_blocks():
            if b.size >= MIN_RUN:
                hit[b.a:b.a + b.size] = [True] * b.size
    return sum(hit)


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

    # Paragraph maps, where the paragraph structure is known: an assembled
    # article carries "paragraphs"; a single-paragraph chunk its own number.
    pmaps = []
    for c, t in zip(chunks, texts):
        if c.get("paragraphs"):
            pmaps.append({k: _norm(v) for k, v in c["paragraphs"].items()})
        elif c.get("paragraph_num") and str(c["paragraph_num"]).isdigit():
            pmaps.append({int(c["paragraph_num"]): t})
        else:
            pmaps.append(None)

    def where(q):
        return [i + 1 for i, t in enumerate(texts) if _verbatim(q, t)]

    # Two coverages, because verbatim-ness alone rewards NOT quoting. The
    # first test answer quoted the undisputed sentence and paraphrased the
    # exceptions — the phase 5 splice again, outside the quotes — while
    # reporting "0 non-verbatim". And sentence coverage alone rewards a
    # token quote: 「妊娠したこと」 in brackets, the claim itself paraphrased.
    # So: claims_quoted (sentences with a verbatim quote) AND quoted_chars
    # (share of claim text that is verbatim).
    bad, loose, para_bad = [], [], []
    n_quotes = claims = claims_quoted = claim_chars = quoted_chars = 0
    para_claims = verbatim_chars = content_chars = 0
    for s in _sentences(body):
        src = [int(x) for x in MARK_RE.findall(s) if 1 <= int(x) <= len(texts)]
        # The paragraph the sentence names. Checkable only when the sentence
        # cites exactly one article and that article has a paragraph map —
        # with two articles cited, 第N項 could belong to either.
        one = src[0] if len(set(src)) == 1 else None
        pm = pmaps[one - 1] if one else None
        refs = ({num(x) for x in PARA_RE.findall(_outside(s))} - {None}) if pm else set()
        if refs:
            para_claims += 1
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
                if refs:
                    # A quote spanning two paragraphs sits in none of them
                    # alone — then there is nothing to compare, not an error.
                    inside = [k for k, t in pm.items() if _verbatim(q, t)]
                    if inside and not set(inside) & refs:
                        para_bad.append({"quote": q, "n": one,
                                         "cited": sorted(refs), "found": inside})
            else:
                bad.append({"quote": q, "n": src, "found_in": where(q)})
        if src:
            claims += 1
            claims_quoted += ok_chars > 0
            claim_chars += len(_norm(MARK_RE.sub("", s)))
            quoted_chars += ok_chars
            # The claim is the text BEFORE its first [N]; what follows is the
            # reference itself (law name, article, caption), not content.
            content = _norm(MARK_RE.split(s)[0])
            content_chars += len(content)
            verbatim_chars += _covered(content, [texts[n - 1] for n in set(src)])

    outside = unicodedata.normalize("NFKC", ANY_QUOTE_RE.sub("", body))
    return {"quotes": n_quotes, "quote_bad": bad, "quote_loose": loose,
            "num_unquoted": NUM_RE.findall(outside),
            "claims": claims, "claims_quoted": claims_quoted,
            "claim_chars": claim_chars, "quoted_chars": quoted_chars,
            "para_claims": para_claims, "para_bad": para_bad,
            "verbatim_chars": verbatim_chars, "content_chars": content_chars}


# ──────────────────────────────────── cli ────────────────────────────────────

# Phase 7: Ruri, no query expansion. sweep.py --ruri, r@5 on golden:
# with expansion 0.96, without 0.94 — one question net (+2 −1), noise.
# Expansion was dropped because the dictionary was written against golden's
# own misses (its gain there is optimistic), and it costs an LLM call per
# query. 0.94 is the one number with nothing fitted to golden.
KW = dict(mode="vector", rerank_on="original")

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
    # Paragraph labels (phase 6). The chunk header carries 「第N項」, but
    # only the body went into the context: the model never saw paragraph
    # numbers and had to count paragraphs itself (phase 5: 時季変更権 put
    # under 第3項 — it is 第5項). Labels also give check_quotes a map to
    # verify that a quote sits in the paragraph the answer names.
    labelled = len({p.get("paragraph_num") for p in parts}) > 1
    seen, bodies, paras, prev = set(), [], {}, None
    for p in parts:
        body = p["text"].split("\n", 1)[-1]
        if body in seen:
            continue
        seen.add(body)
        pn = p.get("paragraph_num")
        if pn and str(pn).isdigit():
            paras.setdefault(int(pn), []).append(body)
        if labelled and pn and pn != prev:
            body = f"（第{pn}項）{body}"
        prev = pn
        bodies.append(body)
    out = dict(c)
    out["text"] = c["text"].split("\n", 1)[0] + "\n" + "\n".join(bodies)
    out["n_chars"] = total
    out["n_paragraphs"] = len(bodies)
    if labelled:
        out["paragraphs"] = {k: "\n".join(v) for k, v in paras.items()}
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


# ── output guard: links (phase 6) ──
# The injection test: a poisoned document made the model append the
# attacker's link 4-8 times out of 8 under every prompt defence — and one
# sentence of stronger wording beat delimiting and escaping, 8/8. Only a
# code check on the output held, 32 of 32. A prompt lowers the odds; an
# allow-list on what leaves the system removes them for this class.
#
# An ALLOW-list, not "is the link in the sources": a poisoned source
# contains it. Bare domains and e-mail addresses count too — an attacker
# does not need the https://.
ALLOWED_DOMAINS = ("e-gov.go.jp", "mhlw.go.jp")
DOMAIN_RE = re.compile(r"(?:https?://|[A-Za-z0-9._%+-]+@)?((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})"
                       r"(?:[/:?#][^\s「」『』、。（）()]*)?")
LINK_MASK = "［リンク削除］"


def _allowed(domain):
    d = domain.lower()
    return any(d == a or d.endswith("." + a) for a in ALLOWED_DOMAINS)


def guard_links(text):
    """-> (text with every non-allow-listed link masked, blocked domains)."""
    blocked = []

    def repl(m):
        if _allowed(m.group(1)):
            return m.group(0)
        blocked.append(m.group(1))
        return LINK_MASK
    return DOMAIN_RE.sub(repl, text), blocked


_URL_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                 "0123456789.-:/?#&=%_@+~")


class StreamGuard:
    """guard_links for streamed text. A link must not be half-sent before it
    can be checked, so a trailing run of URL-like characters is held back
    until something else follows it. Everything emitted is cut at a non-URL
    character, so no link ever straddles two emitted chunks."""

    def __init__(self):
        self.raw, self.sent, self.blocked = "", 0, []

    def push(self, delta):
        self.raw += delta
        cut = len(self.raw)
        while cut > self.sent and self.raw[cut - 1] in _URL_CHARS:
            cut -= 1
        return self._emit(cut)

    def flush(self):
        return self._emit(len(self.raw))

    def _emit(self, cut):
        chunk, blocked = guard_links(self.raw[self.sent:cut])
        self.sent = cut
        self.blocked += blocked
        return chunk


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
    for b in info.get("para_bad", []):
        print(f"  ❌ [{b['n']}] указан 第{','.join(map(str, b['cited']))}項, "
              f"а цитата в 第{','.join(map(str, b['found']))}項  「{b['quote'][:40]}」")


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
        vc = sum(r_.get("verbatim_chars", 0) for r_ in answered)
        tc = sum(r_.get("content_chars", 0) for r_ in answered)
        if tc:
            print(f"  дословно из источника: {vc}/{tc} = {vc / tc:.2f}  "
                  f"текст утверждений, совпадающий с статьёй кусками от {MIN_RUN} символов, с кавычками или без")
    if nq:
        print(f"  дословных:            {nq - len(bads)}/{nq} = {(nq - len(bads)) / nq:.2f}")
    print(f"  не из своей [N]:      {len(bads)}  (текст есть под другим номером: {len(moved)})")
    print(f"  цитат в предложениях без [N]: {sum(len(r_['quote_loose']) for r_ in answered)}")
    print(f"  ответов с числом вне цитат: {sum(1 for r_ in answered if r_['num_unquoted'])}")
    # ── paragraphs: does the quote sit in the 項 the answer names ──
    npc = sum(r_.get("para_claims", 0) for r_ in answered)
    pbad = [(r_, b) for r_ in answered for b in r_.get("para_bad", [])]
    print(f"  утверждений с указанным 項: {npc}; цитата не в указанном 項: {len(pbad)}")
    for r_, b in pbad:
        print(f"  #{r_['id']} [{b['n']}] указан 第{','.join(map(str, b['cited']))}項 "
              f"→ в 第{','.join(map(str, b['found']))}項  「{b['quote'][:40]}」")
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
            # One trace per golden question: retrieval and generation nest
            # under it, so a slow or odd answer can be opened by its id.
            with tracing.span("rag.answer", "CHAIN", input=g["q"],
                              attrs={"golden.id": g["id"]}) as sp:
                with tracing.span("retrieve", "RETRIEVER", input=g["q"]) as rs:
                    ch = retrieve(r, g["q"], mode)
                    tracing.documents(rs, ch)
                ans = generate(g["q"], ch, cl, system=system)
                info = inspect(ans, ch)
                tracing.set_attrs(sp, {"output.value": ans, "rag.verdict": info["verdict"]})

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
    print(f"→ {out}  (последний прогон этого промпта — его читают другие скрипты)")
    log_run(rows, system, mode, "quotes" if quotes else "base")


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
    tracing.setup()
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
    with tracing.span("rag.answer", "CHAIN", input=query) as sp:
        with tracing.span("retrieve", "RETRIEVER", input=query) as rs:
            ch = retrieve(r, query, mode)
            tracing.documents(rs, ch)
        ans = generate(query, ch, system=SYSTEM_QUOTES if quotes else SYSTEM_BASE)
        tracing.set_attrs(sp, {"output.value": ans})
    print(f"\n{guard_links(ans)[0]}\n")
    print("— источники —")
    for i, c in enumerate(ch, 1):
        print(f"[{i}] {c['law_title'][:24]} {c['article_title']}"
              f"{c['article_caption'] or ''}")
    print("— цитаты —")
    print_quotes(inspect(ans, ch))


if __name__ == "__main__":
    main()
