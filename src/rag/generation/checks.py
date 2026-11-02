"""Checks on a generated answer, done by code rather than by a model: the 判定
verdict, cited articles missing from the context, verbatim quotes and the
paragraphs they are cited under."""

import difflib
import re
import unicodedata

from rag.generation.guard import guard_links
from rag.text.jnum import NUM, art_key, num
from rag.text.jnum import fmt as _fmt

# --- verdict and citation markers ---
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


# --- answer analysis ---

def context_arts(chunks):
    """Two sets of articles occurring in the context.

    source      articles we actually supplied (article_title)
    mentioned   plus everything named in chunk captions and bodies

    The distinction is not formal. Japanese law is built on cross-references
    throughout (「法第二十五条第一項の…」, 「労働基準法第八十九条及び第九十条の
    定めるところによる」). An article named that way is not a fabrication, but
    it is not a source either: the model never saw its text, so there is
    nothing to verify the content against.

    The first run counted `source` only, and produced 6 false hallucinations
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

    This is NOT faithfulness. It checks the source only: whether the model
    cited what it was given. The case of "right number, wrong duration" is not
    caught here, and neither is "the right article was in the context but the
    model did not use it". Those need human annotation.
    """
    lines = [ln for ln in answer.splitlines() if ln.strip()]
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
        # 回答不可 and nothing else: never seen once in the first run
        "clean_refusal": verdict == "no" and not marks and not cited,
        "partial_refusal": verdict == "no" and bool(marks or cited),
        # said "I can answer" and hedged in the same breath: over-hedging
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


# --- verbatim quote check ---
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
    copied verbatim from any of the sources: brackets or not.

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
    quotes: the model writes natural Japanese, with the quote mid-sentence and
    the marker at the end: 「…」を支払わなければなりません[1].

    Checked against the sentence's own [N] chunks, not the whole context:
    otherwise text of one article quoted under another's number would pass
    (a real case from earlier runs). found_in says where the text really is: non-empty
    means "real text, wrong number", empty means "not a real quote".
    """
    texts = [_norm(c["text"].split("\n", 1)[-1]) for c in chunks]

    # Paragraph maps, where the paragraph structure is known: an assembled
    # article carries "paragraphs"; a single-paragraph chunk its own number.
    pmaps = []
    for c, t in zip(chunks, texts, strict=True):
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
    # exceptions (the splice described above, again, outside the quotes) while
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
        # cites exactly one article and that article has a paragraph map:
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
                    # alone, then there is nothing to compare, not an error.
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


def print_quotes(info):
    if info["claims"]:
        print(f"  claims with [N]: {info['claims']}, with a verbatim quote: "
              f"{info['claims_quoted']}  ← the rest cannot be checked")
    if info["quotes"] or info["quote_loose"]:
        print(f"  quotes with [N]: {info['quotes']}, not verbatim: "
              f"{len(info['quote_bad'])}; without [N]: {len(info['quote_loose'])}")
    for b in info["quote_bad"]:
        real = (f" → the text is actually in [{','.join(map(str, b['found_in']))}]"
                if b["found_in"] else " → in no article")
        print(f"  FAIL [{','.join(map(str, b['n']))}]「{b['quote'][:50]}」{real}")
    if info["num_unquoted"]:
        print(f"  WARNING: numbers outside quotes: {info['num_unquoted']}")
    for b in info.get("para_bad", []):
        print(f"  FAIL [{b['n']}] cites 第{','.join(map(str, b['cited']))}項, "
              f"but the quote is in 第{','.join(map(str, b['found']))}項  「{b['quote'][:40]}」")


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
    print(f"\nquotes: {nq} in {sum(1 for r_ in answered if r_['quotes'])}/{len(answered)} answers")
    if nc:
        # Read coverage BEFORE verbatim-ness: high verbatim-ness with low
        # coverage means the model quotes only what is easy to quote.
        print(f"  claim coverage:       {ncq}/{nc} = {ncq / nc:.2f}  "
              f"a sentence with [N] and at least one verbatim quote")
        print(f"  text coverage:        {qc}/{cc} = {qc / cc:.2f}  "
              f"share of claim characters inside verbatim quotes")
        vc = sum(r_.get("verbatim_chars", 0) for r_ in answered)
        tc = sum(r_.get("content_chars", 0) for r_ in answered)
        if tc:
            print(f"  verbatim from source: {vc}/{tc} = {vc / tc:.2f}  "
                  f"claim text matching the article in runs of {MIN_RUN}+ characters, quoted or not")
    if nq:
        print(f"  verbatim:             {nq - len(bads)}/{nq} = {(nq - len(bads)) / nq:.2f}")
    print(f"  not from its own [N]: {len(bads)}  (text found under another number: {len(moved)})")
    print(f"  quotes in sentences without [N]: {sum(len(r_['quote_loose']) for r_ in answered)}")
    print(f"  answers with a number outside quotes: {sum(1 for r_ in answered if r_['num_unquoted'])}")
    # --- paragraphs: does the quote sit in the 項 the answer names ---
    npc = sum(r_.get("para_claims", 0) for r_ in answered)
    pbad = [(r_, b) for r_ in answered for b in r_.get("para_bad", [])]
    print(f"  claims naming a 項: {npc}; quote not in the named 項: {len(pbad)}")
    for r_, b in pbad:
        print(f"  #{r_['id']} [{b['n']}] cites 第{','.join(map(str, b['cited']))}項 "
              f"→ in 第{','.join(map(str, b['found']))}項  「{b['quote'][:40]}」")
    for r_, b in bads:
        real = (f" → in [{','.join(map(str, b['found_in']))}]" if b["found_in"]
                else " → nowhere")
        print(f"  #{r_['id']} [{','.join(map(str, b['n']))}]{real}  「{b['quote'][:40]}」")
