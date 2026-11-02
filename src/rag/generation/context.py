"""Assembling what the model reads: retrieval settings, the raw / dedupe /
whole-article modes, and the [N]-numbered context block."""

import collections
import json

from rag.config import ROOT
from rag.text.query_expand import synonyms

CHUNKS = ROOT / "data" / "chunks.jsonl"

TOP_K = 5


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


def term_hint(query):
    """Colloquial -> statutory term pairs from data/synonyms.json (the same
    dictionary hybrid.py uses for SEARCH), shown to the generator too. Search
    expands the query; the generator used to see only the raw question and
    refused #22 (失業保険 vs 基本手当 in the article). First term only,
    as in narrow mode: a long tail dilutes."""
    pairs = [f"{s}＝{terms[0]}" for s, terms in synonyms().items()
             if s in query and terms[0] not in query]
    return "\n\n【用語の対応】" + "、".join(pairs) if pairs else ""


# Ruri, no query expansion. sweep.py --ruri, r@5 on golden:
# with expansion 0.96, without 0.94: one question net (+2 −1), noise.
# Expansion was dropped because the dictionary was written against golden's
# own misses (its gain there is optimistic), and it costs an LLM call per
# query. 0.94 is the one number with nothing fitted to golden.
KW = dict(mode="vector", rerank_on="original")

POOL_MULT = 4     # how many candidates to take before collapsing: TOP_K * POOL_MULT
ARTICLE_CAP = 3000  # characters; beyond this we do not assemble the whole article


def dedupe_by_article(chunks, k):
    """One chunk per article: keep the best paragraph, drop the rest.

    The list is already sorted by the reranker, so the first occurrence is the
    best one: remembering what has been seen is enough.

    The key is (law_id, article_title), not the title alone: 第五条 exists in
    almost every statute in the corpus, so one number would collapse different
    articles together.

    This changes retrieval, not just presentation. More DISTINCT articles
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
    asking, 2項 (six months) is the correct one: the model never saw it and
    could not have used it.

    Level 3 of the chunker splits with overlap, so adjacent pieces partly
    repeat. Exact duplicates are dropped, partial ones remain. That is 1.6% of
    the corpus, but if it ever surfaces in answers, fix it here.
    """
    parts = arts.get((c["law_id"], c["article_title"]), [c])
    total = sum(p["n_chars"] for p in parts)
    if len(parts) == 1 or total > ARTICLE_CAP:
        return c
    # Paragraph labels. The chunk header carries 「第N項」, but
    # only the body went into the context: the model never saw paragraph
    # numbers and had to count paragraphs itself (an earlier run put 時季変更権
    # under 第3項: it is 第5項). Labels also give check_quotes a map to
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

    raw      top-k paragraphs as they come (baseline; duplicates eat slots)
    dedupe   one best paragraph per article (loses neighbouring paragraphs)
    whole    k distinct articles, each in full

    The reranker always gets pool = k * POOL_MULT and we take the same number:
    it scores the entire pool regardless, so collapsing and filtering are free
    in time terms.

    exclude: a set of (law_id, article_title) removed from the pool before
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
