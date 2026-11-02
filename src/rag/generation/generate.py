"""Generating an answer from the retrieved articles, with citations.

The key decision: the model answers ONLY from the supplied chunks, and cites
by an [N] marker: the chunk number we put into the context ourselves. If the
answer is not in the context, it must refuse.

Why a marker rather than an article number: an article number has to be dug
out of free text with a regex, and the model writes it however it likes
(第39条 / 第三十九条 / 第３９条). The [N] marker is ours, so mapping answer to
chunk becomes exact instead of heuristic. The model also writes the article
number for human readers, and that is verified a second, independent way.

  python -m rag.generation.generate "有給って入社してすぐもらえるの？"
  python -m rag.generation.generate --sample     # 5 questions raw, to read by eye
  python -m rag.generation.generate --eval       # run over the golden set
  python -m rag.generation.generate --eval --base     # the original prompt, without verbatim quotes

Every --eval is also logged to data/runs.jsonl with a configuration
fingerprint; experiments/compare.py diffs any two runs.
"""

import json
import sys

import httpx

from rag import tracing
from rag.config import ROOT
from rag.generation.checks import inspect, print_quotes, quote_summary
from rag.generation.context import TOP_K, build_context, retrieve, term_hint
from rag.generation.guard import guard_links
from rag.generation.prompts import GEN_OPTIONS, MODEL, SYSTEM, SYSTEM_BASE, SYSTEM_QUOTES, URL
from rag.generation.runs import DEFAULT_MODE, log_run, report_path
from rag.retrieval.rerank import Reranked

GOLDEN = ROOT / "data" / "golden.jsonl"


def generate(query, chunks, client=None, system=SYSTEM, builder=None):
    # builder: an alternative context format (injection_test.py wraps
    # documents in <document> tags). Default: build_context.
    own = client is None
    client = client or httpx.Client()
    try:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",
             "content": f"【条文】\n{(builder or build_context)(chunks)}\n\n【質問】\n{query}{term_hint(query)}"},
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


def load_golden():
    return [json.loads(line) for line in GOLDEN.open(encoding="utf-8") if line.strip()]


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
            print("— context —")
            for i, c in enumerate(ch, 1):
                print(f"  [{i}] {c['law_title'][:22]} {c['article_title']}"
                      f"{c['article_caption'] or ''}")
            print("— answer —")
            print(ans)
            print("— analysis —")
            v = {"ok": "回答可能", "no": "回答不可",
                 None: "NO VERDICT"}[info["verdict"]]
            print(f"  判定: {v}"
                  f"{' (not on the first line)' if info['verdict_misplaced'] else ''}")
            print(f"  markers {info['marks']}  articles {info['cited']}")
            if info["transitive"]:
                print(f"  ↳ transitive references: {info['transitive']}")
            if info["bad_marks"]:
                print(f"  FAIL marker out of range: {info['bad_marks']}")
            if info["bad_arts"]:
                print(f"  FAIL article not in the context: {info['bad_arts']}")
            if info["partial_refusal"]:
                print("  WARNING: incomplete refusal: 回答不可, yet citations present")
            if info["hedged"] and info["verdict"] == "ok":
                print("  WARNING: hedged: 回答可能, but the text hedges")
            print_quotes(info)
            print()


def cmd_eval(mode=DEFAULT_MODE, quotes=True):
    golden = load_golden()
    r = Reranked()
    rows = []
    system = SYSTEM_QUOTES if quotes else SYSTEM_BASE
    print(f"context assembly: {mode}{' · quote prompt' if quotes else ''}\n")
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
            flag = {"no": " NO", "ok": "   ", None: " ? "}[info["verdict"]]
            if info["bad_arts"] or info["bad_marks"]:
                flag = flag[:2] + "✗"
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

    print(f"\ntotal:                {n}")
    print(f"no verdict:           {len(novrd):>3}   ← format contract broken")
    print(f"回答不可 (refusal):    {len(refus):>3}   of them clean {len(clean)}")
    print(f"hedged:               {len(hedge):>3}   回答可能 + a hedge in the text")
    print(f"answers with no marker: {len(no_mark):>3}")
    print(f"transitive references: {len(trans):>3}   article named in a chunk body, not a source")
    print(f"fabricated references: {len(bad):>3}")
    if with_cit:
        print(f"citation_validity:    "
              f"{sum(r_['citation_validity'] for r_ in with_cit) / len(with_cit):.3f}"
              f"  (regression guard, 1.000 expected)")

    quote_summary(rows)

    # --- retrieval versus generation (§4.5) ---
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
    print(f"  A: article found    {len(q(True, 'ok')):>5}   {len(over):>5}  ← over-refusal")
    print(f"  B: article absent   {len(blind):>5}   {len(q(False, 'no')):>5}")
    print("                     ↑ answered without the source")
    if miss:
        print(f"  refusal on a miss:  {len(q(False, 'no')) / len(miss):.2f}   (higher is better)")
    if hits:
        print(f"  refusal on a hit:   {len(over) / len(hits):.2f}   (lower is better)")
    for label, group in (("over-refusal", over), ("answer without the source", blind)):
        if group:
            print(f"  {label}: " + ", ".join(f"#{r_['id']}" for r_ in group))

    # A separate report: the quote prompt changes generation, and the baseline
    # gen_report_whole.json must stay: the ablation is paired against it.
    out = report_path(mode, "_quotes" if quotes else "")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}  (latest run of this prompt — other scripts read it)")
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
        sys.exit(f"run --eval in mode {mode} first: "
                 f"{report_path(mode).name} is needed for the paired comparison")
    base = {b["id"]: b for b in
            json.loads(report_path(mode).read_text(encoding="utf-8"))}
    targets = [g for g in load_golden() if base.get(g["id"], {}).get("hit")]

    r = Reranked()
    rows = []
    print(f"ablation ({mode}): removing the gold article for {len(targets)} questions\n")
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
            mark = {"no": "REFUS", "ok": "answr", None: "  ?  "}[info["verdict"]]
            print(f"{i:>2}/{len(targets)} {mark} {g['q'][:26]}")

    n = len(rows)
    refused = [x for x in rows if x["verdict"] == "no"]
    answered = [x for x in rows if x["verdict"] == "ok"]
    substituted = [x for x in answered if x["marks"]]

    print(f"\nablated questions:   {n}")
    print(f"refused:             {len(refused):>3}  = {len(refused) / n:.2f}  (higher is better)")
    print(f"answered from a neighbour: {len(substituted):>3}  = {len(substituted) / n:.2f}")

    # Paired table: the same question with the source and without it.
    # If the model senses the presence of a source, the mass should move
    # into the "was 回答可能 -> became 回答不可" cell.
    def cell(b, a):
        return sum(1 for x in rows if x["verdict_base"] == b and x["verdict"] == a)

    print("\npaired comparison (rows — with the source, columns — without)")
    print("              回答可能   回答不可")
    print(f"  回答可能      {cell('ok', 'ok'):>5}   {cell('ok', 'no'):>5}  ← noticed the loss")
    print(f"  回答不可      {cell('no', 'ok'):>5}   {cell('no', 'no'):>5}")
    print("              ↑ did not notice")

    out = report_path(mode, "_ablate")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}")


def main():
    args = sys.argv[1:]
    mode = ("raw" if "--nodedupe" in args else
            "dedupe" if "--dedupe" in args else
            "whole" if "--whole" in args else DEFAULT_MODE)
    # The quote prompt is the default; --base is the original prompt without quotes.
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
    print("— sources —")
    for i, c in enumerate(ch, 1):
        print(f"[{i}] {c['law_title'][:24]} {c['article_title']}"
              f"{c['article_caption'] or ''}")
    print("— quotes —")
    print_quotes(inspect(ans, ch))


if __name__ == "__main__":
    main()
