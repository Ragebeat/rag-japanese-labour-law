"""Is the generation context silently truncated?

generate() sends no num_ctx. Current Ollama picks the default window from
VRAM: 4096 tokens below ~24 GiB. One user message holds five whole articles
and the question; if it does not fit, Ollama trims it WITHOUT an error, and
the answer still reads fine. If it trims the front, the loss is article [1] —
the one the reranker ranked highest.

The check: build exactly the prompt generate() builds, for every golden
question, and ask Ollama for ONE token twice — with the default window and
with a large one. A different prompt_eval_count means that prompt was cut.

Two passes, not alternating per question: changing num_ctx reloads the
model, and alternating would reload it a hundred times.

If the base-prompt report (gen_report_whole.json) exists, truncated questions are matched with their
recorded hit / verdict — to see whether truncation explains any of the
over-refusals measured back then.

  uv run python src/check_ctx.py
"""

import json
import pathlib
import statistics
import sys

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from generate import (DEFAULT_MODE, MODEL, SYSTEM, URL,  # noqa: E402
                      build_context, load_golden, report_path, retrieve)
from rerank import Reranked  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "ctx_check.json"
BIG = 32_768
TOLERANCE = 8      # tokens; the two counts should be identical when nothing is cut


def messages(q, chunks):
    # Byte for byte what generate() sends — otherwise we measure another prompt.
    return [{"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"【条文】\n{build_context(chunks)}\n\n【質問】\n{q}"}]


def count(http, msgs, num_ctx=None):
    options = {"temperature": 0, "num_predict": 1}
    if num_ctx:
        options["num_ctx"] = num_ctx
    r = http.post(URL, json={"model": MODEL, "messages": msgs, "stream": False,
                             "think": False, "options": options}, timeout=300)
    r.raise_for_status()
    return r.json().get("prompt_eval_count", 0)


def main():
    golden = load_golden()
    r = Reranked()
    print(f"retrieving {len(golden)} questions, mode {DEFAULT_MODE}…")
    prompts = [(g, messages(g["q"], retrieve(r, g["q"], DEFAULT_MODE)))
               for g in golden]

    rows = {g["id"]: {"id": g["id"], "q": g["q"]} for g, _ in prompts}
    with httpx.Client() as http:
        for label, ctx in (("default", None), ("big", BIG)):
            print(f"pass \"{label}\" (num_ctx = {ctx or 'default'})…")
            for g, msgs in prompts:
                rows[g["id"]][label] = count(http, msgs, ctx)

    rows = list(rows.values())
    full = [x["big"] for x in rows]
    cut = [x for x in rows if x["default"] < x["big"] - TOLERANCE]

    print(f"\ngeneration prompt size, tokens: min {min(full)} · "
          f"median {int(statistics.median(full))} · max {max(full)}")
    if max(full) > BIG * 0.95:
        print(f"⚠ even {BIG} is borderline — the \"big\" pass itself may have truncated")

    print(f"\ntruncated: {len(cut)}/{len(rows)}")
    if not cut:
        print("→ the earlier measurements used the full context; num_ctx had no effect")
    else:
        window = max(x["default"] for x in cut)
        print(f"default window, judging by the truncated ones: ~{window} tokens")

        base = {}
        if report_path(DEFAULT_MODE).exists():
            base = {b["id"]: b for b in json.loads(
                report_path(DEFAULT_MODE).read_text(encoding="utf-8"))}

        print("\n  id    full    kept   lost       hit  verdict  question")
        for x in sorted(cut, key=lambda x: x["default"] - x["big"]):
            b = base.get(x["id"], {})
            hit = {True: "hit", False: "MISS"}.get(b.get("hit"), " ? ")
            verdict = {"ok": "回答可能", "no": "回答不可"}.get(b.get("verdict"), "   ?   ")
            print(f"  #{x['id']:<4} {x['big']:>6} {x['default']:>6} "
                  f"{x['big'] - x['default']:>8}   {hit:<4} {verdict}  {x['q'][:24]}")

        if base:
            over = [x for x in cut if base.get(x["id"], {}).get("hit")
                    and base[x["id"]].get("verdict") == "no"]
            if over:
                print(f"\n⚠ {len(over)} of the truncated are earlier over-refusals "
                      f"(article found, model refused): "
                      + ", ".join(f"#{x['id']}" for x in over)
                      + "\n  Candidate explanation: the model simply never saw the article")

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
