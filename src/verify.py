"""The verification pass: checking an answer against the context with a separate model.

WHY NOT PLAIN faithfulness. Tested on our worst cases — it does not catch them:
  #7  the "ninety days" came from 労組法15条, which WAS in the context
  #43 the "ten days" came from 39条1項, which was there too
Both statements follow from the context. This matches the measurements:
0 fabricated citations out of 50, distortion in 3 of 20 — there is almost
nothing here of the kind faithfulness catches.

Hence three checks, two of which are useful:
  適用 — is the cited article applicable to what was asked
         (catches substituted institutions: 16 cases out of 30 in the ablation)
  欠落 — was there a provision in the context that changes the answer, and was
         it left unused (catches #43, #22, #19, #31)
  矛盾 — does what is said follow from the context (classic faithfulness)

THE JUDGE IS FROM A DIFFERENT FAMILY than the generator. The same gemma would
be blind in the same places as the generator, and we would be measuring a
model's agreement with itself.

The judge emits controlled tokens rather than prose — a technique
that has already worked twice (the [N] marker, the 判定 verdict).

  uv run python src/verify.py --calibrate   # 20 questions against manual annotation
  uv run python src/verify.py               # the full report for the current mode
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
# The judge must be from a DIFFERENT family than the generator (gemma):
# the same one would be blind in the same places as the generator.
# Set by an environment variable so judge comparisons are reproducible
# rather than a constant edited between runs.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen3:8b")

# ⚠️ No real articles appear in the format examples — an example in a prompt becomes a template.
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
        # The list of citations is passed explicitly. Without it the judge scored all
        # five context chunks, i.e. redid retrieval's work instead of checking the
        # answer — hence 13 blocks out of 20 in the first calibration.
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
    """Parses the judge's three lines. Averages nothing."""
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

    # The judge referenced [4] when there were three chunks — clean that here too.
    if missing:
        idx = [int(m) for m in re.findall(r"\[(\d+)\]", missing)]
        if idx and not all(1 <= i <= n_chunks for i in idx):
            missing = ""
    # The contradiction slot must hold a STATEMENT, not an article name.
    # The judge wrote 「労働者災害補償保険法 第十三条」 — the slot is filled,
    # the regex is satisfied, and there is no content.
    if contra and re.fullmatch(r"[^\u3002]{0,40}第[^\u3002]{1,12}条[^\u3002]{0,20}", contra):
        contra = ""
    return {
        "raw": text,
        "parsed": bool(applic) or missing is not None,
        "inapplicable": bad,
        "missing": missing or "",
        "contradiction": contra or "",
        # block on: an inapplicable citation OR a missing provision
        # (omission blocks on equal terms)
        "block": bool(bad) or bool(missing) or bool(contra),
    }


def run(rows, mode, client, r):
    out = []
    for i, row in enumerate(rows, 1):
        ch = retrieve(r, row["q"], mode)
        raw = judge(row["q"], ch, row["answer"], client)
        v = parse_verdict(raw, len(ch))
        out.append({"id": row["id"], "q": row["q"], **v})
        mark = "BLOCK" if v["block"] else ("  ?  " if not v["parsed"] else "    ")
        print(f"{i:>2}/{len(rows)} {mark} {row['q'][:26]}")
    return out


def cmd_calibrate():
    """The judge against the manual annotation. Agreement per flag, separately.

    ⚠️ The labels in manual_review.md were assigned by an agent (Claude), not
    by a human. This is agreement between two models, not agreement with the
    truth. Written down explicitly so the two are not confused a month from now.

    The annotation was made in dedupe mode — we judge the same answers in the
    same mode, or there is nothing to compare.
    """
    manual = parse_manual()
    if not manual:
        sys.exit("manual_review.md is not filled in")

    path = report_path("dedupe")
    if not path.exists():
        sys.exit(f"{path.name} is needed (rename the old gen_report.json)")
    by_id = {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}
    rows = [by_id[i] for i in sorted(manual) if i in by_id]

    r = Reranked()
    with httpx.Client() as cl:
        res = run(rows, "dedupe", cl, r)

    # Mapping between the judge's flags and the manual labels
    pairs = {
        "欠落 ~ partial": (lambda v: bool(v["missing"]),
                          lambda m: m["substance"] == "partial"),
        "矛盾 ~ distortion": (lambda v: bool(v["contradiction"]),
                             lambda m: m["distortion"] == "yes"),
        "不可 ~ extra": (lambda v: bool(v["inapplicable"]),
                        lambda m: m["extra"] == "yes"),
    }
    print(f"\njudge: {JUDGE_MODEL}   questions: {len(res)}")
    print(f"unparsed: {sum(1 for v in res if not v['parsed'])}")
    for name, (jf, hf) in pairs.items():
        both = only_j = only_h = neither = 0
        for v in res:
            j, h = jf(v), hf(manual[v["id"]])
            both += j and h
            only_j += j and not h
            only_h += h and not j
            neither += not j and not h
        agree = (both + neither) / len(res)
        print(f"  {name:<20} agreement {agree:.2f}  "
              f"both {both} · judge only {only_j} · annotation only {only_h}")

    out = ROOT / "data" / f"verify_calibrate_{JUDGE_MODEL.replace(':', '-')}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}")


def cmd_full(mode=DEFAULT_MODE):
    path = report_path(mode)
    if not path.exists():
        sys.exit(f"{path.name} is needed — run generate.py --eval first")
    rows = json.loads(path.read_text(encoding="utf-8"))

    r = Reranked()
    with httpx.Client() as cl:
        res = run(rows, mode, cl, r)

    n = len(res)
    print(f"\njudge: {JUDGE_MODEL}   mode: {mode}   questions: {n}")
    print(f"unparsed:              {sum(1 for v in res if not v['parsed'])}")
    print(f"inapplicable citation: {sum(1 for v in res if v['inapplicable'])}")
    print(f"missing provision:     {sum(1 for v in res if v['missing'])}")
    print(f"contradiction:         {sum(1 for v in res if v['contradiction'])}")
    blocked = sum(1 for v in res if v["block"])
    print(f"BLOCKED:               {blocked} = {blocked / n:.2f}")

    out = ROOT / "data" / f"verify_{mode}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}")


# ────────────── structural completeness check ──────────────
# Four judges (8B ... 120B) failed 欠落: the best result was 3 of 4 with six
# false positives, and quality does not grow with size. So this class has to
# be caught structurally rather than by a model.
#
# A rule with no tunable number in it:
#   a paragraph counts as omitted if the answer does NOT use it while the
#   reranker scores it NO LOWER than the one that was used.
#
# No threshold -> nothing to fit on twenty data points.

USED_MIN = 10   # characters of common substring — the paragraph counts as used


def _used(paragraph, answer):
    """Was the paragraph used in the answer.

    Legal answers quote almost verbatim (六箇月間継続勤務し全労働日の八割以上出勤),
    so a long common substring is a reliable signal.
    """
    m = difflib.SequenceMatcher(None, paragraph, answer, autojunk=False)
    return m.find_longest_match(0, len(paragraph), 0, len(answer)).size >= USED_MIN


def structural_omission(question, chunks, answer, rr):
    """Articles where the answer took one paragraph while a neighbour scores no lower.

    ⚠️ This only makes sense in whole mode. In dedupe the context holds a
    single paragraph, so flagging "did not take 2項" would mean the model was
    never given it — the same mistake that was worked through on #22.
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
        sys.exit(f"{path.name} is needed")
    rows = json.loads(path.read_text(encoding="utf-8"))
    rr = Reranked()

    res = []
    for i, row in enumerate(rows, 1):
        ch = retrieve(rr, row["q"], mode)
        f = structural_omission(row["q"], ch, row["answer"], rr)
        res.append({"id": row["id"], "q": row["q"], "omission": f})
        print(f"{i:>2}/{len(rows)} {'FLAG' if f else '    '} {row['q'][:26]}")

    n = len(res)
    hit = [r for r in res if r["omission"]]
    print(f"\nmode: {mode}   questions: {n}")
    print(f"flagged: {len(hit)} = {len(hit) / n:.2f}")

    manual = parse_manual()
    if manual:
        common = [r for r in res if r["id"] in manual]
        both = sum(1 for r in common
                   if bool(r["omission"]) and manual[r["id"]]["substance"] == "partial")
        only_s = sum(1 for r in common
                     if r["omission"] and manual[r["id"]]["substance"] != "partial")
        only_m = sum(1 for r in common
                     if not r["omission"] and manual[r["id"]]["substance"] == "partial")
        print(f"against the annotation ({len(common)}): both {both} · structure only {only_s}"
              f" · annotation only {only_m}")

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
