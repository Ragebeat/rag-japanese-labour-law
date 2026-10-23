"""Compare two eval runs: metrics side by side, then per-question regressions.

A mean hides what broke. On 29.09 the quote prompt left over-refusal and
refusal-on-miss unchanged, while #26 lost the key article (労契法 第十条) and
#1 lost its only quote — both found by accident, reading answers. This lists
them by question.

  uv run python src/compare.py                 # the last two runs in runs.jsonl
  uv run python src/compare.py -3 -1           # runs by index in runs.jsonl
  uv run python src/compare.py a.json b.json   # any two reports, logged or not
  uv run python src/compare.py --list          # the run log

A before B: A is the old run, B the new one. "Regression" = worse in B.

Quote fields are RE-CHECKED from the answers with the current check_quotes,
not read from the reports. First run of this script compared a report
scored by the old, buggy quote check (claim coverage 0.25) with one scored
by the fixed check — two definitions of one metric side by side. Only the
answers are comparable; the scoring is redone every time. Same caveat as
rescore_quotes.py: context is rebuilt from the article list, so for articles
over ARTICLE_CAP the check is slightly lenient.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from generate import (ROOT, RUNS_LOG, VERDICT_RE, _by_article, _fmt,  # noqa: E402
                      art_key, check_quotes, load_golden, run_metrics)
from rescore_quotes import rebuild  # noqa: E402

# Which way is better, per metric. None: informational only.
BETTER = {
    "recall": +1, "refusal_at_miss": +1, "citation_validity": +1,
    "claim_coverage": +1, "text_coverage": +1, "verbatim_coverage": +1, "para_claims": +1,
    "over_refusal": -1, "no_verdict": -1, "bad_refs": -1, "quote_bad": -1,
    "para_bad": -1, "num_unquoted_answers": -1, "quotes": None, "n": None,
}


def runs():
    if not RUNS_LOG.exists():
        return []
    return [json.loads(x) for x in RUNS_LOG.read_text(encoding="utf-8").splitlines() if x.strip()]


def resolve(arg, log):
    """A run index into runs.jsonl (-1, 0, 3) or a path to a report."""
    try:
        e = log[int(arg)]
        return ROOT / e["report"], e
    except ValueError:
        p = pathlib.Path(arg)
        return (p if p.is_absolute() else ROOT / p), None
    except IndexError:
        sys.exit(f"no run {arg} in runs.jsonl ({len(log)} in total)")


def describe(path, entry):
    if not entry:
        return f"{path.name}  (not from the log — configuration unknown)"
    return f"{entry['ts']} · {entry['label']} · fingerprint {entry['fingerprint']} · {entry['git']}"


def show_metrics(ma, mb):
    print(f"\n{'metric':<22} {'A':>8} {'B':>8}   Δ")
    for k, better in BETTER.items():
        a, b = ma.get(k), mb.get(k)
        if a is None and b is None:
            continue
        if a is None or b is None or a == b:
            mark = ""
        else:
            d = b - a
            mark = f"{d:+.3f}" if isinstance(d, float) else f"{d:+d}"
            if better:
                mark += "  ✓" if d * better > 0 else "  ✗"
        fa = "—" if a is None else a
        fb = "—" if b is None else b
        print(f"{k:<22} {fa!s:>8} {fb!s:>8}   {mark}")


def rescore(rows, arts):
    for r in rows:
        body = VERDICT_RE.sub("", r["answer"]).strip()
        r.update(check_quotes(body, rebuild(r["context"], arts)))
    return rows


def per_question(ra, rb):
    # art_key returns a SET of (article, sub) keys; fmt turns a set into a list.
    gold = {g["id"]: set(_fmt(set().union(*(art_key(t) for _, t in g["relevant"]))))
            for g in load_golden()}
    by_a = {r["id"]: r for r in ra}
    worse, better = [], []

    for rb_ in rb:
        i, a, b = rb_["id"], by_a.get(rb_["id"]), rb_
        if a is None:
            continue
        w, g = [], []

        if a["verdict"] != b["verdict"]:
            # A refusal is right on a retrieval miss and wrong on a hit.
            to_refusal = b["verdict"] == "no"
            (w if to_refusal == b["hit"] else g).append(
                f"verdict {a['verdict']} → {b['verdict']}"
                f" ({'article found' if b['hit'] else 'retrieval miss'})")

        ga, gb = set(a.get("cited", [])) & gold[i], set(b.get("cited", [])) & gold[i]
        if ga - gb:
            w.append(f"lost a golden source: {', '.join(sorted(ga - gb))}")
        if gb - ga:
            g.append(f"gained a golden source: {', '.join(sorted(gb - ga))}")

        qa, qb = a.get("claims_quoted"), b.get("claims_quoted")
        if qa is not None and qb is not None:
            if qa > 0 and qb == 0:
                w.append(f"quotes disappeared ({qa} → 0)")
            elif qa == 0 and qb > 0:
                g.append(f"quotes appeared (0 → {qb})")

        for key, label in (("quote_bad", "distorted quote"),
                           ("para_bad", "quote not in the cited 項"),
                           ("bad_arts", "wrong article number")):
            na, nb = len(a.get(key, [])), len(b.get(key, []))
            if nb > na:
                w.append(f"{label}: {na} → {nb}")
            elif nb < na:
                g.append(f"{label}: {na} → {nb}")

        if w:
            worse.append((i, b["q"], w))
        if g:
            better.append((i, b["q"], g))

    for title, items in (("Regressions", worse), ("Improvements", better)):
        print(f"\n── {title} ({len(items)}) ──")
        for i, q, notes in items:
            print(f"  #{i:<3} {q[:26]}")
            for n in notes:
                print(f"         {n}")


def main():
    args = sys.argv[1:]
    log = runs()
    if "--list" in args:
        for k, e in enumerate(log):
            m = e["metrics"]
            print(f"{k:>3}  {e['ts']}  {e['label']:<7} {e['fingerprint']}  {e['git']!s:<14}"
                  f" coverage {m.get('claim_coverage')}  para_bad {m.get('para_bad')}"
                  f"  over-refusal {m.get('over_refusal')}")
        return

    if not args:
        if len(log) < 2:
            sys.exit("fewer than two runs in runs.jsonl — pass report paths instead")
        args = ["-2", "-1"]
    if len(args) != 2:
        sys.exit(__doc__)

    (pa, ea), (pb, eb) = resolve(args[0], log), resolve(args[1], log)
    arts = _by_article()
    ra = rescore(json.loads(pa.read_text(encoding="utf-8")), arts)
    rb = rescore(json.loads(pb.read_text(encoding="utf-8")), arts)

    print(f"A: {describe(pa, ea)}")
    print(f"B: {describe(pb, eb)}")
    if ea and eb and ea["fingerprint"] == eb["fingerprint"]:
        print("⚠ same configuration — every difference below is run-to-run variance")
    elif ea and eb:
        changed = [k for k in ea["config"] if ea["config"][k] != eb["config"].get(k)]
        print(f"configuration changed: {', '.join(changed)}")

    # Metrics are recomputed from the rows, not taken from the log: the same
    # definitions apply to a report that was never logged.
    show_metrics(run_metrics(ra), run_metrics(rb))
    per_question(ra, rb)


if __name__ == "__main__":
    main()
