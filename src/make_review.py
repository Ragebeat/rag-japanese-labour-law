"""A form for manual annotation of answers — the reference set for an LLM judge.

Why: #31 (the right article was in the context, the model did not take it) is
invisible to citation_validity, to the verdict and to the score gate. Cases
like that are only caught by a human. Twenty annotated questions become the
set on which an LLM judge is checked BEFORE it is trusted — otherwise the
judge is as unverified an instrument as HNSW with its default ef_search.

SAMPLING (the rule was written before generating):
    from the questions that hit — a systematic sample of every second one by
    id, first 20. Not random and not "the interesting ones": the golden set is
    ordered by statute, so a step over ids preserves domain coverage, while
    picking "interesting" cases would fit the calibration set to known problems.

THREE FLAGS, independent of one another:
    substance  — does it answer the question: full / partial / none
    distortion — does it distort the norm (wrong period, wrong amount, a duty
                 presented as a recommendation): yes / no
    extra      — does it drag in unrelated material from junk chunks: yes / no

Why three rather than one score: an answer can be substantively right and
still carry junk (#4, #5), or cite the article correctly and get the period
wrong. These are different defects with different costs; averaging them is
not allowed.

  uv run python src/make_review.py
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORT = ROOT / "data" / "gen_report_whole.json"
OUT = ROOT / "data" / "manual_review.md"

N = 20
STEP = 2

HEAD = """# Manual annotation of answers — the reference for the judge

Fill in every `grade:` line, replacing `?`. Touch nothing else —
the file is parsed with a regex.

- **substance** = `full` (answers the question) · `partial` (on topic, misses the point)
  · `none` (does not answer)
- **distortion** = `yes` / `no` — is the rule distorted: a wrong period or
  amount, a duty presented as a recommendation, a rule applied to the wrong people
- **extra** = `yes` / `no` — is unrelated material pulled in from noise chunks

Judge **the answer against the question**, not against golden. Golden is
narrow, and an answer from a 施行規則 can be correct.

If unsure between `full` and `partial`, choose `partial`.

---

"""


def main():
    if not REPORT.exists():
        sys.exit(f"no {REPORT.name} — run generate.py --eval first")
    if OUT.exists() and "--force" not in sys.argv:
        sys.exit(f"{OUT.name} already exists and holds annotation; overwriting would erase it.\n"
                 f"View: --parse · rebuild from scratch: --force")

    rows = [r for r in json.loads(REPORT.read_text(encoding="utf-8")) if r["hit"]]
    rows.sort(key=lambda r: r["id"])
    picked = rows[::STEP][:N]

    if len(picked) < N:
        print(f"⚠️ only {len(picked)} questions available at step {STEP}")

    parts = [HEAD]
    for r in picked:
        parts.append(f"## #{r['id']}  {r['q']}\n")
        parts.append("**Context:**\n")
        for i, c in enumerate(r["context"], 1):
            parts.append(f"{i}. `{c}`")
        parts.append("\n**Answer:**\n")
        parts.append("> " + r["answer"].replace("\n", "\n> "))
        parts.append("")
        parts.append("`grade: substance=?  distortion=?  extra=?`")
        parts.append("")
        parts.append("note:")
        parts.append("\n---\n")

    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"questions in the form: {len(picked)}  (id: "
          f"{', '.join(str(r['id']) for r in picked)})")
    print(f"→ {OUT}")


def parse():
    """Reads the filled-in form. Called from the future comparison with the judge."""
    text = OUT.read_text(encoding="utf-8")
    out = {}
    for block in text.split("## #")[1:]:
        qid = int(block.split()[0])
        m = re.search(r"grade:\s*substance=(\w+)\s+distortion=(\w+)\s+extra=(\w+)",
                      block)
        if m and "?" not in m.group(0):
            out[qid] = dict(substance=m.group(1), distortion=m.group(2),
                            extra=m.group(3))
    return out


if __name__ == "__main__":
    if "--parse" in sys.argv:
        g = parse()
        print(f"filled in: {len(g)}")
        for k, v in sorted(g.items()):
            print(f"  #{k}: {v}")
    else:
        main()
