"""Deterministic checks for the agent's loop detector and budget.

The detector never fired against the live model during development: the model
refused or answered before spinning. Provoking a 4B model into a loop to test
a comparison of sets is the wrong way round — the logic is plain code, so it
is tested as plain code, on hand-made inputs.

  uv run python src/test_agent.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from agent import NUM_CTX, STALL_MAX, Budget, Progress  # noqa: E402

A = {("L1", f"第{i}条") for i in range(1, 6)}          # five articles
A4_1 = {("L1", f"第{i}条") for i in range(1, 5)} | {("L2", "第1条")}   # 1 new of 5
A3_2 = {("L1", f"第{i}条") for i in range(1, 4)} | {("L2", "第1条"), ("L2", "第2条")}


def check(name, got, want):
    ok = got == want
    print(f"{'✓' if ok else '✗'} {name}: {got}" + ("" if ok else f"  (expected {want})"))
    return ok


def test_progress():
    r = []
    p = Progress()
    r.append(check("first result — progress", p.update(A, 0.50), 0))
    r.append(check("same result, same score — stall", p.update(A, 0.50), 1))
    r.append(check(f"once more — {STALL_MAX}, stop", p.update(A, 0.50), STALL_MAX))

    p = Progress()
    p.update(A, 0.50)
    r.append(check("score up +0.01 (noise) — stall", p.update(A, 0.51), 1))
    r.append(check("score up +0.05 — progress, counter reset", p.update(A, 0.56), 0))

    p = Progress()
    p.update(A, 0.50)
    r.append(check("1 new of 5 (20%) — stall", p.update(A4_1, 0.50), 1))
    p = Progress()
    p.update(A, 0.50)
    r.append(check("2 new of 5 (40%) — progress", p.update(A3_2, 0.50), 0))

    p = Progress()
    p.update(A, 0.50)
    r.append(check("empty result — stall", p.update(set(), None), 1))
    return r


def test_budget():
    # The shape of the real compound run: 519 -> 9521 after one round of
    # two sub-questions with whole articles.
    r = []
    b = Budget()
    b.charge({"prompt_eval_count": 519, "eval_count": 32})
    b.charge({"prompt_eval_count": 9521, "eval_count": 1100})
    r.append(check("after the first round — budget suffices", b.exhausted(), None))
    b.charge({"prompt_eval_count": 18600, "eval_count": 50})
    r.append(check(f"a second such round — does not fit the {NUM_CTX} window",
                   b.exhausted(), "context"))

    b = Budget()
    b.charge({"prompt_eval_count": 500, "eval_count": 30})
    b.charge({"prompt_eval_count": 3000, "eval_count": 30}, final=True)
    r.append(check("the final turn does not count as prompt growth", b.growth, 0))
    return r


if __name__ == "__main__":
    results = test_progress() + test_budget()
    bad = results.count(False)
    print(f"\n{len(results) - bad}/{len(results)}")
    sys.exit(1 if bad else 0)
