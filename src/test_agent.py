"""Deterministic checks for the agent's loop detector and budget.

The detector never fired against the live model in session 1: the model
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
    print(f"{'✓' if ok else '✗'} {name}: {got}" + ("" if ok else f"  (ждали {want})"))
    return ok


def test_progress():
    r = []
    p = Progress()
    r.append(check("первая выдача — прогресс", p.update(A, 0.50), 0))
    r.append(check("та же выдача, тот же балл — застой", p.update(A, 0.50), 1))
    r.append(check(f"ещё раз — {STALL_MAX}, стоп", p.update(A, 0.50), STALL_MAX))

    p = Progress()
    p.update(A, 0.50)
    r.append(check("рост балла +0.01 (шум) — застой", p.update(A, 0.51), 1))
    r.append(check("рост балла +0.05 — прогресс, счётчик в ноль", p.update(A, 0.56), 0))

    p = Progress()
    p.update(A, 0.50)
    r.append(check("1 новая из 5 (20%) — застой", p.update(A4_1, 0.50), 1))
    p = Progress()
    p.update(A, 0.50)
    r.append(check("2 новые из 5 (40%) — прогресс", p.update(A3_2, 0.50), 0))

    p = Progress()
    p.update(A, 0.50)
    r.append(check("пустая выдача — застой", p.update(set(), None), 1))
    return r


def test_budget():
    # The shape of the real compound run: 519 -> 9521 after one round of
    # two sub-questions with whole articles.
    r = []
    b = Budget()
    b.charge({"prompt_eval_count": 519, "eval_count": 32})
    b.charge({"prompt_eval_count": 9521, "eval_count": 1100})
    r.append(check("после первого раунда — бюджета хватает", b.exhausted(), None))
    b.charge({"prompt_eval_count": 18600, "eval_count": 50})
    r.append(check(f"второй такой раунд — не влезает в окно {NUM_CTX}",
                   b.exhausted(), "context"))

    b = Budget()
    b.charge({"prompt_eval_count": 500, "eval_count": 30})
    b.charge({"prompt_eval_count": 3000, "eval_count": 30}, final=True)
    r.append(check("финальный ход не двигает рост промпта", b.growth, 0))
    return r


if __name__ == "__main__":
    results = test_progress() + test_budget()
    bad = results.count(False)
    print(f"\n{len(results) - bad}/{len(results)}")
    sys.exit(1 if bad else 0)
