"""Deterministic checks for the agent's loop detector and budget.

The detector never fired against the live model during development: the model
refused or answered before spinning. Provoking a 4B model into a loop to test
a comparison of sets is the wrong way round: the logic is plain code, so it
is tested as plain code, on hand-made inputs.

  uv run pytest tests/test_agent.py
"""

from rag.agent.loop import NUM_CTX, STALL_MAX, Budget, Progress

A = {("L1", f"第{i}条") for i in range(1, 6)}          # five articles
A4_1 = {("L1", f"第{i}条") for i in range(1, 5)} | {("L2", "第1条")}   # 1 new of 5
A3_2 = {("L1", f"第{i}条") for i in range(1, 4)} | {("L2", "第1条"), ("L2", "第2条")}


def started(score=0.50):
    p = Progress()
    p.update(A, score)
    return p


def test_same_result_stalls_until_stop():
    p = Progress()
    assert p.update(A, 0.50) == 0            # first result: progress
    assert p.update(A, 0.50) == 1            # same result, same score: stall
    assert p.update(A, 0.50) == STALL_MAX    # once more: stop


def test_score_noise_is_a_stall_a_real_gain_is_progress():
    p = started()
    assert p.update(A, 0.51) == 1            # +0.01 is noise
    assert p.update(A, 0.56) == 0            # +0.05 resets the counter


def test_new_articles_threshold():
    assert started().update(A4_1, 0.50) == 1     # 1 new of 5 (20%): stall
    assert started().update(A3_2, 0.50) == 0     # 2 new of 5 (40%): progress


def test_empty_result_is_a_stall():
    assert started().update(set(), None) == 1


def test_budget_second_round_does_not_fit():
    # The shape of the real compound run: 519 -> 9521 after one round of
    # two sub-questions with whole articles.
    b = Budget()
    b.charge({"prompt_eval_count": 519, "eval_count": 32})
    b.charge({"prompt_eval_count": 9521, "eval_count": 1100})
    assert b.exhausted() is None
    b.charge({"prompt_eval_count": 18600, "eval_count": 50})
    assert b.exhausted() == "context", f"a second round must not fit the {NUM_CTX} window"


def test_final_turn_is_not_prompt_growth():
    b = Budget()
    b.charge({"prompt_eval_count": 500, "eval_count": 30})
    b.charge({"prompt_eval_count": 3000, "eval_count": 30}, final=True)
    assert b.growth == 0
