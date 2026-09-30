"""Agent over the labour-law RAG. Phase 5, session 1: a bare ReAct loop.

ReAct (Reasoning + Acting): the model alternates thinking and calling a tool,
and decides by itself how many steps it needs.

Where the line between code and model runs — one rule from session 1:
whatever can be written as a rule is code. The model keeps only the semantic
rewording of the query.

  code:   the score threshold, widening core -> full corpus, loop detection,
          iterations / tokens / context / wall time, the final turn when a
          budget runs out, the deterministic fallback when the model is gone
  model:  whether to search, what to search for, when it has enough

  uv run python src/agent.py "育休中に残業させられる？"
  uv run python src/agent.py --trace "育休と残業、両方教えて"
"""

import concurrent.futures as cf
import json
import pathlib
import statistics
import sys
import time

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from generate import (KW, MODEL, POOL_MULT, URL, _by_article,  # noqa: E402
                      dedupe_by_article, whole_article)
from query_expand import expand  # noqa: E402
from query_llm import expand_combo  # noqa: E402
from rerank import Reranked  # noqa: E402
import tracing  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
GATE = ROOT / "data" / "score_gate.json"


# ─────────────────────────────── the threshold ───────────────────────────────

def measured_threshold():
    """The same product decision as in score_gate.py: the 5th percentile of
    top-1 scores in the group WITH a source. Read from the measurement, not
    typed in — the 0.5 / 0.1 of the first tool description were by eye.
    """
    rows = json.loads(GATE.read_text(encoding="utf-8"))
    pos = [r["top1_full"] for r in rows if r["hit"]]
    return statistics.quantiles(sorted(pos), n=100)[4]


THRESHOLD = measured_threshold()

# ──────────────────────────────── the limits ─────────────────────────────────
# Each one catches a failure the others cannot. See the phase 5 notes.

MAX_ITER = 8           # a fuse against infinity, not a budget
TOKEN_CAP = 60_000     # cumulative across every model call, history included
NUM_CTX = 32_768       # the window we ask Ollama for; see the ⚠️ in chat()
ANSWER_MAX = 800       # num_predict on every call
WALL_CAP = 240.0       # seconds, the user's patience
FINAL_TIME = 60.0      # held back for the final turn
TOOL_TIMEOUT = 90.0    # per tool call, shorter than the loop. Was 45: the weak path
                       # is two searches (core + widened) plus LLM expansion, and
                       # under MCP one such call ran past 45 s (13 s on the LLM alone)
LLM_TIMEOUT = 90.0     # per model call

NOISE = 0.02           # score difference below this is noise (phase 4 rule)
NEW_SHARE = 0.4        # at least 2 of 5 articles must be new to count as progress
STALL_MAX = 2          # consecutive calls without progress = spinning


# ──────────────────────────────── the tool ───────────────────────────────────

TOOL_NAME = "labor_laws_rag"

# No thresholds, no "rephrase with statutory wording", no core_only:
#  - the threshold lives in code and comes back as `status`
#  - statutory rewording is already done inside by the dictionary + LLM;
#    asking the model to do it too makes a miss impossible to attribute
#  - widening the corpus is a rule, so the code does it
TOOL = {"type": "function", "function": {
    "name": TOOL_NAME,
    "description": (
        "Search the Japanese labour law corpus and return statutory articles "
        "with their full text.\n\n"
        "Corpus: Japanese labour statutes and their ordinances (労働基準法, "
        "育児介護休業法, 労働施策総合推進法, 派遣法, 雇用保険法, 労災保険法 and "
        "others). Nothing outside Japanese labour law is in it — no tax, no "
        "immigration, no company law.\n\n"
        "WHEN TO CALL: for any question about Japanese employment, wages, "
        "working hours, leave, dismissal, harassment or workplace safety. "
        "Call it even when you believe you know the answer. NEVER answer such "
        "questions from your own knowledge.\n\n"
        "Pass EVERY sub-question of the user's question as a separate entry "
        "in `queries`, one topic each. A question about two things is two "
        "entries, never one joined string.\n\n"
        "Each entry in `results` has its own `status`. Act on it per entry: "
        "`ok` — the articles are sources; `weak` — nothing matched, the "
        "candidates are NOT sources; `error` — the search itself failed, do "
        "not retry."
    ),
    "parameters": {"type": "object", "properties": {
        "queries": {"type": "array", "minItems": 1, "maxItems": 4,
                    "items": {"type": "string"},
                    "description": "One Japanese query per sub-question, one "
                                   "topic each. Colloquial phrasing is fine."},
        "k": {"type": "integer", "minimum": 1, "maximum": 10,
              "description": "How many articles. 5 unless the question "
                             "genuinely spans several institutions."},
    }, "required": ["queries"]},
}}

MAX_QUERIES = 4


def _key(c):
    return (c["law_id"], c["article_title"])


def _top(hits):
    return max((c.get("rerank_score", 0.0) for c in hits), default=None)


def _expansion(query):
    """Who rewrote the query and into what — for the trace only.

    combo is an EXCLUSIVE fallback: the LLM runs only when the dictionary
    returns nothing. A partial dictionary hit therefore switches the LLM off.
    Without this line a miss cannot be attributed. The LLM answer is cached
    to disk by the search that just ran, so this costs no second call.
    """
    _, d = expand(query, narrow=True)
    if d:
        return "словарь", d
    _, t = expand_combo(query)
    return ("LLM", t) if t else ("—", [])


class Tool:
    """The RAG wrapped as a tool. Returns (payload for the model, keys, top).

    keys and top are for the loop detector — the model never sees them as such.
    """

    def __init__(self):
        self.rr = Reranked()
        self.arts = _by_article()
        # One worker: a timed-out search keeps running in its thread. In the
        # agent loop that was harmless — a timeout stops the loop. Under MCP
        # there is no loop of ours: the client may simply call again, queue
        # behind the zombie search and time out too. So the guard lives here,
        # in the tool, where it travels with it: see _busy in _one().
        self.pool = cf.ThreadPoolExecutor(max_workers=1)
        self._busy = None
        # Warm-up. The reranker and the embedder load lazily on the first
        # search — longer than TOOL_TIMEOUT on a cold start. The first run
        # died on exactly that: a cold start reported as a search failure.
        # api.py warms up in lifespan for the same reason.
        self._search("有給休暇", 1, core_only=True)

    def _search(self, query, k, core_only):
        # Same assembly as generate.retrieve in `whole` mode, plus core_only,
        # which retrieve hard-codes to True.
        pool = k * POOL_MULT
        with tracing.span("retrieve", "RETRIEVER", input=query,
                          attrs={"retrieval.core_only": core_only, "retrieval.k": k}) as sp:
            cands = self.rr.search(query, k=pool, core_only=core_only, pool=pool, **KW)
            out = [whole_article(c, self.arts) for c in dedupe_by_article(cands, k)]
            tracing.documents(sp, out)
            return out

    def _run(self, query, k):
        hits = self._search(query, k, core_only=True)
        top = _top(hits)
        if top is None or top < THRESHOLD:
            # A rule, not the model's decision: the core missed -> widen.
            return self._search(query, k, core_only=False), True
        return hits, False

    def _one(self, query, k):
        """One sub-query -> (result entry, keys, top, expansion). The entry
        carries its own status: the first compound question showed that a
        single `ok` on a joined string hides an empty half.
        """
        if self._busy is not None and not self._busy.done():
            return ({"query": query, **_error(
                "A previous search is still running (an earlier call timed "
                "out and has not finished). Try again in a minute.")},
                    set(), None, None)
        # carry(): the search runs in a worker thread, and the trace context
        # does not follow it there by itself — without this every search
        # would be a separate orphan trace instead of a child of this call.
        fut = self.pool.submit(tracing.carry(self._run), query, k)
        self._busy = fut
        try:
            hits, widened = fut.result(timeout=TOOL_TIMEOUT)
        except cf.TimeoutError:
            return ({"query": query,
                     **_error(f"Search timed out after {TOOL_TIMEOUT:.0f}s.")},
                    set(), None, None)
        except Exception as e:                      # noqa: BLE001
            return ({"query": query,
                     **_error(f"Search failed: {type(e).__name__}: {e}")},
                    set(), None, None)

        keys, top = {_key(c) for c in hits}, _top(hits)
        exp = _expansion(query)

        if top is not None and top >= THRESHOLD:
            return {
                "query": query,
                "status": "ok",
                "widened": widened,   # found outside the 35 core statutes
                "articles": [{
                    "law": c["law_title"],
                    "article": c["article_title"],
                    "caption": c["article_caption"] or "",
                    "text": c["text"].split("\n", 1)[-1],
                } for c in hits],
            }, keys, top, exp

        # weak: candidates WITHOUT text. Not an instruction "do not answer from
        # them" — the absence of material to answer from.
        return {
            "query": query,
            "status": "weak",
            "widened": widened,
            "message": "No article in the corpus matches this query. The "
                       "candidates below are topically adjacent, NOT sources.",
            "candidates": [{"law": c["law_title"], "article": c["article_title"]}
                           for c in hits],
            "next": ["If the question can be reframed as a different legal "
                     "concept, search for that concept once.",
                     "Otherwise tell the user the corpus has no provision on this."],
        }, keys, top, exp

    def __call__(self, queries, k=5):
        with tracing.span("labor_laws_rag", "TOOL", input=queries,
                          attrs={"tool.name": TOOL_NAME, "retrieval.k": k}) as sp:
            payload, keys, tops, exps = self._call(queries, k)
            tracing.set_attrs(sp, {"output.value": [r.get("status") for r in payload["results"]]})
            return payload, keys, tops, exps

    def _call(self, queries, k=5):
        """-> ({"results": [...]}, union of keys, tops, expansions — the last
        two aligned with results; neither is shown to the model)."""
        try:
            k = max(1, min(int(k or 5), 10))
        except (TypeError, ValueError):
            k = 5
        if isinstance(queries, str):
            queries = [queries]
        queries = [str(q).strip() for q in (queries or []) if str(q).strip()]
        queries = queries[:MAX_QUERIES]
        if not queries:
            return ({"results": [{"query": "", **_error("Empty query.")}]},
                    set(), [None], [None])

        results, keys, tops, exps = [], set(), [], []
        for q in queries:
            with tracing.span("sub_query", "CHAIN", input=q) as sp:
                r, kk, t, e = self._one(q, k)
                tracing.set_attrs(sp, {
                    "output.value": r["status"], "rag.status": r["status"],
                    "rag.top": t, "rag.widened": r.get("widened"),
                    "rag.expansion.source": e[0] if e else None,
                    "rag.expansion.terms": list(e[1]) if e else None})
            results.append(r)
            keys |= kk
            tops.append(t)
            exps.append(e)
            if r["status"] == "error":
                break   # infrastructure is down: the rest would fail the same way
        return {"results": results}, keys, tops, exps


def _error(message):
    # The one status after which the right move is to stop, not to try again.
    return {"status": "error", "message": message, "next": []}


# ──────────────────────────── the loop detector ──────────────────────────────

class Progress:
    """Looks at what came back, not at what was asked.

    Query texts always differ a little — the model reshuffles words — so a
    detector on queries either never fires or cuts off a legitimate
    refinement. The set of (law_id, article) does not lie.

    Progress = enough new articles OR the best score rose beyond noise.
    """

    def __init__(self):
        self.seen, self.best, self.stalls = set(), 0.0, 0

    def update(self, keys, top):
        fresh = bool(keys) and len(keys - self.seen) / len(keys) >= NEW_SHARE
        grew = top is not None and top > self.best + NOISE
        self.seen |= keys
        if top is not None:
            self.best = max(self.best, top)
        self.stalls = 0 if (fresh or grew) else self.stalls + 1
        return self.stalls


# ──────────────────────────────── the budget ─────────────────────────────────

class Budget:
    """Stops BEFORE a call that could not be followed by the final turn.

    Next prompt ≈ last prompt + the largest growth seen so far (a tool result
    with whole articles is the biggest step). The check requires room for one
    more ordinary step AND the final turn after it — otherwise stop now,
    while the final answer still fits.
    """

    def __init__(self):
        self.t0 = time.monotonic()
        self.iters = self.tokens = self.last_prompt = self.growth = 0
        self.prompts = []   # for the sanity check in the trace

    def charge(self, resp, final=False):
        p = resp.get("prompt_eval_count", 0)
        self.tokens += p + resp.get("eval_count", 0)
        self.iters += 1
        if final:
            # No tool schema in the final prompt, so it is ~150 tokens
            # shorter by construction. Not comparable with the other calls:
            # the first run's "non-monotonic" alarm was this, not the cache.
            return
        self.prompts.append(p)
        if self.last_prompt:
            self.growth = max(self.growth, p - self.last_prompt)
        self.last_prompt = p

    def elapsed(self):
        return time.monotonic() - self.t0

    def exhausted(self):
        if self.iters >= MAX_ITER:
            return "iterations"
        if self.elapsed() > WALL_CAP - FINAL_TIME:
            return "time"
        step = self.last_prompt + self.growth + ANSWER_MAX
        final = self.last_prompt + 2 * self.growth + ANSWER_MAX
        if final > NUM_CTX:
            return "context"
        if self.tokens + step + final > TOKEN_CAP:
            return "tokens"
        return None


# ──────────────────────────────── the model ──────────────────────────────────

SYSTEM = """あなたは日本の労働法令について答えるアシスタントです。

1. 労働法令に関する質問には、必ず先に labor_laws_rag で検索する。自分の知識だけで答えない
2. 複合的な質問は、論点ごとに queries の別々の要素に分けて検索する。論点を一つの文字列につなげない
3. 回答の根拠は status が ok の検索結果の条文だけ。status が weak の候補は根拠ではない。status は論点ごとに判断する
4. 根拠の条文は「◯◯法 第N条」の形で必ず示す
5. 数字・日数・金額は条文の通りに引用する。推測しない
6. 該当する条文が見つからない論点は、見つからなかったと書く"""

NUDGE = "まず labor_laws_rag で検索してから答えてください。"

# The code decides THAT a second attempt happens; the model decides WHAT to
# search. On the tax question the model skipped the reframing suggested in
# `next` and refused — while 労基法 第24条 was in the corpus all along.
# No example concept here: an example in a prompt becomes a template (phase 4).
REFRAME = """次の論点は該当する条文が見つかりませんでした: {qs}
質問の背景にある別の法的概念で言い換え、一度だけ再検索してください。
同じ言葉の並べ替えではなく、別の制度・原則として捉え直すこと。
それでも見つからなければ、見つからなかったと答えてください。"""

FINAL_BASE = """これまでに status: ok で得た条文だけを根拠に答えてください。自分の知識で補わないこと。
最後の行に「未確認: 」として、確認できなかった論点を書くこと（なければ「未確認: なし」）。"""

# The reason changes what the user must be told. A failed search is NOT
# "the corpus has no provision": the first run said exactly that after a
# timeout, and the user would have left believing the law is silent.
FINAL_HEAD = {
    "tool_error": "【検索システムの障害】検索が失敗しました。条文が存在しないのではなく、"
                  "検索できなかったことを回答の冒頭で明記すること。",
    "loop": "【検索終了】同じ結果が続いたため、これ以上検索しません。",
}
FINAL_DEFAULT = "【検索終了: 予算上限】これ以上は検索できません。"


def chat(http, messages, tools=True):
    # ⚠️ num_ctx is set explicitly. Ollama's default window is small, and a
    # prompt that does not fit is trimmed SILENTLY — no error, the model just
    # never sees the start of the history. With whole articles piling up in
    # the history that would happen within two or three steps.
    body = {"model": MODEL, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 0, "num_predict": ANSWER_MAX,
                        "num_ctx": NUM_CTX}}
    if tools:
        body["tools"] = [TOOL]
    with tracing.span("ollama.chat", "LLM") as sp:
        tracing.llm_request(sp, MODEL, messages, body["options"], body.get("tools"))
        r = http.post(URL, json=body, timeout=LLM_TIMEOUT)
        r.raise_for_status()
        resp = r.json()
        tracing.llm_response(sp, resp)
        return resp


def fallback(reason, found):
    """No model at all. A deterministic answer is better than an exception."""
    arts = "\n".join(f"- {law} {art}" for law, art in found) or "- なし"
    return (f"回答を生成できませんでした（{reason}）。\n"
            f"検索で見つかった条文:\n{arts}")


# ──────────────────────────────── the loop ───────────────────────────────────

def run(question, tool, trace=False):
    """One question = one trace: the AGENT span is the root, every model
    call and tool call nests under it."""
    with tracing.span("agent", "AGENT", input=question) as sp:
        res = _run(question, tool, trace)
        tracing.set_attrs(sp, {
            "output.value": res["answer"], "agent.stop": res["stop"],
            "agent.iterations": res["iterations"], "agent.tokens": res["tokens"],
            "agent.found": [f"{law} {art}" for law, art in res["found"]]})
        return res


def _run(question, tool, trace=False):
    budget, prog = Budget(), Progress()
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}]
    found = []            # (law, article) from `ok` results, in order, no repeats
    called = nudged = reframed = False
    stop = None

    def log(s):
        if trace:
            print(f"  {budget.elapsed():6.1f}s  {s}")

    with httpx.Client() as http:
        while True:
            stop = budget.exhausted()
            if stop:
                break
            try:
                resp = chat(http, msgs)
            except Exception as e:                  # noqa: BLE001
                log(f"✗ модель: {type(e).__name__}: {e}")
                stop = "llm_error"
                break
            budget.charge(resp)
            msg = resp["message"]
            msgs.append(msg)
            calls = msg.get("tool_calls") or []
            log(f"#{budget.iters} prompt={budget.last_prompt} "
                f"всего={budget.tokens} вызовов={len(calls)}")

            if not calls:
                # Answering without a single search breaks rule 1. One nudge,
                # enforced by code; a second refusal is recorded, not fought.
                if not called and not nudged:
                    nudged = True
                    msgs.append({"role": "user", "content": NUDGE})
                    log("↻ ответ без поиска — толчок")
                    continue
                return _result(msg.get("content", "").strip(), "answered",
                               budget, found, ungrounded=not called)

            weak_qs = []
            for call in calls:
                fn = call.get("function", {})
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if fn.get("name") != TOOL_NAME:
                    payload = {"results": [{"query": "", **_error(
                        f"Unknown tool {fn.get('name')!r}.")}]}
                    keys, tops, exps = set(), [None], [None]
                else:
                    called = True
                    # `query` tolerated for a model that ignores the schema
                    payload, keys, tops, exps = tool(
                        args.get("queries") or args.get("query", ""),
                        args.get("k", 5))
                msgs.append({"role": "tool", "tool_name": fn.get("name", ""),
                             "content": json.dumps(payload, ensure_ascii=False)})

                errored = False
                for r, t, e in zip(payload["results"], tops, exps):
                    top_s = f"{t:.3f}" if t is not None else "—"
                    log(f"  → {r['query']!r} | {r['status']} top={top_s}"
                        f"{' widened' if r.get('widened') else ''}")
                    if e:
                        log(f"      расширение ({e[0]}): {e[1]}")
                    if r["status"] == "ok":
                        for a in r["articles"]:
                            if (a["law"], a["article"]) not in found:
                                found.append((a["law"], a["article"]))
                    if r["status"] == "error":
                        errored = True
                        log(f"  ✗ {r['message']}")
                    if r["status"] == "weak":
                        weak_qs.append(r["query"])

                top = max((t for t in tops if t is not None), default=None)
                if errored:
                    stop = "tool_error"      # code decides; the model is not asked
                elif prog.update(keys, top) >= STALL_MAX:
                    stop = "loop"
                    log(f"  ⟲ {STALL_MAX} вызова без прогресса")
            if stop:
                break
            # One reframing attempt, once per question. After it the model is
            # free to refuse; further retries are the loop detector's business.
            if weak_qs and not reframed:
                reframed = True
                msgs.append({"role": "user",
                             "content": REFRAME.format(qs="、".join(weak_qs))})
                log("↻ weak — толчок на переформулировку")

        # ── the final turn: exhausting a budget is a last move, not an exception
        log(f"■ стоп: {stop} → финальный ход")
        if stop == "llm_error":
            return _result(fallback(stop, found), stop, budget, found)
        try:
            head = FINAL_HEAD.get(stop, FINAL_DEFAULT)
            resp = chat(http, msgs + [{"role": "user",
                                       "content": f"{head}\n{FINAL_BASE}"}],
                        tools=False)
            budget.charge(resp, final=True)
            text = resp["message"].get("content", "").strip()
        except Exception as e:                      # noqa: BLE001
            log(f"✗ финальный ход: {type(e).__name__}: {e}")
            text = fallback(stop, found)
        return _result(text, stop, budget, found)


def _result(answer, stop, budget, found, ungrounded=False):
    return {"answer": answer, "stop": stop, "iterations": budget.iters,
            "tokens": budget.tokens, "seconds": round(budget.elapsed(), 1),
            "found": found, "ungrounded": ungrounded, "prompts": budget.prompts}


def main():
    args = sys.argv[1:]
    trace = "--trace" in args
    q = " ".join(a for a in args if not a.startswith("--")) or \
        "育休中に残業させられる？"

    print(f"порог (5-й перцентиль, score_gate.json): {THRESHOLD:.3f}")
    tracing.setup()
    tool = Tool()
    print(f"вопрос: {q}\n")
    res = run(q, tool, trace=trace)

    print("\n— ответ —")
    print(res["answer"])
    print("\n— итог —")
    print(f"стоп: {res['stop']}  итераций: {res['iterations']}  "
          f"токенов: {res['tokens']}  время: {res['seconds']}s")
    if res["ungrounded"]:
        print("⚠ ответ без единого поиска")
    for law, art in res["found"]:
        print(f"  • {law} {art}")

    # Sanity check, the phase 4 lesson: make sure the counter measures what we
    # think it measures. The history only grows, so prompt_eval_count must too.
    # If it drops, Ollama is counting only the uncached part of the prompt,
    # and the token budget above is reading the wrong number.
    p = res["prompts"]
    if len(p) < 2:
        print(f"prompt_eval_count: {p} — меньше двух вызовов с тулзами, проверка не информативна")
    elif any(b < a for a, b in zip(p, p[1:])):
        print(f"⚠ prompt_eval_count не монотонен: {p} — бюджет токенов врёт")
    else:
        print(f"prompt_eval_count монотонен: {p}")


if __name__ == "__main__":
    main()
