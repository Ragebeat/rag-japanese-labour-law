"""The same agent as agent.py, as a LangGraph state graph. Phase 5, sessions 2 and 4.

LangGraph describes an agent as a graph: nodes do the work, edges decide what
runs next, and one state object travels through. agent.py already WAS such a
graph — its transitions were `if`s inside a `while`. Here they have names.

Session 2 — what the move to a graph forced:

1. STATE IS DATA. A checkpointer saves the state after every node, so the
   budget and the loop detector live in plain dicts, loaded into the same
   tested classes (Budget, Progress) inside a node and written back.
2. HUMAN WAIT IS NOT BUDGET. The time budget stores seconds SPENT, not a start
   time — a graph paused for an hour must not wake up out of time.
3. A NODE WITH interrupt() RUNS TWICE. On resume the node re-executes from the
   top. Nothing before interrupt() may have side effects — or cost much.

Session 4 — memory, three kinds:

SHORT-TERM = the thread. Several questions in one conversation share the
message history (thread_id). The system prompt is rebuilt every turn, so it
always carries the current long-term facts.

COMPACTION = keep conclusions, drop raw material (the session 1 rule). After
every answer the full article texts are stripped from old tool messages; the
(law, article) registry and the answers stay. One round of two sub-questions
was +9 000 tokens — without this, the second question would not fit.

LONG-TERM = a store across threads, and here the design decisions matter more
than the code:
  - ALLOW-LIST, not deny-list. Only employment type, company size and
    prefecture can ever be stored. Pregnancy, health, disputes cannot be
    proposed at all — there is no field for them. A deny-list of "sensitive
    words" would leak the first phrasing nobody anticipated.
  - THE MODEL PROPOSES, CODE VERIFIES, THE HUMAN DECIDES. gemma extracts
    candidates; code keeps only values literally present in what the user
    said (no inference — "asks about 育休, so probably..." is rejected);
    the user confirms through interrupt().
  - EVERY FACT CARRIES A DATE. Facts go stale; after 90 days code marks them
    for re-confirmation instead of trusting them.
  - THE EXPENSIVE CALL SITS OUTSIDE THE interrupt() NODE. Extraction is in
    `propose`, the pause is in `confirm_memory`: otherwise the LLM call would
    run twice (point 3 above).
  - /forget DELETES. Not a flag, not "used to be" — the file loses the lines.

  uv run python src/agent_graph.py --trace
  uv run python src/agent_graph.py --trace --hitl "会社が住民税を給料から天引きするのは合法？"
  uv run python src/agent_graph.py --mermaid
"""

import json
import pathlib
import sys
import time
import unicodedata
import uuid
from datetime import date
from typing import TypedDict

import httpx
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from agent import (FINAL_BASE, FINAL_DEFAULT, FINAL_HEAD, LLM_TIMEOUT,  # noqa: E402
                   MODEL, NUDGE, REFRAME, STALL_MAX, SYSTEM, THRESHOLD,
                   TOOL_NAME, URL, Budget, Progress, Tool, _error, chat,
                   fallback)

ROOT = pathlib.Path(__file__).resolve().parent.parent
MEMORY_FILE = ROOT / "data" / "user_memory.json"   # in .gitignore: personal data
NS = ("users", "local", "profile")
STALE_DAYS = 90


class State(TypedDict, total=False):
    messages: list
    question: str
    human_notes: list    # what the human typed at ask_human — also user speech
    found: list          # [law, article] from `ok` results
    called: bool
    nudged: bool
    reframed: bool
    weak_qs: list
    stop: str | None
    answer: str
    ungrounded: bool
    budget: dict | None
    progress: dict | None
    proposal: dict | None


# ─────────────────────── state <-> the tested objects ────────────────────────

def load_budget(d):
    b = Budget()
    if d:
        b.iters, b.tokens = d["iters"], d["tokens"]
        b.last_prompt, b.growth = d["last_prompt"], d["growth"]
        b.prompts = list(d["prompts"])
        b.t0 = time.monotonic() - d["spent"]     # see point 2 in the docstring
    return b


def dump_budget(b):
    return {"iters": b.iters, "tokens": b.tokens, "last_prompt": b.last_prompt,
            "growth": b.growth, "prompts": b.prompts, "spent": b.elapsed()}


def load_progress(d):
    p = Progress()
    if d:
        p.seen = {tuple(k) for k in d["seen"]}
        p.best, p.stalls = d["best"], d["stalls"]
    return p


def dump_progress(p):
    return {"seen": sorted(list(k) for k in p.seen),
            "best": p.best, "stalls": p.stalls}


# ───────────────────────────── long-term memory ──────────────────────────────

# The allow-list. Adding a field here is a privacy decision, not a code change.
LABELS = {"employment_type": "雇用形態", "company_size": "会社の従業員数",
          "prefecture": "勤務地（都道府県）"}
EMPLOYMENT = {"正社員": ["正社員"], "パート": ["パート"],
              "アルバイト": ["アルバイト", "バイト"], "契約社員": ["契約社員"],
              "派遣": ["派遣"]}

EXTRACT = """ユーザーの発言から、本人が明言した事実だけを抽出してください。
推測はしない。発言に書かれていないものは必ず null にする。
- employment_type: 正社員 / パート / アルバイト / 契約社員 / 派遣 のいずれか
- company_size: 勤務先の従業員数（整数）
- prefecture: 勤務地の都道府県"""

# The schema allows null for every field — without a slot for "not said" the
# model invents a value (phase 3).
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "employment_type": {"type": ["string", "null"]},
        "company_size": {"type": ["integer", "null"]},
        "prefecture": {"type": ["string", "null"]},
    },
    "required": ["employment_type", "company_size", "prefecture"],
}


def load_store():
    store = InMemoryStore()
    if MEMORY_FILE.exists():
        for k, v in json.loads(MEMORY_FILE.read_text(encoding="utf-8")).items():
            store.put(NS, k, v)
    return store


def profile(store):
    return {it.key: it.value for it in store.search(NS)}


def save_store(store):
    MEMORY_FILE.write_text(json.dumps(profile(store), ensure_ascii=False,
                                      indent=2), encoding="utf-8")


def system_prompt(store):
    """The system prompt with the long-term facts, rebuilt every turn.

    Staleness is decided by code, not left to the model: a fact older than
    STALE_DAYS is marked for re-confirmation.
    """
    prof = profile(store)
    if not prof:
        return SYSTEM
    today, lines = date.today(), []
    for k, v in prof.items():
        age = (today - date.fromisoformat(v["stated_at"])).days
        stale = "（古い情報・使う前に本人に確認すること）" if age > STALE_DAYS else ""
        lines.append(f"- {LABELS[k]}: {v['value']}"
                     f"（{v['stated_at']} に本人が述べた）{stale}")
    return (SYSTEM + "\n\n【このユーザーについて本人が以前述べた事実】\n"
            + "\n".join(lines) + "\n回答に関係する場合のみ使うこと。")


def extract_facts(http, text):
    body = {"model": MODEL, "stream": False, "think": False,
            "format": EXTRACT_SCHEMA, "options": {"temperature": 0},
            "messages": [{"role": "system", "content": EXTRACT},
                         {"role": "user", "content": text}]}
    try:
        r = http.post(URL, json=body, timeout=LLM_TIMEOUT)
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])
    except Exception:                               # noqa: BLE001
        return {}


def verify_stated(facts, text):
    """Keeps only values that literally occur in what the user said.

    The model is asked not to infer; this makes it impossible. "三十人" in
    kanji numerals is rejected too — a missed fact is cheap, a stored guess
    is not.
    """
    t = unicodedata.normalize("NFKC", text)
    out = {}
    et = facts.get("employment_type")
    if et in EMPLOYMENT and any(w in t for w in EMPLOYMENT[et]):
        out["employment_type"] = et
    cs = facts.get("company_size")
    if isinstance(cs, int) and cs > 0 and str(cs) in t:
        out["company_size"] = cs
    pf = facts.get("prefecture")
    if isinstance(pf, str) and pf.rstrip("都道府県") and pf.rstrip("都道府県") in t:
        out["prefecture"] = pf
    return out


# ─────────────────────────────── the graph ───────────────────────────────────

def build(tool, http, store, hitl=False, trace=False):
    t_start = time.monotonic()

    def log(s):
        if trace:
            print(f"  {time.monotonic() - t_start:6.1f}s  {s}")

    # ── nodes: the search loop (sessions 1–2) ───────────────────────────────

    def model(s):
        b = load_budget(s.get("budget"))
        stop = b.exhausted()
        if stop:
            return {"stop": stop, "budget": dump_budget(b)}
        try:
            resp = chat(http, s["messages"])
        except Exception as e:                      # noqa: BLE001
            log(f"✗ модель: {type(e).__name__}: {e}")
            return {"stop": "llm_error", "budget": dump_budget(b)}
        b.charge(resp)
        msg = resp["message"]
        log(f"[model] #{b.iters} prompt={b.last_prompt} всего={b.tokens} "
            f"вызовов={len(msg.get('tool_calls') or [])}")
        return {"messages": s["messages"] + [msg], "budget": dump_budget(b)}

    def tools(s):
        # Every node that takes time loads and dumps the budget. A node that
        # skipped it would silently drop its seconds from the time budget.
        b = load_budget(s["budget"])
        p = load_progress(s.get("progress"))
        msgs = list(s["messages"])
        found = [list(x) for x in s.get("found", [])]
        called, weak_qs, stop = s.get("called", False), [], None

        for call in msgs[-1].get("tool_calls") or []:
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
                payload, keys, tops, exps = tool(
                    args.get("queries") or args.get("query", ""), args.get("k", 5))
            msgs.append({"role": "tool", "tool_name": fn.get("name", ""),
                         "content": json.dumps(payload, ensure_ascii=False)})

            errored = False
            for r, t, e in zip(payload["results"], tops, exps):
                top_s = f"{t:.3f}" if t is not None else "—"
                log(f"[tools] {r['query']!r} | {r['status']} top={top_s}"
                    f"{' widened' if r.get('widened') else ''}")
                if e:
                    log(f"        расширение ({e[0]}): {e[1]}")
                if r["status"] == "ok":
                    for a in r["articles"]:
                        if [a["law"], a["article"]] not in found:
                            found.append([a["law"], a["article"]])
                elif r["status"] == "weak":
                    weak_qs.append(r["query"])
                elif r["status"] == "error":
                    errored = True
                    log(f"        ✗ {r['message']}")

            top = max((t for t in tops if t is not None), default=None)
            if errored:
                stop = "tool_error"
            elif p.update(keys, top) >= STALL_MAX:
                stop = "loop"
                log(f"        ⟲ {STALL_MAX} вызова без прогресса")
            if stop:
                break

        return {"messages": msgs, "found": found, "called": called,
                "weak_qs": weak_qs, "stop": stop,
                "budget": dump_budget(b), "progress": dump_progress(p)}

    def nudge(s):
        log("[nudge] ответ без поиска")
        return {"messages": s["messages"] + [{"role": "user", "content": NUDGE}],
                "nudged": True}

    def reframe(s):
        log("[reframe] weak — модель переформулирует сама")
        text = REFRAME.format(qs="、".join(s["weak_qs"]))
        return {"messages": s["messages"] + [{"role": "user", "content": text}],
                "reframed": True}

    def ask_human(s):
        # Nothing above interrupt() may have side effects — see point 3.
        ans = interrupt({"kind": "ask", "weak": s["weak_qs"],
                         "prompt": "Эти подвопросы не нашлись. Уточни другими "
                                   "словами или что за ситуация (пусто — "
                                   "прекратить поиск):"})
        log(f"[ask_human] ответ: {ans!r}")
        if not str(ans).strip():
            return {"reframed": True, "messages": s["messages"] + [{
                "role": "user", "content": "追加情報はありません。見つからなかった"
                                           "論点は、見つからなかったと答えてください。"}]}
        return {"reframed": True,
                "human_notes": s.get("human_notes", []) + [str(ans)],
                "messages": s["messages"] + [{
                    "role": "user", "content": f"補足: {ans}\nこの補足を踏まえて、"
                    "見つからなかった論点を一度だけ再検索してください。"}]}

    def final(s):
        stop, found = s["stop"], s.get("found", [])
        log(f"[final] стоп: {stop}")
        if stop == "llm_error":
            text = fallback(stop, found)
            return {"answer": text, "messages": s["messages"] + [
                {"role": "assistant", "content": text}]}
        b = load_budget(s["budget"])
        head = FINAL_HEAD.get(stop, FINAL_DEFAULT)
        try:
            resp = chat(http, s["messages"] + [
                {"role": "user", "content": f"{head}\n{FINAL_BASE}"}], tools=False)
            b.charge(resp, final=True)
            text = resp["message"].get("content", "").strip()
        except Exception as e:                      # noqa: BLE001
            log(f"✗ финальный ход: {type(e).__name__}: {e}")
            text = fallback(stop, found)
        # The answer goes into the history (the FINAL instruction does not):
        # the next question in this thread must see what was said.
        return {"answer": text, "budget": dump_budget(b),
                "messages": s["messages"] + [{"role": "assistant", "content": text}]}

    def answer(s):
        return {"answer": s["messages"][-1].get("content", "").strip(),
                "stop": "answered", "ungrounded": not s.get("called")}

    # ── nodes: memory (session 4) ───────────────────────────────────────────

    def compact(s):
        """Keep conclusions, drop raw material.

        Tool messages keep the query, the status and the (law, article)
        registry; the article texts go. If a later question needs an article
        again, one search brings it back — cheaper than carrying it forever.
        """
        msgs, before, after = [], 0, 0
        for m in s["messages"]:
            if m.get("role") != "tool":
                msgs.append(m)
                continue
            before += len(m["content"])
            try:
                p = json.loads(m["content"])
            except json.JSONDecodeError:
                msgs.append(m)
                after += len(m["content"])
                continue
            slim = {"results": [{
                "query": r.get("query"), "status": r.get("status"),
                "articles": [{"law": a["law"], "article": a["article"]}
                             for a in r.get("articles", [])],
            } for r in p.get("results", [])],
                "note": "本文は省略済み。必要なら同じ条文を再検索すること。"}
            content = json.dumps(slim, ensure_ascii=False)
            after += len(content)
            msgs.append({**m, "content": content})
        log(f"[compact] тулза в истории: {before} → {after} символов")
        return {"messages": msgs}

    def propose(s):
        # The LLM call lives HERE, not in confirm_memory: the node with
        # interrupt() runs twice, and so would the call.
        said = "\n".join([s.get("question", "")] + s.get("human_notes", []))
        facts = verify_stated(extract_facts(http, said), said)
        current = profile(store)
        new = {k: v for k, v in facts.items()
               if current.get(k, {}).get("value") != v}
        log(f"[propose] извлечено и проверено: {facts} → новое: {new}")
        return {"proposal": new or None}

    def confirm_memory(s):
        ans = interrupt({"kind": "memory", "proposal": s["proposal"]})
        # After interrupt(): runs once, on resume. Safe for the side effect.
        if str(ans).strip().lower() in ("y", "yes", "д", "да", "はい"):
            today = date.today().isoformat()
            for k, v in s["proposal"].items():
                store.put(NS, k, {"value": v, "stated_at": today})
            save_store(store)
            log(f"[confirm_memory] сохранено: {s['proposal']}")
        else:
            log("[confirm_memory] отказ — ничего не сохранено")
        return {"proposal": None}

    # ── edges ────────────────────────────────────────────────────────────────

    def after_model(s):
        if s.get("stop"):
            return "final"
        if s["messages"][-1].get("tool_calls"):
            return "tools"
        if not s.get("called") and not s.get("nudged"):
            return "nudge"
        return "answer"

    def after_tools(s):
        if s.get("stop"):
            return "final"
        if s.get("weak_qs") and not s.get("reframed"):
            return "ask_human" if hitl else "reframe"
        return "model"

    def after_propose(s):
        return "confirm_memory" if s.get("proposal") else END

    g = StateGraph(State)
    for name, fn in [("model", model), ("tools", tools), ("nudge", nudge),
                     ("reframe", reframe), ("ask_human", ask_human),
                     ("final", final), ("answer", answer),
                     ("compact", compact), ("propose", propose),
                     ("confirm_memory", confirm_memory)]:
        g.add_node(name, fn)

    g.add_edge(START, "model")
    g.add_conditional_edges("model", after_model,
                            ["tools", "nudge", "final", "answer"])
    g.add_conditional_edges("tools", after_tools,
                            ["model", "reframe", "ask_human", "final"])
    for n in ("nudge", "reframe", "ask_human"):
        g.add_edge(n, "model")
    g.add_edge("final", "compact")
    g.add_edge("answer", "compact")
    g.add_edge("compact", "propose")
    g.add_conditional_edges("propose", after_propose, ["confirm_memory", END])
    g.add_edge("confirm_memory", END)

    # The checkpointer holds short-term memory (per thread); the store holds
    # long-term memory (across threads). InMemorySaver dies with the process —
    # a conversation does not survive a restart, the profile file does.
    return g.compile(checkpointer=InMemorySaver())


# ───────────────────────────────── cli ───────────────────────────────────────

def show(values):
    b = values.get("budget") or {}
    print("\n— ответ —")
    print(values.get("answer", ""))
    print(f"\n[стоп: {values.get('stop')}  итераций: {b.get('iters')}  "
          f"токенов: {b.get('tokens')}  время: {b.get('spent', 0):.1f}s]")
    if values.get("ungrounded"):
        print("⚠ ответ без единого поиска")


def run_turn(graph, thread, q, store):
    config = {"configurable": {"thread_id": thread}}
    prev = graph.get_state(config).values.get("messages") or []
    # The system prompt is replaced every turn: it carries the CURRENT
    # long-term facts, including ones confirmed a minute ago.
    msgs = [{"role": "system", "content": system_prompt(store)}] + prev[1:] + [
        {"role": "user", "content": q}]
    out = graph.invoke({"messages": msgs, "question": q, "human_notes": [],
                        "found": [], "stop": None, "answer": "", "weak_qs": [],
                        "called": False, "nudged": False, "reframed": False,
                        "ungrounded": False, "budget": None, "progress": None,
                        "proposal": None}, config)
    shown = False
    while (st := graph.get_state(config)).interrupts:
        p = st.interrupts[0].value
        if p["kind"] == "memory":
            if not shown:
                show(st.values)       # the answer first, the memory question after
                shown = True
            print("\n💾 Запомнить на будущее (только то, что ты сам сказал):")
            for k, v in p["proposal"].items():
                print(f"   · {LABELS[k]}: {v}")
            out = graph.invoke(Command(resume=input("  сохранить? (y/n) > ")), config)
        else:
            print(f"\n⏸ {p['prompt']}")
            for w in p["weak"]:
                print(f"   · {w}")
            out = graph.invoke(Command(resume=input("> ")), config)
    if not shown:
        show(out)


def main():
    args = sys.argv[1:]
    if "--mermaid" in args:
        # Drawing needs no models: nodes are never executed.
        print(build(None, None, None).get_graph().draw_mermaid())
        return

    trace, hitl = "--trace" in args, "--hitl" in args
    first = " ".join(a for a in args if not a.startswith("--"))

    print(f"порог (5-й перцентиль, score_gate.json): {THRESHOLD:.3f}")
    tool = Tool()
    store = load_store()
    prof = profile(store)
    print(f"долгая память: {json.dumps(prof, ensure_ascii=False) if prof else 'пусто'}")
    print("команды: /new — новый разговор · /mem — что помню · "
          "/forget — забыть всё · /quit")

    thread = str(uuid.uuid4())
    with httpx.Client() as http:
        graph = build(tool, http, store, hitl=hitl, trace=trace)
        pending = first
        while True:
            q = pending or input("\n? ").strip()
            pending = None
            if not q:
                continue
            if q == "/quit":
                break
            if q == "/new":
                thread = str(uuid.uuid4())
                print("— новый разговор: короткая память пуста, долгая осталась")
                continue
            if q == "/mem":
                prof = profile(store)
                print(json.dumps(prof, ensure_ascii=False, indent=2) if prof else "пусто")
                continue
            if q == "/forget":
                for k in list(profile(store)):
                    store.delete(NS, k)
                save_store(store)
                print("— долгая память удалена (строки стёрты из файла)")
                continue
            run_turn(graph, thread, q, store)


if __name__ == "__main__":
    main()
