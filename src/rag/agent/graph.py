"""The same agent as agent/loop.py, as a LangGraph state graph.

LangGraph describes an agent as a graph: nodes do the work, edges decide what
runs next, and one state object travels through. agent/loop.py already WAS such a
graph: its transitions were `if`s inside a `while`. Here they have names.

What the move to a graph forced:

1. STATE IS DATA. A checkpointer saves the state after every node, so the
   budget and the loop detector live in plain dicts, loaded into the same
   tested classes (Budget, Progress) inside a node and written back.
2. HUMAN WAIT IS NOT BUDGET. The time budget stores seconds SPENT, not a start
   time: a graph paused for an hour must not wake up out of time.
3. A NODE WITH interrupt() RUNS TWICE. On resume the node re-executes from the
   top. Nothing before interrupt() may have side effects, or cost much.

Memory, three kinds:

SHORT-TERM = the thread. Several questions in one conversation share the
message history (thread_id). The system prompt is rebuilt every turn, so it
always carries the current long-term facts.

COMPACTION = keep conclusions, drop raw material (the same rule as agent/loop.py). After
every answer the full article texts are stripped from old tool messages; the
(law, article) registry and the answers stay. One round of two sub-questions
was +9 000 tokens: without this, the second question would not fit.

LONG-TERM = a store across threads, and here the design decisions matter more
than the code:
  - ALLOW-LIST, not deny-list. Only employment type, company size and
    prefecture can ever be stored. Pregnancy, health, disputes cannot be
    proposed at all: there is no field for them. A deny-list of "sensitive
    words" would leak the first phrasing nobody anticipated.
  - THE MODEL PROPOSES, CODE VERIFIES, THE HUMAN DECIDES. gemma extracts
    candidates; code keeps only values literally present in what the user
    said (no inference: "asks about 育休, so probably..." is rejected);
    the user confirms through interrupt().
  - EVERY FACT CARRIES A DATE. Facts go stale; after 90 days code marks them
    for re-confirmation instead of trusting them.
  - THE EXPENSIVE CALL SITS OUTSIDE THE interrupt() NODE. Extraction is in
    `propose`, the pause is in `confirm_memory`: otherwise the LLM call would
    run twice (point 3 above).
  - /forget DELETES. Not a flag, not "used to be": the file loses the lines.

  uv run python -m rag.agent.graph --trace
  uv run python -m rag.agent.graph --trace --hitl "会社が住民税を給料から天引きするのは合法？"
  uv run python -m rag.agent.graph --mermaid
"""

import json
import sys
import time
import uuid
from datetime import date
from typing import TypedDict

import httpx
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from rag import tracing
from rag.agent.loop import (
    FINAL_BASE,
    FINAL_DEFAULT,
    FINAL_HEAD,
    NUDGE,
    REFRAME,
    STALL_MAX,
    THRESHOLD,
    TOOL_NAME,
    Budget,
    Progress,
    Tool,
    _error,
    chat,
    fallback,
)
from rag.agent.memory import (
    LABELS,
    check_owner,
    extract_facts,
    load_store,
    namespace,
    profile,
    save_store,
    system_prompt,
    verify_stated,
)
from rag.generation.guard import guard_links


class State(TypedDict, total=False):
    messages: list
    question: str
    human_notes: list    # what the human typed at ask_human: also user speech
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
    owner: str           # user_id that started the thread: see check_owner


# --- state <-> the tested objects ---

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


# --- the graph ---

def build(tool, http, store, user_id, hitl=False, trace=False):
    t_start = time.monotonic()

    def log(s):
        if trace:
            print(f"  {time.monotonic() - t_start:6.1f}s  {s}")

    # --- nodes: the search loop ---

    def model(s):
        b = load_budget(s.get("budget"))
        stop = b.exhausted()
        if stop:
            return {"stop": stop, "budget": dump_budget(b)}
        try:
            resp = chat(http, s["messages"])
        except Exception as e:                      # noqa: BLE001
            log(f"✗ model: {type(e).__name__}: {e}")
            return {"stop": "llm_error", "budget": dump_budget(b)}
        b.charge(resp)
        msg = resp["message"]
        log(f"[model] #{b.iters} prompt={b.last_prompt} total={b.tokens} "
            f"calls={len(msg.get('tool_calls') or [])}")
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
            for r, t, e in zip(payload["results"], tops, exps, strict=False):
                top_s = f"{t:.3f}" if t is not None else "—"
                log(f"[tools] {r['query']!r} | {r['status']} top={top_s}"
                    f"{' widened' if r.get('widened') else ''}")
                if e:
                    log(f"        expansion ({e[0]}): {e[1]}")
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
                log(f"        ⟲ {STALL_MAX} calls without progress")
            if stop:
                break

        return {"messages": msgs, "found": found, "called": called,
                "weak_qs": weak_qs, "stop": stop,
                "budget": dump_budget(b), "progress": dump_progress(p)}

    def nudge(s):
        log("[nudge] answer without a search")
        return {"messages": s["messages"] + [{"role": "user", "content": NUDGE}],
                "nudged": True}

    def reframe(s):
        log("[reframe] weak — the model rephrases by itself")
        text = REFRAME.format(qs="、".join(s["weak_qs"]))
        return {"messages": s["messages"] + [{"role": "user", "content": text}],
                "reframed": True}

    def ask_human(s):
        # Nothing above interrupt() may have side effects: see point 3.
        ans = interrupt({"kind": "ask", "weak": s["weak_qs"],
                         "prompt": "Nothing found for these sub-questions. Rephrase, "
                                   "or describe the situation (empty — "
                                   "stop searching):"})
        log(f"[ask_human] answer: {ans!r}")
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
        log(f"[final] stop: {stop}")
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
            log(f"✗ final turn: {type(e).__name__}: {e}")
            text = fallback(stop, found)
        # The answer goes into the history (the FINAL instruction does not):
        # the next question in this thread must see what was said.
        return {"answer": text, "budget": dump_budget(b),
                "messages": s["messages"] + [{"role": "assistant", "content": text}]}

    def answer(s):
        return {"answer": s["messages"][-1].get("content", "").strip(),
                "stop": "answered", "ungrounded": not s.get("called")}

    # --- nodes: memory ---

    def compact(s):
        """Keep conclusions, drop raw material.

        Tool messages keep the query, the status and the (law, article)
        registry; the article texts go. If a later question needs an article
        again, one search brings it back: cheaper than carrying it forever.
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
        log(f"[compact] tool output in history: {before} → {after} chars")
        return {"messages": msgs}

    def propose(s):
        # The LLM call lives HERE, not in confirm_memory: the node with
        # interrupt() runs twice, and so would the call.
        said = "\n".join([s.get("question", "")] + s.get("human_notes", []))
        facts = verify_stated(extract_facts(http, said), said)
        current = profile(store, user_id)
        new = {k: v for k, v in facts.items()
               if current.get(k, {}).get("value") != v}
        log(f"[propose] extracted and checked: {facts} → new: {new}")
        return {"proposal": new or None}

    def confirm_memory(s):
        ans = interrupt({"kind": "memory", "proposal": s["proposal"]})
        # After interrupt(): runs once, on resume. Safe for the side effect.
        if str(ans).strip().lower() in ("y", "yes", "д", "да", "はい"):
            today = date.today().isoformat()
            for k, v in s["proposal"].items():
                store.put(namespace(user_id), k, {"value": v, "stated_at": today})
            save_store(store, user_id)
            log(f"[confirm_memory] saved: {s['proposal']}")
        else:
            log("[confirm_memory] declined — nothing saved")
        return {"proposal": None}

    # --- edges ---

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
    # long-term memory (across threads). InMemorySaver dies with the process:
    # a conversation does not survive a restart, the profile file does.
    return g.compile(checkpointer=InMemorySaver())


# --- cli ---

def show(values):
    b = values.get("budget") or {}
    # The output guard at the boundary where the answer reaches a person.
    answer, blocked = guard_links(values.get("answer", ""))
    print("\n— answer —")
    print(answer)
    if blocked:
        print(f"WARNING: links outside the allow-list removed: {blocked}")
    print(f"\n[stop: {values.get('stop')}  iterations: {b.get('iters')}  "
          f"tokens: {b.get('tokens')}  time: {b.get('spent', 0):.1f}s]")
    if values.get("ungrounded"):
        print("WARNING: answered without a single search")


def run_turn(graph, thread, q, store, user_id):
    """One turn = one trace. The span also covers a human-in-the-loop pause:
    the trace's wall time then includes waiting for a person, while the
    model and tool spans inside show only their own time."""
    with tracing.span("agent_graph.turn", "AGENT", input=q,
                      attrs={"session.id": thread}):
        _run_turn(graph, thread, q, store, user_id)


def _run_turn(graph, thread, q, store, user_id):
    config = {"configurable": {"thread_id": thread}}
    st = graph.get_state(config).values
    check_owner(st, user_id)               # before reading a single message
    prev = st.get("messages") or []
    # The system prompt is replaced every turn: it carries the CURRENT
    # long-term facts, including ones confirmed a minute ago.
    msgs = [{"role": "system", "content": system_prompt(store, user_id)}] + prev[1:] + [
        {"role": "user", "content": q}]
    out = graph.invoke({"messages": msgs, "question": q, "human_notes": [],
                        "owner": user_id,
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
            print("\nRemember for next time (only what you said yourself):")
            for k, v in p["proposal"].items():
                print(f"   · {LABELS[k]}: {v}")
            out = graph.invoke(Command(resume=input("  save? (y/n) > ")), config)
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
        print(build(None, None, None, None).get_graph().draw_mermaid())
        return

    trace, hitl = "--trace" in args, "--hitl" in args
    first = " ".join(a for a in args if not a.startswith("--"))

    print(f"threshold (5th percentile, score_gate.json): {THRESHOLD:.3f}")
    tracing.setup()
    tool = Tool()
    # The CLI's authentication is the OS session: whoever runs it is the
    # one user. An API would take the id from its auth layer, right here.
    user_id = "local"
    store = load_store(user_id)
    prof = profile(store, user_id)
    print(f"long-term memory: {json.dumps(prof, ensure_ascii=False) if prof else 'empty'}")
    print("commands: /new — new conversation · /mem — what I remember · "
          "/forget — forget everything · /quit")

    thread = str(uuid.uuid4())
    with httpx.Client() as http:
        graph = build(tool, http, store, user_id, hitl=hitl, trace=trace)
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
                print("— new conversation: short-term memory cleared, long-term kept")
                continue
            if q == "/mem":
                prof = profile(store, user_id)
                print(json.dumps(prof, ensure_ascii=False, indent=2) if prof else "empty")
                continue
            if q == "/forget":
                for k in list(profile(store, user_id)):
                    store.delete(namespace(user_id), k)
                save_store(store, user_id)
                print("— long-term memory deleted (lines removed from the file)")
                continue
            run_turn(graph, thread, q, store, user_id)


if __name__ == "__main__":
    main()
