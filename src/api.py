"""HTTP layer over the pipeline. FastAPI + SSE.

Three decisions taken up front, because retrofitting them costs more:

1. MODELS LIVE IN THE PROCESS. Reranked and the cross-encoder are loaded in
   lifespan and reused. In the CLI scripts the reranker was loaded again on
   every run — 15 seconds that would otherwise be paid on every web request.

2. WE RETURN MORE THAN TEXT. Context with scores, parsed markers, the verdict
   and the list of UNUSED chunks. All of it exists at generation time anyway;
   dropping it would force the UI to show the result instead of the work.

3. STREAMING. Context goes out as the first event (retrieval is already
   finished), then tokens, then the final analysis. The user sees the sources
   before the model starts writing.

4. ONE REQUEST AT A TIME, THE REST IN A BOUNDED QUEUE (OWASP
   unbounded consumption). Reranking and generation saturate this machine;
   two requests in parallel are not faster, both just take twice as long.
   The queue is bounded: an unbounded one only moves the consumption into
   open connections, and the tenth in line would wait ten minutes.

  uv run uvicorn src.api:app --reload --port 8000
"""

import asyncio
import functools
import json
import logging
import pathlib
import sys

import anyio
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rerank  # noqa: E402
from generate import (DEFAULT_MODE, GEN_OPTIONS, MODEL, SYSTEM,  # noqa: E402
                      TOP_K, Reranked, StreamGuard, build_context,
                      guard_links, inspect, retrieve)

OLLAMA = "http://localhost:11434/api/chat"
MAX_WAITING = 3        # behind the one being served; ~1 min each on this machine

log = logging.getLogger("api")
state = {}


class Line:
    """A FIFO queue with positions. line[0] is being served, the rest wait.

    Every departure swaps in a fresh Event and sets the old one: all waiters
    wake, recompute their position and report it. Single-threaded asyncio,
    so reading the position and picking up the event cannot interleave.
    """

    def __init__(self, max_waiting=MAX_WAITING):
        self.line, self.max_waiting = [], max_waiting
        self._moved = asyncio.Event()

    def join(self):
        """-> a ticket, or None when the queue is full."""
        if len(self.line) > self.max_waiting:
            return None
        t = object()
        self.line.append(t)
        return t

    def ahead(self, t):
        return self.line.index(t)

    async def moved(self):
        await self._moved.wait()

    def leave(self, t):
        self.line.remove(t)
        old, self._moved = self._moved, asyncio.Event()
        old.set()


async def in_thread(fn, *a, **kw):
    """Blocking work off the event loop. retrieve() is 7-13 s of reranking on
    the CPU; called straight from `async def` it froze the whole server,
    /health included — the classic blocking call inside async, in real code.

    anyio does NOT abandon the thread on cancellation (the default): if the
    client leaves, the request waits for the thread to finish before the
    queue slot is released. A thread cannot be killed; releasing early would
    start the next request alongside it — two rerankers on one CPU, exactly
    what the queue prevents. The same zombie as Tool._busy in agent.py.
    """
    return await anyio.to_thread.run_sync(functools.partial(fn, *a, **kw))


async def lifespan(app):
    # DB connection and cross-encoder: once per process.
    state["rr"] = Reranked()
    rerank._get()          # warm-up: otherwise the first request pays 15 seconds
    import hybrid
    if hybrid.EMBEDDER != "bge-m3":   # in-process embedder: warm it up too
        hybrid._st_model(hybrid.EMBEDDER)
    state["http"] = httpx.AsyncClient(timeout=300)
    state["line"] = Line()
    yield
    await state["http"].aclose()


app = FastAPI(title="Japanese Labour Law RAG", lifespan=lifespan)

# No CORS middleware — on purpose. It was allow_origins=["*"], and nothing
# needed it: in development Vite proxies /ask to this port, in a demo the
# built front end is served from this same port. Both are same-origin. The
# only thing "*" did was let ANY site open in the browser call this API and
# read the answers (OWASP review).


class Ask(BaseModel):
    q: str = Field(min_length=1, max_length=500)
    mode: str = Field(default=DEFAULT_MODE, pattern="^(raw|dedupe|whole)$")
    k: int = Field(default=TOP_K, ge=1, le=10)


def _sources(chunks):
    """Context in a shape fit for display: heading, score, full text."""
    return [{
        "n": i,
        "law": c["law_title"],
        "article": c["article_title"],
        "caption": c["article_caption"] or "",
        "score": round(c.get("rerank_score", 0.0), 3),
        "rank_before": c.get("rank_before"),
        "text": c["text"].split("\n", 1)[-1],
    } for i, c in enumerate(chunks, 1)]


def _payload(question, chunks, answer, blocked=()):
    info = inspect(answer, chunks)
    used = set(info["marks"])
    return {
        "question": question,
        "answer": answer,
        # Links masked by the output guard before the answer left the server.
        "blocked_links": list(blocked),
        "verdict": info["verdict"],
        "sources": _sources(chunks),
        "used": sorted(used),
        # Chunks the model did not take. Showing them matters more than it looks:
        # this is where you see what retrieval found and generation ignored.
        "unused": [i for i in range(1, len(chunks) + 1) if i not in used],
        "cited_articles": info["cited"],
        "transitive": info["transitive"],
        "fabricated": info["bad_arts"],
        "hedged": info["hedged"],
    }


@app.get("/health")
async def health():
    return {"ok": True, "generator": MODEL, "mode": DEFAULT_MODE}


@app.post("/ask")
async def ask(req: Ask):
    """The whole answer at once. For scripts and debugging. Waits in the
    same queue, silently; 429 only when the queue itself is full."""
    line = state["line"]
    t = line.join()
    if t is None:
        raise HTTPException(429, "queue full", headers={"Retry-After": "60"})
    try:
        while line.ahead(t) > 0:
            await line.moved()
        chunks = await in_thread(retrieve, state["rr"], req.q, req.mode, k=req.k)
        if not chunks:
            raise HTTPException(404, "nothing found")

        r = await state["http"].post(OLLAMA, json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": f"【条文】\n{build_context(chunks)}\n\n【質問】\n{req.q}"},
            ],
            "stream": False, "think": False,
            "options": GEN_OPTIONS,
        })
        r.raise_for_status()
        answer, blocked = guard_links(r.json()["message"]["content"].strip())
        return _payload(req.q, chunks, answer, blocked)
    finally:
        line.leave(t)


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/ask/stream")
async def ask_stream(req: Ask):
    """SSE: queue position while waiting, then sources, tokens, analysis.

    The ticket is taken INSIDE the generator, not before StreamingResponse:
    if the client left before the stream started, a ticket taken outside
    would never reach a finally and would block the queue forever.
    Errors go out as codes, not str(e): exception text can carry internals
    (paths, hosts, the query itself) — details go to the server log.
    """

    async def gen():
        line = state["line"]
        t = line.join()
        if t is None:
            yield _sse("error", {"code": "queue_full"})
            return
        try:
            while (ahead := line.ahead(t)) > 0:
                yield _sse("queued", {"ahead": ahead})
                await line.moved()
            async for frame in _answer_frames(req):
                yield frame
        finally:
            line.leave(t)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",     # without this nginx buffers the stream
    })


async def _answer_frames(req):
    """One answer as SSE frames. Runs only while holding the queue's head."""
    try:
        chunks = await in_thread(retrieve, state["rr"], req.q, req.mode, k=req.k)
    except Exception:                               # noqa: BLE001
        log.exception("retrieve failed")
        yield _sse("error", {"code": "internal"})
        return
    if not chunks:
        yield _sse("error", {"code": "not_found"})
        return

    # Sources go out before generation — they can already be displayed.
    yield _sse("sources", {"sources": _sources(chunks)})

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"【条文】\n{build_context(chunks)}\n\n【質問】\n{req.q}"},
        ],
        "stream": True, "think": False,
        "options": GEN_OPTIONS,
    }
    parts = []
    # The guard sits between the model and the socket: a link is held
    # back until it is complete and checked — never half-sent.
    guard = StreamGuard()
    try:
        async with state["http"].stream("POST", OLLAMA, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                piece = json.loads(line)
                delta = piece.get("message", {}).get("content", "")
                if delta:
                    parts.append(delta)
                    safe = guard.push(delta)
                    if safe:
                        yield _sse("delta", {"text": safe})
                if piece.get("done"):
                    break
    except Exception:                               # noqa: BLE001
        log.exception("generation failed")
        yield _sse("error", {"code": "internal"})
        return
    tail = guard.flush()
    if tail:
        yield _sse("delta", {"text": tail})

    # The analysis runs on the complete text: markers and the verdict are
    # only fully visible at the end. The answer is the guarded one — the
    # same text the user just saw streamed.
    answer = guard_links("".join(parts).strip())[0]
    yield _sse("done", _payload(req.q, chunks, answer, guard.blocked))


# The built front end, if present. Mounted last so it does not intercept
# /ask and /health. In development the UI runs on Vite (proxying to :8000),
# for a demo — npm run build and everything serves from one port.
_dist = pathlib.Path(__file__).resolve().parent.parent / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")
