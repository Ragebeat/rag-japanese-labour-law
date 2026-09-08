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

  uv run uvicorn src.api:app --reload --port 8000
"""

import json
import pathlib
import sys

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rerank  # noqa: E402
from generate import (DEFAULT_MODE, MODEL, SYSTEM, TOP_K,  # noqa: E402
                      Reranked, build_context, inspect, retrieve)

OLLAMA = "http://localhost:11434/api/chat"

state = {}


async def lifespan(app):
    # DB connection and cross-encoder: once per process.
    state["rr"] = Reranked()
    rerank._get()          # warm-up: otherwise the first request pays 15 seconds
    state["http"] = httpx.AsyncClient(timeout=300)
    yield
    await state["http"].aclose()


app = FastAPI(title="Japanese Labour Law RAG", lifespan=lifespan)

# In development the front end runs on a different port.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


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


def _payload(question, chunks, answer):
    info = inspect(answer, chunks)
    used = set(info["marks"])
    return {
        "question": question,
        "answer": answer,
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
    """The whole answer at once. For scripts and debugging."""
    chunks = retrieve(state["rr"], req.q, req.mode, k=req.k)
    if not chunks:
        raise HTTPException(404, "ничего не найдено")

    r = await state["http"].post(OLLAMA, json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"【条文】\n{build_context(chunks)}\n\n【質問】\n{req.q}"},
        ],
        "stream": False, "think": False,
        "options": {"temperature": 0},
    })
    r.raise_for_status()
    return _payload(req.q, chunks, r.json()["message"]["content"].strip())


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/ask/stream")
async def ask_stream(req: Ask):
    """SSE: sources first, then tokens, then the analysis of the answer."""

    async def gen():
        try:
            chunks = retrieve(state["rr"], req.q, req.mode, k=req.k)
        except Exception as e:                      # noqa: BLE001
            yield _sse("error", {"message": str(e)})
            return
        if not chunks:
            yield _sse("error", {"message": "ничего не найдено"})
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
            "options": {"temperature": 0},
        }
        parts = []
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
                        yield _sse("delta", {"text": delta})
                    if piece.get("done"):
                        break
        except Exception as e:                      # noqa: BLE001
            yield _sse("error", {"message": str(e)})
            return

        # The analysis runs on the complete text: markers and the verdict are
        # only fully visible at the end.
        yield _sse("done", _payload(req.q, chunks, "".join(parts).strip()))

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",     # without this nginx buffers the stream
    })


# The built front end, if present. Mounted last so it does not intercept
# /ask and /health. In development the UI runs on Vite (proxying to :8000),
# for a demo — npm run build and everything serves from one port.
_dist = pathlib.Path(__file__).resolve().parent.parent / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")
