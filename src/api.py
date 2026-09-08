"""HTTP-слой над конвейером. FastAPI + SSE.

Три решения, заложенных сразу, потому что переделывать дороже:

1. МОДЕЛИ ЖИВУТ В ПРОЦЕССЕ. Reranked и cross-encoder поднимаются в lifespan
   и переиспользуются. В скриптах реранкер грузился заново при каждом запуске
   — 15 секунд, которые в вебе были бы на каждом запросе.

2. ОТДАЁМ НЕ ТОЛЬКО ТЕКСТ. Контекст со score, разбор маркеров, вердикт и
   список НЕиспользованных чанков. Все эти данные и так есть в момент
   генерации — выбросить их значит заставить интерфейс показывать результат
   вместо работы системы.

3. СТРИМИНГ. Контекст уходит первым событием (retrieval уже завершён), потом
   токены, потом итоговый разбор. Пользователь видит источники до того, как
   模ель начала писать.

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
    # Соединение с БД и cross-encoder — один раз на процесс.
    state["rr"] = Reranked()
    rerank._get()          # прогрев: иначе первый запрос платит 15 секунд
    state["http"] = httpx.AsyncClient(timeout=300)
    yield
    await state["http"].aclose()


app = FastAPI(title="Japanese Labour Law RAG", lifespan=lifespan)

# Фронт живёт на другом порту в разработке.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class Ask(BaseModel):
    q: str = Field(min_length=1, max_length=500)
    mode: str = Field(default=DEFAULT_MODE, pattern="^(raw|dedupe|whole)$")
    k: int = Field(default=TOP_K, ge=1, le=10)


def _sources(chunks):
    """Контекст в виде, пригодном для показа: заголовок, score, полный текст."""
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
        # Чанки, которые модель не взяла. Показывать их важнее, чем кажется:
        # именно там видно, что retrieval нашёл, а генерация проигнорировала.
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
    """Синхронный ответ целиком. Для скриптов и отладки."""
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
    """SSE: сначала источники, затем токены, затем разбор ответа."""

    async def gen():
        try:
            chunks = retrieve(state["rr"], req.q, req.mode, k=req.k)
        except Exception as e:                      # noqa: BLE001
            yield _sse("error", {"message": str(e)})
            return
        if not chunks:
            yield _sse("error", {"message": "ничего не найдено"})
            return

        # Источники уходят до генерации — их уже можно показывать.
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

        # Разбор считаем по полному тексту — маркеры и вердикт целиком
        # видны только в конце.
        yield _sse("done", _payload(req.q, chunks, "".join(parts).strip()))

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",     # иначе nginx буферизует поток
    })


# Собранный фронт — если есть. Монтируется последним, чтобы не перехватывать
# /ask и /health. В разработке фронт живёт на Vite (прокси на :8000),
# для демо — npm run build и всё крутится на одном порту.
_dist = pathlib.Path(__file__).resolve().parent.parent / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")
