"""Embedding the chunks through Ollama + BGE-M3.

The model returns normalised vectors (|v| = 1), so dot and cosine give the
same ordering. Resumable: after an interruption a restart continues from
where it stopped.
"""

import json
import pathlib
import random
import time

import httpx
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "chunks.jsonl"
VECTORS = ROOT / "data" / "vectors.npy"
STATE = ROOT / "data" / "embed_state.json"

MODEL = "bge-m3"
URL = "http://localhost:11434/api/embed"
DIM = 1024
BATCH = 32
CHECKPOINT = 20          # batches between state saves
RETRIES = 4              # attempts per batch


def embed_batch(client, texts):
    """Ollama errors here are transient (model swap, timeout) — retry."""
    last = None
    for attempt in range(RETRIES):
        try:
            r = client.post(URL, json={"model": MODEL, "input": texts},
                            timeout=300)
            if r.status_code == 200:
                return r.json()["embeddings"]
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except (httpx.TransportError, httpx.TimeoutException) as e:
            last = f"{type(e).__name__}: {e}"

        if attempt < RETRIES - 1:
            delay = 2 ** attempt + random.random()
            print(f"    повтор {attempt + 1}/{RETRIES - 1} через {delay:.1f}с — {last}")
            time.sleep(delay)

    raise RuntimeError(f"батч не взялся за {RETRIES} попыток: {last}")


def main():
    texts = []
    ids = []
    with CHUNKS.open(encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            texts.append(c["text"])
            ids.append(c["chunk_id"])
    n = len(texts)
    print(f"чанков: {n}")

    # memmap to disk — 20162 x 1024 x 4 bytes ~ 80 MB
    mode = "r+" if VECTORS.exists() else "w+"
    vecs = np.lib.format.open_memmap(VECTORS, mode=mode, dtype=np.float32,
                                     shape=(n, DIM))

    done = 0
    if STATE.exists():
        st = json.loads(STATE.read_text(encoding="utf-8"))
        if st.get("n") == n:
            done = st.get("done", 0)
            print(f"продолжаем с {done}")
        else:
            print("корпус изменился — считаем заново")

    t0 = time.time()
    with httpx.Client() as client:
        for bi, start in enumerate(range(done, n, BATCH)):
            batch = texts[start:start + BATCH]
            vecs[start:start + len(batch)] = np.array(embed_batch(client, batch),
                                                      dtype=np.float32)
            done = start + len(batch)

            if bi % CHECKPOINT == 0 or done >= n:
                vecs.flush()
                STATE.write_text(json.dumps({"n": n, "done": done}),
                                 encoding="utf-8")
                el = time.time() - t0
                rate = (done - (n - (n - done)) if False else done) / max(el, 1e-6)
                eta = (n - done) / max(rate, 1e-6)
                print(f"{done}/{n}  {rate:.1f} чанк/с  ETA {eta/60:.1f} мин")

    vecs.flush()
    STATE.write_text(json.dumps({"n": n, "done": n}), encoding="utf-8")

    (ROOT / "data" / "chunk_ids.json").write_text(
        json.dumps(ids, ensure_ascii=False), encoding="utf-8")

    norms = np.linalg.norm(np.asarray(vecs[:100]), axis=1)
    print(f"\nготово. форма {vecs.shape}")
    print(f"нормы первых 100: min {norms.min():.4f}  max {norms.max():.4f}")
    print(f"время: {(time.time() - t0)/60:.1f} мин")


if __name__ == "__main__":
    main()