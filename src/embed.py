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
            print(f"    retry {attempt + 1}/{RETRIES - 1} in {delay:.1f}s — {last}")
            time.sleep(delay)

    raise RuntimeError(f"batch failed after {RETRIES} attempts: {last}")


def main():
    texts = []
    ids = []
    with CHUNKS.open(encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            texts.append(c["text"])
            ids.append(c["chunk_id"])
    n = len(texts)
    print(f"chunks: {n}")

    done = 0
    if STATE.exists():
        st = json.loads(STATE.read_text(encoding="utf-8"))
        if st.get("n") == n:
            done = st.get("done", 0)
            print(f"resuming from {done}")
        else:
            print("the corpus changed — starting over")
    # Resume only into a file of the right shape. "r+" ignores the shape
    # argument, so a file left by a different corpus kept its old row count:
    # the vectors were recomputed, the stale tail rows were not dropped.
    if done and VECTORS.exists() and np.load(VECTORS, mmap_mode="r").shape != (n, DIM):
        print("vectors.npy has a different shape — starting over")
        done = 0

    # memmap to disk — 20162 x 1024 x 4 bytes ~ 80 MB
    vecs = np.lib.format.open_memmap(VECTORS, mode="r+" if done else "w+",
                                     dtype=np.float32, shape=(n, DIM))

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
                print(f"{done}/{n}  {rate:.1f} chunks/s  ETA {eta/60:.1f} min")

    vecs.flush()
    STATE.write_text(json.dumps({"n": n, "done": n}), encoding="utf-8")

    (ROOT / "data" / "chunk_ids.json").write_text(
        json.dumps(ids, ensure_ascii=False), encoding="utf-8")

    norms = np.linalg.norm(np.asarray(vecs[:100]), axis=1)
    print(f"\ndone. shape {vecs.shape}")
    print(f"norms of the first 100: min {norms.min():.4f}  max {norms.max():.4f}")
    print(f"time: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()