"""Sentence-transformers encoders shared by the embedder comparison and the index.

The model table, the cache layout in data/emb/ and the resumable encoder live
here so that embed_ruri (index) and experiments/embed_compare read one copy.
"""

import json
import pathlib
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
EMB_DIR = ROOT / "data" / "emb"
SLICE = 128          # chunks per encode call between checkpoints

# key -> (HF id, query prefix, document prefix, revision). Prefixes from the
# model cards; revisions pinned 01.10: the commits the
# vectors in data/emb/ were computed with.
MODELS = {
    "multilingual-e5-large": ("intfloat/multilingual-e5-large",
                              "query: ", "passage: ",
                              "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"),
    "ruri-v3-310m": ("cl-nagoya/ruri-v3-310m",
                     "検索クエリ: ", "検索文書: ",
                     "18b60fb8c2b9df296fb4212bb7d23ef94e579cd3"),
}


def embedding_dim(model):
    """Embedding size; sentence-transformers renamed the method (get_sentence_embedding_dimension is deprecated)."""
    get = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return get()


def encode_docs(model, key, texts, prefix):
    """Core chunks -> normalised vectors, cached and resumable."""
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    path, state = EMB_DIR / f"{key}.npy", EMB_DIR / f"{key}.state.json"
    n, dim = len(texts), embedding_dim(model)
    done = 0
    if path.exists() and state.exists():
        st = json.loads(state.read_text(encoding="utf-8"))
        if st.get("n") == n and st.get("prefix") == prefix:
            done = st["done"]
    vecs = np.lib.format.open_memmap(path, mode="r+" if done else "w+",
                                     dtype=np.float32, shape=(n, dim))
    if done < n:
        print(f"  encoding documents: {done}/{n}")
    t0, start_done = time.time(), done
    while done < n:
        batch = [prefix + t for t in texts[done:done + SLICE]]
        vecs[done:done + len(batch)] = model.encode(
            batch, batch_size=16, normalize_embeddings=True,
            show_progress_bar=False)
        done += len(batch)
        vecs.flush()
        state.write_text(json.dumps({"n": n, "done": done, "prefix": prefix}),
                         encoding="utf-8")
        rate = (done - start_done) / max(time.time() - t0, 1e-6)
        print(f"  {done}/{n}  {rate:.1f} chunks/s  ETA {(n - done) / max(rate, 1e-6) / 60:.1f} min")
    return np.asarray(vecs)
