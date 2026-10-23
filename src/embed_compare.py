"""Comparing embedding models on Japanese labour law.

Two levels, measured separately:

1. BARE VECTOR SEARCH (this script). Pure model comparison: no expansion,
   no reranker. recall@5 is what goes into the context without a reranker;
   recall@20 is the ceiling the reranker can pull from (pool = 20). A model
   that loses at @5 but wins at @20 may win after reranking.
2. FULL PIPELINE — only for the leader of level 1, against BGE-M3. Not here:
   the expansion dictionary and the gate were tuned on BGE-M3's misses, so
   the full pipeline is home ground for the incumbent.

Fairness rules:
- the SAME text for every model: c["text"], header line included — exactly
  what BGE-M3 embedded in embed.py
- core only (7 452 chunks): golden is evaluated on core
- each model gets ITS OWN prefixes. Forgetting them drops recall silently,
  and the model "loses" to its wiring, not to itself
- inputs longer than the model's limit are counted and reported: e5 stops
  at 512 tokens, the others at 8 192
- the baseline is the existing vectors.npy + Ollama bge-m3 for queries; it
  must reproduce eval.py's core recall@5 = 0.68, or this script counts
  differently from before and nothing below it is comparable

Torch here is CPU-only (a Pascal GPU is not supported by current PyTorch builds), so
encoding is slow. Vectors are cached per model in data/emb/ and the
encoding resumes after an interruption.

  uv run python src/embed_compare.py                  # all models
  uv run python src/embed_compare.py --only ruri      # one (substring of the key)
"""

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval import first_hit  # noqa: E402
from search import Index  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden.jsonl"
EMB_DIR = ROOT / "data" / "emb"
OUT = ROOT / "data" / "embed_compare.json"

KS = (1, 5, 10, 20)
MAXK = max(KS)
SLICE = 128          # chunks per encode call between checkpoints

# key -> (HF id, query prefix, document prefix, revision). Prefixes from the
# model cards; revisions pinned 01.10 — the commits the
# vectors in data/emb/ were computed with.
MODELS = {
    "multilingual-e5-large": ("intfloat/multilingual-e5-large",
                              "query: ", "passage: ",
                              "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"),
    "ruri-v3-310m": ("cl-nagoya/ruri-v3-310m",
                     "検索クエリ: ", "検索文書: ",
                     "18b60fb8c2b9df296fb4212bb7d23ef94e579cd3"),
}
BASELINE = "bge-m3 (ollama)"



def _dim(model):
    """Embedding size; sentence-transformers renamed the method (get_sentence_embedding_dimension is deprecated)."""
    get = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return get()

def load_golden():
    return [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]


def metrics(positions):
    n = len(positions)
    m = {f"recall@{k}": sum(1 for p in positions if p and p <= k) / n for k in KS}
    m["MRR@20"] = sum(1 / p for p in positions if p) / n
    return m


def ranked(vecs, q, meta, k=MAXK):
    """Exact top-k by dot product (all vectors normalised) -> eval.py's shape."""
    scores = vecs @ q
    top = np.argpartition(-scores, k)[:k]
    top = top[np.argsort(-scores[top])]
    return [(float(scores[i]), meta[i]) for i in top]


def encode_docs(model, key, texts, prefix):
    """Core chunks -> normalised vectors, cached and resumable."""
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    path, state = EMB_DIR / f"{key}.npy", EMB_DIR / f"{key}.state.json"
    n, dim = len(texts), _dim(model)
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


def truncated(model, texts, prefix):
    """How many inputs exceed the model's window — they are cut silently."""
    limit = model.max_seq_length
    tok = model.tokenizer
    lens = [len(tok(prefix + t, add_special_tokens=True)["input_ids"]) for t in texts]
    return limit, sum(1 for x in lens if x > limit)


def run_baseline(golden, core_idx, meta):
    """The existing BGE-M3 vectors and Ollama for queries — the configuration
    every BGE-M3 number was measured on."""
    idx = Index()
    vecs = idx.vecs[core_idx]
    rows = []
    for g in golden:
        res = ranked(vecs, idx.embed(g["q"]), meta)
        rows.append({"id": g["id"], "pos": first_hit(res, g["relevant"])})
    return rows


def run_model(key, golden, texts, meta):
    from sentence_transformers import SentenceTransformer
    hf_id, qp, dp, rev = MODELS[key]
    print(f"\n=== {key} ({hf_id}) ===")
    model = SentenceTransformer(hf_id, device="cpu", revision=rev)
    limit, cut = truncated(model, texts, dp)
    print(f"  window {limit} tokens, longer than the window: {cut} of {len(texts)} chunks")
    vecs = encode_docs(model, key, texts, dp)
    qv = model.encode([qp + g["q"] for g in golden], batch_size=16,
                      normalize_embeddings=True, show_progress_bar=False)
    rows = [{"id": g["id"], "pos": first_hit(ranked(vecs, q, meta), g["relevant"])}
            for g, q in zip(golden, qv)]
    return rows, {"hf_id": hf_id, "query_prefix": qp, "doc_prefix": dp,
                  "revision": rev, "window": limit, "truncated": cut}


def diff(base, other, k, qs):
    b = {r["id"]: r["pos"] for r in base}
    o = {r["id"]: r["pos"] for r in other}

    def ok(p):
        return p is not None and p <= k

    fixed = [i for i in b if not ok(b[i]) and ok(o[i])]
    broke = [i for i in b if ok(b[i]) and not ok(o[i])]
    print(f"  @{k}: fixed {len(fixed)}, broke {len(broke)}")
    for i in fixed:
        print(f"     + #{i:>2} {qs[i][:30]:<32} {b[i]} → {o[i]}")
    for i in broke:
        print(f"     − #{i:>2} {qs[i][:30]:<32} {b[i]} → {o[i]}")


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    golden = load_golden()
    qs = {g["id"]: g["q"] for g in golden}

    all_meta = [json.loads(l) for l in (ROOT / "data" / "chunks.jsonl").open(encoding="utf-8")]
    core_idx = np.array([i for i, c in enumerate(all_meta) if c["core"]])
    meta = [all_meta[i] for i in core_idx]
    texts = [c["text"] for c in meta]
    print(f"questions: {len(golden)}, core chunks: {len(meta)}")

    report = {}
    base = run_baseline(golden, core_idx, meta)
    report[BASELINE] = {"metrics": metrics([r["pos"] for r in base]), "rows": base}
    r5 = report[BASELINE]["metrics"]["recall@5"]
    print(f"\nbaseline {BASELINE}: recall@5 = {r5:.3f}  (eval.py core: 0.68)")
    if abs(r5 - 0.68) > 1e-9:
        print("⚠ the baseline does not reproduce eval.py — the comparison below is invalid, investigate first")

    for key in MODELS:
        if only and only not in key:
            continue
        rows, info = run_model(key, golden, texts, meta)
        report[key] = {"metrics": metrics([r["pos"] for r in rows]), "rows": rows, **info}

    print(f"\n{'model':<24}" + "".join(f"{f'r@{k}':>8}" for k in KS) + f"{'MRR':>8}")
    print("-" * (24 + 8 * (len(KS) + 1)))
    for key, rep in report.items():
        m = rep["metrics"]
        print(f"{key:<24}" + "".join(f"{m[f'recall@{k}']:>8.3f}" for k in KS)
              + f"{m['MRR@20']:>8.3f}")
    print("⚠ 50 questions: a difference below 0.02 (one question) is noise")

    for key, rep in report.items():
        if key == BASELINE:
            continue
        print(f"\n{BASELINE} → {key}")
        for k in (5, 20):
            diff(report[BASELINE]["rows"], rep["rows"], k, qs)

    # --only measures one model: keep the other models' results already in the file
    # (overwriting dropped them, and the README table needs all three)
    if only and OUT.exists():
        report = {**json.loads(OUT.read_text(encoding="utf-8")), **report}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
