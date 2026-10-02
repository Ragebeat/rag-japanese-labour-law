"""Ruri v3 vectors for the whole corpus, into Postgres next to BGE-M3 (phase 7).

  uv run python src/embed_ruri.py encode   # non-core chunks (core reused from embed_compare)
  uv run python src/embed_ruri.py load     # column embedding_ruri + HNSW index
  uv run python src/embed_ruri.py check    # HNSW vs exact search, per ef_search

Why a second column and not a new table: the rows, the tsvector and every
metadata field are the same; only the vector differs. Hybrid picks the
column by its embedder (EMBEDDERS in hybrid.py), BGE-M3 stays untouched, and
switching back is one environment variable.

The core 7 452 chunks were already encoded by embed_compare.py with the same
model, revision and document prefix — they are reused, not recomputed. Only
the remaining ~12 700 are encoded here.

⚠️ db_init.py `schema` drops the table, and this column with it. After a
full rebuild run `load` here again (the vectors stay cached in data/emb/).

`check` exists because HNSW is approximate and EF_SEARCH = 100 was measured
on BGE-M3's vectors (hybrid.py). A different model has a different geometry;
the same ef can drop results silently. Exact numpy top-20 is the reference.
"""

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from embed_compare import EMB_DIR, MODELS, encode_docs  # noqa: E402
from hybrid import EMBEDDERS, Hybrid  # noqa: E402
from db_init import connect  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "chunks.jsonl"
GOLDEN = ROOT / "data" / "golden.jsonl"
KEY = "ruri-v3-310m"
FULL = EMB_DIR / f"{KEY}.full.npy"
CFG = EMBEDDERS[KEY]


def load_meta():
    return [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]


def cmd_encode():
    from sentence_transformers import SentenceTransformer
    hf_id, _qp, dp, rev = MODELS[KEY]
    assert (hf_id, rev) == (CFG["hf"], CFG["revision"]), \
        "embed_compare.MODELS and hybrid.EMBEDDERS disagree on the model"
    meta = load_meta()
    core = [i for i, c in enumerate(meta) if c["core"]]
    rest = [i for i, c in enumerate(meta) if not c["core"]]

    core_vecs = np.load(EMB_DIR / f"{KEY}.npy")
    assert len(core_vecs) == len(core), "core cache is from a different corpus"

    model = SentenceTransformer(hf_id, device="cpu", revision=rev)
    print(f"ядро из кэша: {len(core)}, кодировать: {len(rest)}")
    rest_vecs = encode_docs(model, f"{KEY}.rest", [meta[i]["text"] for i in rest], dp)

    full = np.zeros((len(meta), CFG["dim"]), dtype=np.float32)
    full[core] = core_vecs
    full[rest] = rest_vecs
    norms = np.linalg.norm(full, axis=1)
    assert norms.min() > 0.99, "a zero or unnormalised vector slipped through"
    np.save(FULL, full)
    print(f"→ {FULL}  форма {full.shape}")


def cmd_load():
    full = np.load(FULL)
    t0 = time.time()
    with connect() as conn:
        n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        assert n == len(full), f"{n} rows in chunks vs {len(full)} vectors"
        conn.execute(f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS "
                     f"{CFG['column']} vector({CFG['dim']})")
        # COPY into a temp table, then one UPDATE by id: 20k single-row
        # UPDATEs would take minutes.
        conn.execute(f"CREATE TEMP TABLE ruri_tmp (id int4, v vector({CFG['dim']}))")
        with conn.cursor() as cur:
            with cur.copy("COPY ruri_tmp (id, v) FROM STDIN WITH (FORMAT BINARY)") as cp:
                cp.set_types(["int4", "vector"])
                for i, v in enumerate(full):    # id = row number, as in db_init
                    cp.write_row((i, v))
        conn.execute(f"UPDATE chunks c SET {CFG['column']} = t.v "
                     f"FROM ruri_tmp t WHERE c.id = t.id")
        missing = conn.execute(f"SELECT count(*) FROM chunks "
                               f"WHERE {CFG['column']} IS NULL").fetchone()[0]
        assert missing == 0, f"{missing} rows without a Ruri vector"
        print(f"  векторы: {time.time() - t0:.0f}с")

        t1 = time.time()
        conn.execute("DROP INDEX IF EXISTS idx_chunks_hnsw_ruri")
        # Same parameters as BGE-M3's index (db_init.py) — one variable at a time.
        conn.execute(f"CREATE INDEX idx_chunks_hnsw_ruri ON chunks USING hnsw "
                     f"({CFG['column']} vector_ip_ops) WITH (m = 16, ef_construction = 64)")
        conn.commit()
        print(f"  индекс HNSW: {time.time() - t1:.0f}с")


def cmd_check():
    """Top-20 overlap of HNSW vs exact search over golden queries, core only."""
    meta = load_meta()
    core = np.array([i for i, c in enumerate(meta) if c["core"]])
    full = np.load(FULL)
    golden = [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]
    h = Hybrid(embedder=KEY)
    qv = [h.embed(g["q"]) for g in golden]

    exact = []
    for q in qv:
        s = full[core] @ q
        exact.append(set(core[np.argsort(-s)[:20]].tolist()))

    print(f"{'ef_search':>10} {'совпадение топ-20':>18} {'худший вопрос':>15}")
    for ef in (40, 100, 200, 400):
        h.conn.execute(f"SET hnsw.ef_search = {ef}")
        overlaps = []
        for q, ex in zip(qv, exact):
            rows = h.conn.execute(
                f"SELECT id FROM chunks WHERE core "
                f"ORDER BY {CFG['column']} <#> %s LIMIT 20", (q,)).fetchall()
            overlaps.append(len(ex & {r[0] for r in rows}) / 20)
        print(f"{ef:>10} {np.mean(overlaps):>18.3f} {min(overlaps):>15.2f}")
    print("1.000 = HNSW returns exactly the exact top-20. Take the smallest ef "
          "that reaches it; hybrid.EF_SEARCH applies to both columns.")


CMDS = {"encode": cmd_encode, "load": cmd_load, "check": cmd_check}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print(__doc__)
    else:
        CMDS[sys.argv[1]]()
