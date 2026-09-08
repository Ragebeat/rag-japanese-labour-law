"""Схема БД и заливка чанков с векторами и японским tsvector.

  python src/db_init.py schema   — создать таблицу (DROP + CREATE)
  python src/db_init.py load     — залить данные
  python src/db_init.py index    — построить индексы
  python src/db_init.py stats    — проверить, что получилось

Индексы строим ПОСЛЕ заливки — строить до значит перестраивать
дерево на каждом INSERT.
"""

import json
import os
import pathlib
import sys
import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tokenize_ja import to_tsvector_text  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "chunks.jsonl"
VECTORS = ROOT / "data" / "vectors.npy"

DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://rag:{}@localhost:5433/ragdb".format(
        os.environ.get("POSTGRES_PASSWORD", "")
    ),
)

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS chunks;

CREATE TABLE chunks (
    id              integer PRIMARY KEY,
    chunk_id        text NOT NULL,
    text            text NOT NULL,

    law_id          text NOT NULL,
    law_title       text NOT NULL,
    law_type        text NOT NULL,
    core            boolean NOT NULL,
    article_num     text,
    article_title   text,
    article_caption text,
    paragraph_num   text,
    item_num        text,
    level           smallint,
    n_chars         integer,

    embedding       vector(1024) NOT NULL,
    -- японский токенизируется Sudachi снаружи, в Postgres
    -- попадает уже разбитым по пробелам → 'simple' без стемминга
    tokens          text NOT NULL,
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('simple', tokens)) STORED
);
"""

INDEXES = [
    # HNSW по скалярному произведению: векторы нормализованы (|v|=1),
    # значит ip даёт тот же порядок, что cosine, но без лишнего деления
    ("idx_chunks_hnsw",
     "CREATE INDEX idx_chunks_hnsw ON chunks "
     "USING hnsw (embedding vector_ip_ops) WITH (m = 16, ef_construction = 64)"),
    ("idx_chunks_tsv",
     "CREATE INDEX idx_chunks_tsv ON chunks USING gin (tsv)"),
    ("idx_chunks_core",
     "CREATE INDEX idx_chunks_core ON chunks (core)"),
    ("idx_chunks_law",
     "CREATE INDEX idx_chunks_law ON chunks (law_id, article_title)"),
]


def connect():
    conn = psycopg.connect(DSN)
    register_vector(conn)
    return conn


def cmd_schema():
    with connect() as conn:
        conn.execute(SCHEMA)
        conn.commit()
    print("таблица chunks создана")


def cmd_load():
    meta = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
    vecs = np.load(VECTORS)
    assert len(meta) == len(vecs), f"{len(meta)} чанков vs {len(vecs)} векторов"

    t0 = time.time()
    with connect() as conn:
        with conn.cursor() as cur:
            with cur.copy(
                "COPY chunks (id, chunk_id, text, law_id, law_title, law_type,"
                " core, article_num, article_title, article_caption,"
                " paragraph_num, item_num, level, n_chars, embedding, tokens)"
                " FROM STDIN WITH (FORMAT BINARY)"
            ) as copy:
                copy.set_types([
                    "int4", "text", "text", "text", "text", "text",
                    "bool", "text", "text", "text",
                    "text", "text", "int2", "int4", "vector", "text",
                ])
                for i, (c, v) in enumerate(zip(meta, vecs)):
                    copy.write_row((
                        i, c["chunk_id"], c["text"], c["law_id"],
                        c["law_title"], c["law_type"], c["core"],
                        c["article_num"], c["article_title"],
                        c["article_caption"], c["paragraph_num"],
                        c["item_num"], c["level"], c["n_chars"],
                        v, to_tsvector_text(c["text"]),
                    ))
                    if i % 2000 == 0:
                        print(f"  {i}/{len(meta)}  {time.time()-t0:.0f}с")
        conn.commit()
    print(f"залито {len(meta)} за {(time.time()-t0)/60:.1f} мин")


def cmd_index():
    with connect() as conn:
        for name, sql in INDEXES:
            t0 = time.time()
            conn.execute(f"DROP INDEX IF EXISTS {name}")
            conn.execute(sql)
            conn.commit()
            print(f"  {name}: {time.time()-t0:.1f}с")
    print("индексы построены")


def cmd_stats():
    with connect() as conn:
        n, ncore = conn.execute(
            "SELECT count(*), count(*) FILTER (WHERE core) FROM chunks"
        ).fetchone()
        print(f"чанков: {n}   core: {ncore}")

        size = conn.execute(
            "SELECT pg_size_pretty(pg_total_relation_size('chunks'))"
        ).fetchone()[0]
        print(f"размер таблицы с индексами: {size}")

        for name, in conn.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename='chunks'"
        ).fetchall():
            isize = conn.execute(
                "SELECT pg_size_pretty(pg_relation_size(%s))", (name,)
            ).fetchone()[0]
            print(f"  {name:<22} {isize}")

        print("\nчастые токены (проверка Sudachi):")
        for word, cnt in conn.execute("""
            SELECT word, ndoc FROM ts_stat(
              'SELECT tsv FROM chunks'
            ) ORDER BY ndoc DESC LIMIT 12
        """).fetchall():
            print(f"  {word:<12} {cnt}")


CMDS = {"schema": cmd_schema, "load": cmd_load,
        "index": cmd_index, "stats": cmd_stats}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print(__doc__)
    else:
        CMDS[sys.argv[1]]()