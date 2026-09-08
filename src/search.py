"""Vector search over the corpus.

The vectors are normalised (|v| = 1), so dot equals cosine.

  python src/search.py "有給ってすぐもらえる？"
  python src/search.py "パワハラ対策？" --core
"""

import json
import pathlib
import sys

import httpx
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "chunks.jsonl"
VECTORS = ROOT / "data" / "vectors.npy"

MODEL = "bge-m3"
URL = "http://localhost:11434/api/embed"


class Index:
    def __init__(self):
        self.meta = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
        self.vecs = np.load(VECTORS)
        assert len(self.meta) == len(self.vecs), (
            f"чанков {len(self.meta)}, векторов {len(self.vecs)} — рассогласованы"
        )
        self.core_mask = np.array([c["core"] for c in self.meta])
        self._client = httpx.Client()

    def embed(self, text):
        r = self._client.post(URL, json={"model": MODEL, "input": text},
                              timeout=60)
        r.raise_for_status()
        return np.array(r.json()["embeddings"][0], dtype=np.float32)

    def search(self, query, k=10, core_only=False):
        q = self.embed(query)
        scores = self.vecs @ q
        if core_only:
            scores = np.where(self.core_mask, scores, -np.inf)
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return [(float(scores[i]), self.meta[i]) for i in top]


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    core_only = "--core" in args
    query = " ".join(a for a in args if a != "--core")

    idx = Index()
    n = int(idx.core_mask.sum()) if core_only else len(idx.meta)
    print(f"запрос: {query}")
    print(f"режим: {'core' if core_only else 'all'} ({n} чанков)\n")

    for rank, (score, c) in enumerate(idx.search(query, k=10,
                                                 core_only=core_only), 1):
        head = f"{c['law_title'][:22]} {c['article_title']}{c['article_caption']}"
        body = c["text"].split("\n", 1)[-1][:90]
        print(f"{rank:>2}. {score:.4f}  {head}")
        print(f"      {body}")


if __name__ == "__main__":
    main()