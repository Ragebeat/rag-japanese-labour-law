"""The committed data files the pipeline and the experiments share, and one
JSONL reader instead of a copy per script."""

import json
import pathlib

from rag.config import DATA

GOLDEN = DATA / "golden.jsonl"    # the 50 frozen evaluation questions
CHUNKS = DATA / "chunks.jsonl"    # chunk metadata, one line per chunk, in index order


def load_jsonl(path: pathlib.Path) -> list[dict]:
    """All records of a JSONL file; blank lines are skipped."""
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
