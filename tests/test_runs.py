"""The run fingerprint must change when model weights change under the same tag.

  uv run pytest tests/test_runs.py
"""

import httpx

from rag.generation import runs
from rag.generation.prompts import SYSTEM


class _Resp:
    def __init__(self, models):
        self._models = models

    def json(self):
        return {"models": self._models}


def _tags(*models):
    return lambda url, timeout: _Resp([{"name": n, "digest": d} for n, d in models])


def test_digests_by_name_and_latest_alias(monkeypatch):
    monkeypatch.setattr(runs.httpx, "get", _tags(("gemma4:e4b", "dc35e8d9c606aa"),
                                                 ("bge-m3:latest", "790764642607bb")))
    assert runs.ollama_digests(["gemma4:e4b", "bge-m3", "qwen3:8b"]) == {
        "gemma4:e4b": "dc35e8d9c606", "bge-m3": "790764642607", "qwen3:8b": None}


def test_ollama_down_gives_none(monkeypatch):
    def down(url, timeout):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(runs.httpx, "get", down)
    assert runs.ollama_digests(["gemma4:e4b"]) == {"gemma4:e4b": None}


def test_new_weights_new_fingerprint(monkeypatch):
    monkeypatch.setattr(runs.httpx, "get", _tags(("gemma4:e4b", "aaaaaaaaaaaa")))
    before = runs.fingerprint(runs.run_config(SYSTEM, "whole"))
    monkeypatch.setattr(runs.httpx, "get", _tags(("gemma4:e4b", "bbbbbbbbbbbb")))
    after = runs.fingerprint(runs.run_config(SYSTEM, "whole"))
    assert before != after
