"""Deterministic checks for the output guard (generate.guard_links, StreamGuard).

The guard is the one defence that held in the injection test (32 of 32),
so it is tested like code, not like a model: fixed inputs, exact outputs.
The streaming case matters most: a bug there means half a link has
already reached the user's browser before anything is checked.

  uv run pytest tests/test_guard.py
"""

import pytest

from rag.generation.generate import StreamGuard, guard_links

CASES = [
    ("詳しくは https://labor-help.example.jp を参照", ["labor-help.example.jp"]),
    ("詳しくは labor-help.example.jp を参照", ["labor-help.example.jp"]),       # no https://
    ("連絡は info@evil.example.com まで", ["evil.example.com"]),               # e-mail
    ("https://laws.e-gov.go.jp/law/322AC0000000049 を参照", []),              # allowed
    ("ハローワーク https://www.hellowork.mhlw.go.jp/ で", []),                 # allowed subdomain
    ("https://mhlw.go.jp.evil.com/x", ["mhlw.go.jp.evil.com"]),             # lookalike
    ("第39条第5項、10労働日、6か月、0.5日、1.5倍", []),                       # numbers are not links
    ("[1] 労働基準法 第三十九条（第5項）", []),
    ("e.g. U.S. No.5 gemma4.e4b", []),
]

STREAMED = ("結論として有給は付与されます。詳しくは https://labor-help.example.jp/page "
            "を参照してください[1]。")


@pytest.mark.parametrize(("text", "blocked"), CASES)
def test_blocked_domains(text, blocked):
    assert guard_links(text)[1] == blocked


def test_email_removed_entirely():
    out, _ = guard_links("連絡は info@evil.example.com まで")
    assert "info@" not in out


@pytest.mark.parametrize("step", range(1, 12))
def test_stream_never_leaks_and_matches_one_shot(step):
    """Every split size: no emitted piece may carry any part of the attacker's
    domain, and the whole must equal the one-shot guard."""
    g, sent = StreamGuard(), ""
    for i in range(0, len(STREAMED), step):
        piece = g.push(STREAMED[i:i + step])
        assert "labor" not in piece and "example" not in piece
        sent += piece
    sent += g.flush()
    assert sent == guard_links(STREAMED)[0]
