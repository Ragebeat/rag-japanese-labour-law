"""Deterministic checks for the output guard (generate.guard_links, StreamGuard).

The guard is the one defence that held in the injection test (32 of 32),
so it is tested like code, not like a model: fixed inputs, exact outputs.
The streaming case matters most — a bug there means half a link has
already reached the user's browser before anything is checked.

  uv run python src/test_guard.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from generate import StreamGuard, guard_links  # noqa: E402

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


def check(name, got, want):
    ok = got == want
    print(f"{'✓' if ok else '✗'} {name}: {got}" + ("" if ok else f"  (expected {want})"))
    return ok


def run():
    r = [check(text[:28], guard_links(text)[1], want) for text, want in CASES]

    out, _ = guard_links("連絡は info@evil.example.com まで")
    r.append(check("e-mail removed entirely", "info@" in out, False))

    # Streaming: every split size; no emitted piece may carry any part of the
    # attacker's domain, and the whole must equal the one-shot guard.
    full = ("結論として有給は付与されます。詳しくは https://labor-help.example.jp/page "
            "を参照してください[1]。")
    leaked, mismatch = [], []
    for step in range(1, 12):
        g, sent = StreamGuard(), ""
        for i in range(0, len(full), step):
            piece = g.push(full[i:i + step])
            if "labor" in piece or "example" in piece:
                leaked.append(step)
            sent += piece
        sent += g.flush()
        if sent != guard_links(full)[0]:
            mismatch.append(step)
    r.append(check("stream: no fragment of the address leaks", leaked, []))
    r.append(check("stream: result equals the whole-text check", mismatch, []))
    return r


if __name__ == "__main__":
    res = run()
    bad = res.count(False)
    print(f"\n{len(res) - bad}/{len(res)}")
    sys.exit(1 if bad else 0)
