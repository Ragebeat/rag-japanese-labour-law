"""Run registry: each --eval is stored under a fingerprint of everything that
shapes the answer and logged with its metrics to data/runs.jsonl."""

import hashlib
import json
import subprocess
from datetime import datetime

from rag.config import ROOT
from rag.generation.context import ARTICLE_CAP, KW, POOL_MULT, TOP_K
from rag.generation.prompts import GEN_OPTIONS, MODEL
from rag.retrieval import hybrid

# Default context assembly mode. `whole` was chosen on measurements:
# recall@5 equals dedupe (0.88), over-refusal 2 -> 0, and no truncated
# articles. Not worse on any metric and better on two.
DEFAULT_MODE = "whole"


def report_path(mode, kind=""):
    """Reports are named after the assembly mode; otherwise, a month later,
    there is no telling which configuration produced which numbers. The cost
    of that confusion is known: one whole series of runs was measured on dedupe and
    had to be re-measured.
    """
    return ROOT / "data" / f"gen_report{kind}_{mode}.json"

# On 29.09 every --eval needed a manual Copy-Item first, or the new report
# overwrote the one to compare against, and the comparison itself was done
# by reading two JSON files. Now each run is kept under a fingerprint of
# everything that shapes the answer, and logged with its metrics.
#
# Bump CONTEXT_FORMAT by hand whenever build_context() / whole_article()
# change what the model sees: code is not part of the fingerprint, so a
# format change would otherwise hide under the same hash.
CONTEXT_FORMAT = "whole + 「（第N項）」 labels + 【用語の対応】 hint, 2026-10-02"
RUNS_DIR = ROOT / "data" / "runs"
RUNS_LOG = ROOT / "data" / "runs.jsonl"


def run_config(system, mode):
    """Everything that shapes an answer, read from the modules themselves:
    not retyped here, or this would become a fourth copy that drifts.
    The expander model is in: switching it (qwen → gemma) moved reranker
    scores enough to break an earlier "finding"."""
    from rag.retrieval import query_llm, rerank
    return {
        "generator": MODEL,
        "gen_options": GEN_OPTIONS,
        "system_sha": hashlib.sha256(system.encode()).hexdigest()[:12],
        "context_format": CONTEXT_FORMAT,
        "mode": mode,
        "retrieval": {"kw": KW, "top_k": TOP_K, "pool_mult": POOL_MULT,
                      "article_cap": ARTICLE_CAP},
        "expander": getattr(query_llm, "MODEL", None),
        "reranker": getattr(rerank, "MODEL", None),
        "reranker_revision": getattr(rerank, "REVISION", None),
        "embedder": hybrid.EMBEDDER,
    }


def fingerprint(config):
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def git_state():
    """Short commit, "+dirty" if src/ has uncommitted changes, then the
    commit alone does not reproduce the run."""
    try:
        run = lambda *a: subprocess.run(["git", *a], capture_output=True,  # noqa: E731
                                        text=True, cwd=ROOT, timeout=5).stdout.strip()
        head = run("rev-parse", "--short", "HEAD")
        return head + ("+dirty" if run("status", "--porcelain", "--", "src") else "")
    except Exception:                               # noqa: BLE001
        return None


def run_metrics(rows):
    """One summary per run, the same definitions for every run: that is
    what makes two runs comparable. Old reports without quote fields get
    None there, not zeros."""
    hit = [r for r in rows if r["hit"]]
    miss = [r for r in rows if not r["hit"]]
    ans = [r for r in rows if r["verdict"] == "ok"]
    cv = [r["citation_validity"] for r in rows if r.get("citation_validity") is not None]
    has_q = all("claims" in r for r in ans)

    def s(k):
        return sum(r.get(k, 0) for r in ans)

    def ratio(a, b):
        return round(a / b, 3) if b else None

    return {
        "n": len(rows),
        "recall": ratio(len(hit), len(rows)),
        "over_refusal": sum(1 for r in hit if r["verdict"] == "no"),
        "refusal_at_miss": ratio(sum(1 for r in miss if r["verdict"] == "no"), len(miss)),
        "no_verdict": sum(1 for r in rows if r["verdict"] not in ("ok", "no")),
        "bad_refs": sum(len(r.get("bad_arts", [])) for r in rows),
        "blocked_links": sum(len(r.get("blocked_links", [])) for r in rows),
        "citation_validity": ratio(sum(cv), len(cv)),
        "claim_coverage": ratio(s("claims_quoted"), s("claims")) if has_q else None,
        "text_coverage": ratio(s("quoted_chars"), s("claim_chars")) if has_q else None,
        "verbatim_coverage": ratio(s("verbatim_chars"), s("content_chars")) if has_q else None,
        "quotes": s("quotes") if has_q else None,
        "quote_bad": sum(len(r.get("quote_bad", [])) for r in ans) if has_q else None,
        "para_claims": s("para_claims") if has_q else None,
        "para_bad": sum(len(r.get("para_bad", [])) for r in ans) if has_q else None,
        "num_unquoted_answers": sum(1 for r in ans if r.get("num_unquoted")) if has_q else None,
    }


def log_run(rows, system, mode, label):
    """Keep this run's report under its own name and append a line to
    runs.jsonl. Two runs with the same fingerprint are a repeat, exactly
    what is needed to measure run-to-run noise."""
    config = run_config(system, mode)
    fp = fingerprint(config)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"{ts}_{label}_{fp}_{mode}.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    entry = {"ts": ts, "label": label, "fingerprint": fp, "git": git_state(),
             "report": str(path.relative_to(ROOT)), "config": config,
             "metrics": run_metrics(rows)}
    with RUNS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"→ run logged: {label} · fingerprint {fp} · {entry['git']} · {path.name}")
    return entry
