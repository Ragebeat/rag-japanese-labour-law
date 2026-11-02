# Decisions

Changes to how the repository is built or measured, with the reason. Measurement decisions themselves are in the README next to their numbers.

## 1. One package instead of 40 scripts (2026-11-02)

`src/` held 40 flat scripts that imported each other through `sys.path.insert`, with `# noqa: E402` on every such import; `generate.py` had grown to 1 063 lines holding the prompts, context assembly, answer checks, the link guard, the run registry and the CLI.

Now `src/rag/` is a package installed by `uv sync` (`ingest`, `index`, `retrieval`, `generation`, `agent`), the measurement scripts live in `experiments/`, and the deterministic checks are pytest tests in `tests/`. `generate.py` is split into `prompts`, `context`, `checks`, `guard`, `runs` and `generate`. `ROOT`, the data paths and the JSONL reader exist once (`config.py`, `data.py`).

How it was checked: the deterministic tests passed before and after each step, and a golden-set run after the move received the same context for all 50 questions as the last run before it, with the same fingerprint. Commands in the README changed from `python src/x.py` to `python -m rag.<area>.x` or `python -m experiments.x`.

## 2. The model digest is part of the run fingerprint (2026-11-02)

The same check showed generation metrics outside the range of three earlier identical runs while retrieval was identical. The cause: `gemma4:e4b` had been re-pulled under the same tag. The fingerprint held the model name only, so new weights produced the old hash and looked like run-to-run noise.

`runs.run_config()` now records the Ollama digests of the generator and the query expander (`ollama_digests()`, from `/api/tags`). Fingerprints of runs from 2 November on therefore differ from the earlier ones, `251a4dd9` included; the earlier entries in `runs.jsonl` stay as they were. If Ollama is unreachable the digest is `None`, which also yields a distinct fingerprint rather than a false match.
