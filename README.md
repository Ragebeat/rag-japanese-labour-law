# RAG over Japanese Labour Law

A retrieval-and-answer system over the corpus of Japanese labour legislation. It answers colloquial questions (「有給って入社してすぐもらえるの？」 — "do I get paid leave right after joining?") with statutory provisions and precise source citations.

Built as a portfolio project, but with production-grade measurement discipline: **every improvement is backed by a number on a golden dataset, and every rejected design was rejected on evidence rather than taste.**

---

## Results

50 annotated questions, recall@5 on the core-corpus slice.

| Configuration | r@1 | r@5 | MRR | sec/query |
|---|---|---|---|---|
| naive vector search | 0.380 | 0.680 | 0.495 | 0.28 |
| BM25 alone | 0.200 | 0.320 | 0.257 | 0.011 |
| hybrid BM25 + RRF | 0.320 | 0.600 | 0.439 | 0.31 |
| + dictionary query expansion | 0.480 | 0.780 | 0.582 | 0.27 |
| + LLM fallback to the dictionary | 0.480 | 0.820 | 0.604 | 0.28 |
| + cross-encoder reranking | **0.600** | **0.860** | **0.720** | 15.2 |
| + article-level context assembly | 0.600 | **0.880** | 0.720 | 15.2 |

**recall@5: 0.68 → 0.88. MRR: 0.495 → 0.720.**

On the generation side: the citation contract holds in 50 out of 50 answers, and there are 0 citations to articles absent from the context.

---

## Interface

![interface](docs/screenshot.png)

The web UI is built to show **how the system works**, not just what it produced. The right-hand column lists everything retrieval returned, with reranker scores; articles the answer actually used are highlighted, the rest are dimmed. The screenshot shows the typical picture: five articles supplied, one used, and an order-of-magnitude gap in scores (0.768 against 0.054 and below).

The `[1]` markers in the answer are clickable and tie each statement to its source. The 回答可能 / 回答不可 verdict is rendered as a badge — it is a field the model is required to emit on the first line, not a phrase fished out of prose.

Responses arrive over SSE: sources are pushed as the first event, before generation starts.

The interface is available in Japanese and Russian. Questions are always Japanese — the corpus, the embeddings and the prompt are Japanese, so accepting other input would be misleading.

---

## Architecture

```
question
  ↓ query expansion: term dictionary → LLM as fallback
  ↓ vector search (BGE-M3 + pgvector, HNSW)        top-20
  ↓ cross-encoder reranking (bge-reranker-v2-m3)   top-20 → top-5 articles
  ↓ context assembly: each article in full
  ↓ generation under a citation contract (gemma4:e4b)
answer + [N] markers + an explicit "can / cannot answer" verdict
```

Everything runs locally: Ollama, Postgres in Docker, a GTX 1060 6GB.

### Corpus

**e-Gov 法令検索 API**, category 020 (Labour): 259 documents, 34 MB of XML.

Japan has no single labour code. The law is spread across separate statutes: 労基法 is central, but parental leave lives in 育児介護休業法, workplace harassment in 労働施策総合推進法, agency work in 派遣法. On top of that sits a layer of cabinet and ministerial ordinances, which the statute defers to with the phrase 「厚生労働省令で定める」 — and that is where the operative detail actually lives.

A **core subset of 35 documents** was selected by a formal rule (14 statutes plus their implementing ordinances), with two manual exceptions the rule cannot capture. The subset is a **flag in chunk metadata**, not a separate index — which made the cost of noise measurable in a single run: **+0.08 recall@5** from excluding unrelated statutes.

### Chunking

Derived from the corpus distribution rather than copied from a tutorial. Article length spans **260× between the median and the 99th percentile** (200 characters vs 2,372, max 21,157). A fixed 512-token window would leave 90% of chunks half empty and still cut the tail mid-sentence.

Three levels:

| Level | Rule | Chunks | Share |
|---|---|---|---|
| 1 | whole paragraph | 17,805 | 87.6% |
| 2 | paragraph > 800 chars → split by item | 2,032 | 10.0% |
| 3 | item > 800 → split by sentence, with overlap | 325 | 1.6% |

**20,162 chunks total**, 7,452 of them in the core.

Every chunk carries a context prefix: `労働基準法 第三十六条（時間外及び休日の労働）第1項`. Without it a bare paragraph is often meaningless — 「前項の協定においては」 ("under the agreement in the preceding paragraph") with no indication of which paragraph that is.

Sentence splitting on 。 alone produced fragments starting with `）` and `」`: the full stop occurs **inside brackets** in legal Japanese. Enforcing bracket balance removed 167 false splits.

### Storage

Postgres 17 + pgvector 0.8.6 in Docker. Vector and full-text indexes live in one table; the core filter is a pre-filter (`WHERE core`), which guarantees k results.

Japanese full-text search runs on **Sudachi outside the database**: text is tokenised and stored space-separated, because the built-in `to_tsvector` would turn an entire Japanese article into a single token.

**HNSW loses results silently.** After migrating from numpy to pgvector, the correct article disappeared from the top 10. The cause was the default `hnsw.ef_search = 40`; at 100 the ranking matched exact search again. The index does not fail or warn — it simply returns a list without the right answer.

### Japanese-specific problems

A layer of issues that does not exist in English corpora:

- **No spaces between words** → BM25 requires morphological analysis. Sudachi in mode A (fine-grained): in mode C a query for 有給 would never match 年次有給休暇
- **`normalized_form()` returns a canonical orthography, not a dictionary form**: `する → 為る`, `十 → 10`, `雇入れ → 雇い入れ`. Those spellings appear neither in the statute nor in the query. Solution: filter on the normalised form, index the surface form
- **Article numbers are written inconsistently**: 第39条 / 第三十九条 / 第３９条. Normalisation lives in its own module (`jnum.py`) and comparison happens on canonical pairs
- **Cross-references**: 「法第二十五条第一項の…」. An article named that way in an answer is neither a hallucination nor a source — it is a third category and is counted separately

---

## What did not work

This is the most useful section. Half the work went into rejected designs, and each was rejected with a number attached.

### Hybrid BM25 + RRF: −0.18 recall@5

The textbook argument — "vector search alone loses exact terms" — assumes the user writes in the same words as the document. Here the opposite holds: colloquial Japanese against bureaucratic Japanese.

| Query | Tokens BM25 can actually use |
|---|---|
| 有給って入社してすぐもらえるの？ | 1 |
| パワハラ対策は会社の義務？ | 0 (the statute says 優越的な関係を背景とした言動) |
| 育休っていつまで取れる？ | 0 (the statute says 育児休業) |

RRF fixed 3 questions and **broke 12**. The mechanism: on a common word such as 会社, BM25 returned fifty near-random candidates, and fusion gave them the same weight as the vector's exact hits.

> **Hybrid retrieval wins when the two methods fail on different queries.** Here they failed on the same ones.

The gradient reads directly: the smaller the BM25 contribution, the better — 0.50 → 0.54 → 0.58 → 0.60, converging on 0.68 with BM25 switched off entirely.

### Expanding the golden set to match what was retrieved

Analysis showed that four of the six apparent retrieval misses were **artefacts of our own annotation**: the golden set labels a statutory article, while the working answer lives in the implementing ordinance.

A formal rule was written ("an ordinance article counts if it explicitly references the statutory article via 法第N条"), applied to all 50 questions, and produced 193 candidates. After filtering out hub articles (労基則第六条の二 references 16 parent articles at once), 83 remained.

It was still rejected: for "do I get leave right after joining", the rule added articles about how to compute pay during leave. The rule was applied flawlessly and **not one of the additions answers the question**.

> **"Implements article X" and "answers this question about X" are different relations.** The first is encoded in the structure of the corpus. The second is not in the corpus at all — it is a property of the question, and the corpus knows nothing about the question.

**The golden set is deliberately frozen.** The metric compares configurations against each other; for that, annotation must be stable, not perfect.

### A verification pass as a gate

Four judges were evaluated: qwen3:8b, qwen3.5, gemma4:31b, gpt-oss:120b.

| | 8b | 3.5 | 31b | 120b |
|---|---|---|---|---|
| found a missing provision (of 4) | **3** | 0 | 0 | 1 |
| found an inapplicable citation (of 8) | 7 | 6 | **8** | 7 |
| share of answers blocked | 0.65 | 0.65 | 0.65 | 0.80 |

**Applicability checking works** — it catches the case where the model cites a correct article that does not govern the question. **Completeness checking works on none of the models.**

> Judge quality on this task **is not a function of model size**: the spread between two 8B models is wider than the spread between 8B and 120B.

It cannot serve as a gate: two thirds of all answers get flagged regardless of which judge is used.

### A structural completeness check

Since the models failed, the same class was attacked structurally: a paragraph counts as omitted if the answer does not use it while the reranker scores it no lower than the paragraph the answer did use. A rule with no tunable threshold.

It failed for a fundamental reason: **the cross-encoder does not discriminate between paragraphs within a single article.** All ten paragraphs of 労基法39条 fell within `+0.000 … +0.041`, whereas the same reranker separates a relevant from an irrelevant article by `+0.934` against `+0.369` — twenty times wider.

The signal is absent in both the LLM and the reranker. Tuning a rule on top of noise is not worth doing.

---

## Refusal when the source is absent

A separately measured question: what does the system do when the governing article is not in the context?

Only 6 of 50 questions were natural retrieval misses — too few to separate signal from chance. So a **controlled ablation** was built: for the 44 questions where retrieval succeeded, the gold article was forcibly removed from the candidate pool and replaced by the next-best candidate.

The design is paired — same questions, the only difference being the presence of the source. "Some questions are simply harder" is excluded by construction.

| Method | Refusal when source absent | False refusal |
|---|---|---|
| model verdict via prompt | 0.25 | 0.00 |
| threshold on reranker score | 0.18 | 0.05 |
| union of the two | **0.41** | 0.07 |

The paired contingency table is strictly one-sided: all 11 discordant pairs point the same way, none the other. The signal is real but weak.

Of the 30 answers produced without a source, **16 substitute a neighbouring institution**: the model answers with an accurate quotation from the wrong statute. The worst case: asked "I was fired without notice, is that allowed?" with 労基法20条 removed, the system answered from 労組法15条 (termination of a collective agreement) and reported a **ninety-day** notice period instead of thirty. Every automatic metric rated this answer as perfect — markers in range, citation valid, zero fabricated references.

> **For a legal assistant the danger is not an invented number. It is a correct quotation of the wrong provision, delivered fluently and with confidence.**

This also explains why combining the model verdict with the score threshold beats either alone: they fail on different questions (only 4 overlaps out of 18). Precisely the condition BM25 and vector search did not satisfy.

---

## Evaluation

### Golden dataset

50 pairs of "question → list of acceptable articles", keyed on `law_id` plus article number. The questions are colloquial; every article number was verified against the statutory text.

recall@k counts as a hit if **at least one** relevant article appears in the top k: the user needs an answer, not an exhaustive digest.

### Generation metrics

- **citation contract** — an `[N]` marker per chunk plus a `判定: 回答可能 / 回答不可` verdict on the first line. Held 50/50
- **citation validity** — 1.000, zero fabricated references. This metric serves as a regression guard, not a quality indicator: the model can only cite what it can see
- **false refusal** — 0 of 44 in the working configuration
- **manual annotation of 20 answers** across three independent flags: substantive answer / distortion of content / extraneous material

### One technique that worked three times

Instead of parsing meaning out of free text, **require a structural token that we control**:

- a chunk number `[N]` instead of an article number (the model writes article numbers however it likes; chunk numbers are ours)
- `判定:` on the first line instead of a refusal buried in prose (before this, refusal appeared in ten different phrasings and never cleanly)
- the list of cited markers passed to the judge explicitly rather than inferred by it

Each time this replaced a heuristic with an exact match.

---

## Caveats

Read these before using the numbers.

- **`recall@5 = 0.88` is understated by our own annotation.** Four of the six misses were manually verified as golden-set artefacts. The real ceiling is ≈ 0.96. The annotation was deliberately not revised after measurement began
- **The manual annotation of 20 answers was produced by a model** (Claude, via an agent), not a human. It is a second opinion with access to the primary source, not a human gold standard
- **Two of the four judges are cloud-hosted** (`gemma4:31b-cloud`, `gpt-oss:120b-cloud`). The rest of the system is fully local
- **All generation numbers were obtained with `gemma4:e4b`.** Changing the generator makes them incomparable
- **The golden set is 50 questions.** A difference below 0.02 on this sample is noise and is not grounds for choosing a configuration

---

## Open problems

- **Paragraph selection within an article is uncontrolled.** The reranker does not discriminate between paragraphs, the generator picks arbitrarily, the judge sees no difference. The fix belongs earlier in the pipeline: require the generator to name the applicable paragraph explicitly
- A router for queries that cite an article number directly — lexical lookup beats vector search there, but it needs its own evaluation slice
- **Latency.** A full answer takes about 30 seconds on a GTX 1060: roughly 15 for reranking twenty pairs, the rest for generation. Candidates for speeding it up: reranking the top 10 instead of 20, or a quantised cross-encoder — either changes the configuration and requires re-measuring everything
- RAGAS, as a cross-check against this project's own metric implementation

---

## Running it

```bash
docker compose up -d                          # Postgres + pgvector
uv sync

uv run python src/fetch_manifest.py           # statute list from e-Gov
uv run python src/download_laws.py            # statute bodies (XML)
uv run python src/analyze.py                  # corpus statistics
uv run python src/chunk.py                    # chunking
uv run python src/embed.py                    # embeddings (~25 min)
uv run python src/db_init.py                  # load + indexes

uv run python src/generate.py "有給って入社してすぐもらえるの？"
uv run python src/generate.py --eval          # golden-set run
uv run python src/generate.py --ablate        # ablation
uv run python src/verify.py --calibrate       # judge calibration
```

Web:

```bash
uv run uvicorn src.api:app --port 8000        # API + SSE, Swagger at /docs

cd web && npm install && npm run dev          # UI on :5173
cd web && npm run build                       # build; served by the same FastAPI
```

Environment variables: `GEN_MODEL`, `JUDGE_MODEL`, Postgres credentials — all in `.env`.

## Stack

Python 3.14 · Postgres 17 + pgvector · Ollama (BGE-M3, gemma4:e4b) · sentence-transformers (bge-reranker-v2-m3) · SudachiPy · FastAPI + SSE · React + Vite · httpx · numpy
