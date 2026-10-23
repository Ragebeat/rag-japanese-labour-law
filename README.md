# RAG over Japanese Labour Law

*Languages: English · [Русский](README.ru.md) · [日本語](README.ja.md)*

> **recall@5 0.94** on a frozen 50-question golden set · fully local (Ollama, Postgres + pgvector, a GTX 1060) · every rejected design was rejected with a number.

A retrieval-and-answer system over the corpus of Japanese labour legislation. It answers colloquial questions (「有給って入社してすぐもらえるの？」 — "do I get paid leave right after joining?") with statutory provisions and precise source citations.

Built as a portfolio project, but with production-grade measurement discipline: **every improvement is backed by a number on a golden dataset, and every rejected design was rejected on evidence rather than taste.**

**Contents:** [Results](#results) · [Interface](#interface) · [Architecture](#architecture) · [What did not work](#what-did-not-work) · [Refusal when the source is absent](#refusal-when-the-source-is-absent) · [Security and operations](#security-and-operations) · [Evaluation](#evaluation) · [Caveats](#caveats) · [Open problems](#open-problems) · [Where the numbers come from](#where-the-numbers-come-from) · [Running it](#running-it) · [Repository layout](#repository-layout) · [Stack](#stack) · [Extension: the RAG as a tool for an agent](#extension-the-rag-as-a-tool-for-an-agent) · [License](#license)

---

## Results

50 annotated questions, recall@5 on the core-corpus slice.

| Configuration | r@1 | r@5 | MRR | sec/query |
|---|---|---|---|---|
| naive vector search | 0.380 | 0.680 | 0.495 | 0.28 |
| BM25 alone | 0.200 | 0.320 | 0.257 | 0.011 |
| hybrid BM25 + RRF | 0.320 | 0.500 | 0.414 | 0.33 |
| + dictionary query expansion | 0.480 | 0.780 | 0.582 | 0.27 |
| + LLM fallback to the dictionary | 0.480 | 0.820 | 0.604 | 0.28 |
| + cross-encoder reranking | **0.600** | **0.860** | **0.720** | 15.2 |
| + article-level context assembly | 0.600 | **0.880** | 0.720 | 15.2 |

**BGE-M3 pipeline: recall@5 0.68 → 0.88, MRR 0.495 → 0.720. Adopted configuration (Ruri-v3 + reranker): recall@5 0.94** — see below.

### Swapping the embedder: BGE-M3 → Ruri-v3

Bare vector search on the core slice, no expansion and no reranker, same 50 questions:

| Model | r@1 | r@5 | r@10 | r@20 | MRR |
|---|---|---|---|---|---|
| bge-m3 | 0.38 | 0.68 | 0.76 | 0.92 | 0.51 |
| multilingual-e5-large | 0.52 | 0.76 | 0.78 | 0.86 | 0.61 |
| **ruri-v3-310m** | **0.56** | **0.86** | **0.94** | **1.00** | **0.71** |

r@20 is reported because the reranker works on a pool of 20: it is the ceiling for what comes after. By r@5 alone e5 beats BGE-M3, by r@20 it loses.

Full pipeline:

| Configuration | r@5 | sec/query |
|---|---|---|
| bge-m3 + expansion + reranking | 0.86 | 15.6 |
| ruri-v3 alone | 0.86 | 0.2 |
| **ruri-v3 + reranking** | **0.94** | 15.2 |
| ruri-v3 + expansion + reranking | 0.96 | 14.0 |

Bare Ruri equals the whole tuned BGE-M3 pipeline. The configuration adopted is Ruri + reranker **without** query expansion: the 0.02 that expansion adds is one question, and the dictionary is the one component developed against this golden set, so its gain on this set is optimistic. 0.94 is the figure without that bias.

Generation, same prompt, BGE-M3 → Ruri: context recall 0.88 → 0.94, claim coverage by verbatim quotes 0.57 → 0.64, text matching the source 0.62 → 0.68, paragraph-label errors 3 → 0. The adopted configuration adds one prompt line (copy the article number as written in the heading); on it, means of three runs: claim coverage 0.62, text matching the source 0.64, one paragraph-label error (#39, see Caveats).

### Verbatim quotes, checked by code

A correct article with a wrong detail is the failure that citation markers do not catch. The generator is required to quote the statute verbatim in 「…」, and code checks every quote against the cited chunk and the cited paragraph (項). In the adopted configuration (mean of three runs) **0.62 of claims are covered by verified quotes**, and 0.64 of the answer text matches the source verbatim. The check also caught the first case of "real text under the wrong marker" automatically.

On the generation side (BGE-M3 runs): the citation contract holds in 50 out of 50 answers, and there are 0 citations to articles absent from the context. The Ruri runs are not at that level yet: see Caveats.

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
  ↓ vector search (Ruri-v3-310m in-process + pgvector HNSW, ef_search 200)  top-20
  ↓ cross-encoder reranking (bge-reranker-v2-m3)   top-20 → top-5 articles
  ↓ context assembly: each article in full
  ↓ term hint: colloquial → statutory pairs after the question 【用語の対応】
  ↓ generation under a citation contract with verbatim quotes (gemma4:e4b)
  ↓ output guard: links checked against an allow-list, also mid-stream
answer + [N] markers + an explicit "can / cannot answer" verdict
```

The earlier pipeline (BGE-M3 + query expansion: term dictionary, LLM as fallback) is still reproducible: `EMBEDDER=bge-m3` and the configurations in `sweep.py`.

Everything runs locally: Ollama, Postgres in Docker, a GTX 1060 6GB; Ruri and the reranker run on CPU.

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

**HNSW loses results silently.** After migrating from numpy to pgvector, the correct article disappeared from the top 10. The cause was the default `hnsw.ef_search = 40`; at 100 the ranking matched exact search again. The index does not fail or warn — it simply returns a list without the right answer. (100 was enough for BGE-M3; after the switch to Ruri it is 200 — see below.)

### Japanese-specific problems

A layer of issues that does not exist in English corpora:

- **No spaces between words** → BM25 requires morphological analysis. Sudachi in mode A (fine-grained): in mode C a query for 有給 would never match 年次有給休暇
- **`normalized_form()` returns a canonical orthography, not a dictionary form**: `する → 為る`, `十 → 10`, `雇入れ → 雇い入れ`. Those spellings appear neither in the statute nor in the query. Solution: filter on the normalised form, index the surface form
- **Article numbers are written inconsistently**: 第39条 / 第三十九条 / 第３９条. Normalisation lives in its own module (`jnum.py`) and comparison happens on canonical pairs
- **Cross-references**: 「法第二十五条第一項の…」. An article named that way in an answer is neither a hallucination nor a source — it is a third category and is counted separately
- **Colloquial wording is not statutory wording.** 失業保険 does not occur in the Employment Insurance Act; the article says 基本手当. Retrieval put the article first, yet the generator refused on that same context 9 times out of 10; the question rewritten as 基本手当 gave 0 refusals out of 10. A general prompt rule fixed it at a price (see *What did not work*). What worked was showing the generator the colloquial → statutory pairs the search dictionary already holds, as a `【用語の対応】` line after the question
- **Width variants of the query.** Measured on perturbed copies of the 50 questions: full-width digits and Latin (３６協定) cost nothing, the embedder handles them; half-width kana (ﾊﾟﾜﾊﾗ) lost 2 of the 11 questions it touched; a trailing space lost 1. `fold_query()` converts half-width kana and strips the query. Full NFKC was rejected: it also turns ？ and （） into ASCII, away from the form used in the corpus. Old/new kanji forms were not measured
- **Index parameters belong to the model.** Overlap of the HNSW top-20 with exact search on Ruri: `ef_search` 40 → ~0.91, 100 → ~0.97, 200 → ≥0.995 (the index build is randomized, so overlap shifts by about 0.01 between rebuilds). A value tuned on another model silently loses part of the candidate pool; 200 is used

### The term hint, and how to extend the dictionary

The generator sees the question as typed, so a colloquial word that the statute never uses (失業保険 for 基本手当) can make it refuse even when retrieval put the right article first. `term_hint()` in `generate.py` looks the question up in `data/synonyms.json` (27 hand-written colloquial → statutory pairs) and appends the first statutory term of every key found as one line after the question: `【用語の対応】失業保険＝基本手当`. Matching is a plain substring test: no LLM, no extra latency. Retrieval does not use the dictionary (vector mode, expansion off); it only informs the generator.

Measured effect, three runs against three controls: one over-refusal removed (#22: 9 refusals in 10 → 0 in 10), six questions better in every run, four worse in every run (#16, #36, #39, #40; #39 is a metric artefact).

To extend it:

1. Add a pair only after checking that the statutory term occurs in the corpus text (`toc.py`).
2. One term per key. A longer list pulls the model toward neighbouring articles, the same dilution seen in search.
3. Short keys also match inside longer words. Check each new key against every question it will touch (e.g. 組合 inside 健康保険組合).
4. Compare three `--eval` runs with the previous configuration, never one: coverage, over-refusal, refusal at a miss and the per-question regression list. A loss repeated in every run belongs to the dictionary.
5. Test new entries on questions that are not in the golden set. The current gain is measured on the vocabulary the dictionary was written next to, and is optimistic.

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

### A prompt rule against over-refusal

Two questions got a refusal with the gold article in the context. One is a false premise ("will five years of work make me a regular employee?": the article is about conversion to an open-ended contract, not to 正社員; refused 10 times out of 10, which is defensible). The other is the vocabulary gap described above.

A general rule in the prompt ("a colloquial wording that differs only in vocabulary is not a reason to refuse") removed one over-refusal and cost quality: claim coverage 0.63 → 0.57 (means of three runs each, old prompt as the control; the ranges do not overlap), with regressions on four other questions. Reverted. A vocabulary problem is better fixed in the data the model sees than in the rules it follows.

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

## Security and operations

Reviewed against the OWASP Top 10 for LLM Applications (2026 edition).

### Indirect prompt injection, measured

A poisoned document was placed in the context with an instruction to append a link to every answer. Defences in the prompt (marking documents as data, delimiter tags, escaping) only lower the odds: the model followed the instruction in **4–8 of 8** questions under every prompt defence, and one sentence of stronger wording beat delimiting and escaping **8 of 8**. A code check on the output held **32 of 32**.

> **For a class of harm that can be checked mechanically, the defence belongs in code, not in a request to the model.**

The check is an allow-list of domains (e-Gov, MHLW), not "is the link in the sources" — a poisoned source contains it. Bare domains and e-mail addresses count. In streaming, a link is held back until it is complete and checked, so half of one never reaches the client. Links are only one kind of harm; the allow-list does not catch a phone number or a changed meaning, which is why the corpus itself is restricted to one trusted source.

### Operations

- **Traces without content by default.** Every step is traced (Phoenix + OpenTelemetry), but questions, prompts, answers and tool arguments are masked unless `TRACE_CONTENT=1`; the length survives. Every latency finding in this project came from metadata alone, and full traces would otherwise be a store of user questions and agent memory. Exceptions are recorded by type only. Phoenix binds to localhost
- **API.** No CORS middleware (it was `*`, and nothing needed it: dev goes through the Vite proxy, the built UI is same-origin). One request at a time with a bounded FIFO queue — the client sees its position over SSE, 429 only when the queue is full. Retrieval runs off the event loop; if a client leaves mid-rerank, the slot is held until the thread finishes. Errors go out as codes, details to the log
- **Agent memory** (`agent_graph.py`) is isolated per `user_id`, a required parameter, validated because it becomes a file name; a thread records its owner and is checked before its history is read
- **Pinned versions**: reranker and Ruri by Hugging Face commit (the reranker's is part of the run fingerprint), Phoenix by image digest

---

## Evaluation

### Golden dataset

50 pairs of "question → list of acceptable articles", keyed on `law_id` plus article number. The questions are colloquial; every article number was verified against the statutory text.

recall@k counts as a hit if **at least one** relevant article appears in the top k: the user needs an answer, not an exhaustive digest.

### Generation metrics

- **citation contract** — an `[N]` marker per chunk plus a `判定: 回答可能 / 回答不可` verdict on the first line. Held 50/50
- **citation validity** — 1.000, zero fabricated references. This metric serves as a regression guard, not a quality indicator: the model can only cite what it can see
- **false refusal** — 0 of 44 on the BGE-M3 pipeline; 1 of 47 on Ruri (#24, the false-premise question described in *What did not work*)
- **context truncation** — none. Ollama silently trims a prompt that exceeds its default window; `check_ctx.py` found 0 of 50 prompts cut, and the window is now set explicitly (`num_ctx` 32768)
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

- **`recall@5 = 0.88` (the BGE-M3 pipeline) is understated by our own annotation.** Four of the six misses were manually verified as golden-set artefacts. The real ceiling is ≈ 0.96. The annotation was deliberately not revised after measurement began — which is also why Ruri's 0.94 is close to what this golden set can show at all
- **The manual annotation of 20 answers was produced by a model** (Claude, via an agent), not a human. It is a second opinion with access to the primary source, not a human gold standard
- **Two of the four judges are cloud-hosted** (`gemma4:31b-cloud`, `gpt-oss:120b-cloud`). The rest of the system is fully local
- **All generation numbers were obtained with `gemma4:e4b`.** Changing the generator makes them incomparable
- **The golden set is 50 questions.** A difference below 0.02 on this sample is noise and is not grounds for choosing a configuration
- **Generation is not deterministic even at temperature 0.** On one fixed context the same question gave 1 refusal in 3 runs in one session and 9 in 10 in another; claim coverage of one configuration moves by 0.04–0.08 between identical runs. Configurations here are compared over three runs, with control reruns of the baseline
- **The query-expansion dictionary and the generator's term hint are one file** (`data/synonyms.json`, 27 pairs). Ten of its keys never occur in the golden questions, but 25 of the 50 questions contain one, and a check on fresh questions was not done. The gain on this set is optimistic
- **Citation validity on the Ruri runs is below 1.000**: 2 invented article references across 50 answers in the adopted configuration, the same two in all three runs (#3, #20). Both are garbled article numbers in running text (e.g. "第十六十六条" for 第六十六条); the label after the `[N]` marker is correct. A prompt line telling the model to copy the article number as written in the heading removed a third one, a wrong label (#36). The 50/50 figure above is from the BGE-M3 runs
- **One paragraph-label error in the adopted configuration** (#39, in all three runs): a four-character quote 「休業給付」 is found in a neighbouring paragraph of the cited article. #39 is a retrieval miss, and the error is closer to a metric artefact than to a wrong citation

---

## Open problems

- **Paragraph selection within an article is uncontrolled.** The reranker does not discriminate between paragraphs, the generator picks arbitrarily, the judge sees no difference. The fix belongs earlier in the pipeline: require the generator to name the applicable paragraph explicitly
- A router for queries that cite an article number directly — lexical lookup beats vector search there, but it needs its own evaluation slice
- **Latency.** A full answer takes about 30 seconds on a GTX 1060: roughly 15 for reranking twenty pairs, the rest for generation. Candidates for speeding it up: reranking the top 10 instead of 20, or a quantised cross-encoder — either changes the configuration and requires re-measuring everything
- **The score-based refusal gate does not work on Ruri.** AUC 0.62 between hits and misses; the median top-1 score of a miss (+0.535) is higher than that of a hit (+0.363), because the reranker is confident in a neighbouring article. With 3 natural misses in 50 questions it cannot be re-tuned: revisit when the golden set has at least 10
- RAGAS, as a cross-check against this project's own metric implementation

---

## Where the numbers come from

Every figure above is backed by a script in `src/` and a result file in `data/`. The result files are committed: they are measurement records and cannot be regenerated without the full local setup.

| Claim | Script | Result |
|---|---|---|
| Retrieval table (BGE-M3 rows); core vs full corpus, +0.08 | `eval.py` · `eval_hybrid.py` | `eval_report.json` · `eval_hybrid.json` |
| Hybrid BM25: weight gradient; reranker pool size | `sweep.py` · `sweep.py --pool` | `sweep.json` · `sweep_pool.json` |
| Embedder comparison (bare vector search) | `embed_compare.py` | `embed_compare.json` |
| Full pipeline on Ruri: 0.94, 0.96 | `sweep.py --ruri` | `sweep_ruri.json` |
| HNSW `ef_search` overlap with exact search | `embed_ruri.py check` | console output |
| Width variants of the query | `perturb_test.py` | console output |
| Generation metrics, every prompt change (three runs each) | `generate.py --eval` · `compare.py` | `runs.jsonl` · `runs/` |
| Context assembly: whole articles vs deduplicated chunks | `generate.py --eval` | `gen_report_whole.json` · `gen_report_dedupe.json` · `gen_report_nodedupe.json` |
| Verbatim-quote check, re-scored on saved answers | `rescore_quotes.py` | `gen_report_quotes_*.json` |
| Golden-set expansion (rejected) | `expand_golden.py` | `golden_proposed.jsonl` · `golden_review.md` |
| Judges table; manual annotation of 20 answers | `verify.py --calibrate` · `make_review.py` | `verify_calibrate_*.json` · `manual_review.md` |
| Structural completeness check | `verify.py --structural` | `omission_*.json` |
| Refusal when the source is absent | `generate.py --ablate` · `score_gate.py` · `margin_gate.py` | `gen_report_ablate_*.json` · `score_gate.json` · `margin_gate.json` |
| Indirect prompt injection | `injection_test.py` · `injection_test.py --escape` | `injection.json` · `injection_escape.json` |
| Context truncation: 0 of 50 prompts cut | `check_ctx.py` | `ctx_check.json` |

Run fingerprints in `data/runs.jsonl`: `8b80e255` BGE-M3 with the quote prompt · `abda3d7e` Ruri, same prompt · `a47039ed` the rejected over-refusal rule · `cfa11f74` the term hint · `251a4dd9` the adopted configuration (term hint + copy article numbers as written). `compare.py` diffs any two of them question by question.

---

## Running it

There are two paths. **Run the system** is the minimum to get answers: Ruri, the reranker and the generator. **Reproduce the experiments** adds what the numbers in this README were measured with, including the earlier BGE-M3 pipeline.

### Run the system

Needs: Docker, [uv](https://docs.astral.sh/uv/) (installs Python 3.14 from `.python-version`), [Ollama](https://ollama.com) with `ollama pull gemma4:e4b`, Node.js for the web UI. Ruri-v3 and the reranker are downloaded from Hugging Face on first use (pinned commits) and run on the CPU.

```bash
cp .env.example .env                          # PowerShell: Copy-Item .env.example .env; set POSTGRES_PASSWORD
docker compose up -d --wait                   # Postgres + pgvector; --wait: until it accepts connections
uv sync

# data/manifest.json is committed and pins each statute to a revision (law_revision_id);
# fetch_manifest.py rebuilds it from today's e-Gov list — only to move to newer texts.
uv run python src/download_laws.py            # statute bodies (XML), the pinned revisions
uv run python src/chunk.py                    # chunking
uv run python src/db_init.py schema           # table (drops it if it exists)
uv run python src/db_init.py load             # chunks; the BGE-M3 column stays empty without embed.py
uv run python src/db_init.py index            # indexes, after loading
uv run python src/embed_ruri.py encode        # Ruri for the whole corpus: CPU, cached in data/emb/, resumable
uv run python src/embed_ruri.py load          # column embedding_ruri + HNSW

uv run python src/generate.py "有給って入社してすぐもらえるの？"
uv run python src/test_quotes.py; uv run python src/test_guard.py; uv run python src/test_agent.py; uv run python src/test_memory.py
```

Web:

```bash
uv run uvicorn src.api:app --port 8000        # API + SSE, Swagger at /docs

cd web && npm install && npm run dev          # UI on :5173
cd web && npm run build                       # build; served by the same FastAPI
```

### Reproduce the experiments

On top of the system above. Also needs `ollama pull bge-m3`, and `ollama pull qwen3:8b` for `verify.py`. Every generation number in this README was measured on `gemma4:e4b`.

```bash
uv run python src/analyze.py                  # corpus statistics behind the chunking rules
uv run python src/embed.py                    # BGE-M3 vectors (~25 min)
uv run python src/db_init.py schema           # reload with the BGE-M3 column filled
uv run python src/db_init.py load
uv run python src/db_init.py index
uv run python src/embed_ruri.py load          # schema dropped the Ruri column; the vectors are cached
uv run python src/embed_ruri.py check         # HNSW vs exact search per ef_search
uv run python src/embed_compare.py            # embedder comparison (all models; --only ruri for one)

uv run python src/generate.py --eval          # golden-set run, logged to data/runs.jsonl
uv run python src/compare.py                  # diff the last two runs, per question
uv run python src/generate.py --ablate        # refusal when the source is absent
uv run python src/verify.py --calibrate       # judge calibration
```

Every other experiment is listed with its script and result file in *Where the numbers come from*; each script's docstring gives its exact command.

## Repository layout

Every script starts with a docstring that says what it does and how to run it. ▶ marks a script you run; the rest are modules imported by others.

**Corpus and index** (see *Running it*: system path, then experiments path)
- ▶ `fetch_manifest.py` — the statute list with pinned revisions (`data/manifest.json`, committed; rerun only to move to newer texts)
- ▶ `download_laws.py` — statute XML by the pinned revisions
- ▶ `analyze.py` — corpus statistics the chunking rules were derived from (experiments)
- ▶ `chunk.py` — three-level chunking
- ▶ `embed.py` — BGE-M3 vectors through Ollama: the earlier pipeline, experiments only
- ▶ `db_init.py` — Postgres schema, load, indexes
- ▶ `embed_compare.py` — embedder comparison on the core slice (experiments; needs the BGE-M3 vectors)
- ▶ `embed_ruri.py` — Ruri vectors for the whole corpus, HNSW index, `ef_search` check

**Answering**
- ▶ `generate.py` — retrieval + generation + all answer checks; also `--eval`, `--ablate`, the run log (`data/runs.jsonl`)
- ▶ `api.py` — FastAPI + SSE over the same pipeline (`uvicorn src.api:app`); the UI is in `web/`

**Retrieval modules**
- `env.py` — `.env` loading and the database DSN
- `hybrid.py` — search in Postgres: vector (Ruri or BGE-M3), BM25, RRF fusion, optional query expansion
- `rerank.py` — cross-encoder reranking on top of `hybrid.py`
- `query_expand.py` — the colloquial → statutory dictionary (`data/synonyms.json`); also feeds the term hint
- `query_llm.py` — LLM query expansion, the fallback to the dictionary; used only by the earlier BGE-M3 pipeline and `sweep.py` (model: `LLM_MODEL`)
- `search.py` — the first numpy vector search; kept as the baseline that `eval.py` and `embed_compare.py` reproduce
- `tokenize_ja.py` — Sudachi tokenisation for BM25 · `jnum.py` — Japanese numerals and article references

**Evaluation and experiments** (each one is listed with its result file in *Where the numbers come from*)
- ▶ `eval.py`, `eval_hybrid.py`, `sweep.py` — retrieval metrics on the golden set
- ▶ `compare.py` — diff two runs from the log, per question · ▶ `rescore_quotes.py` — re-score saved answers without regenerating
- ▶ `verify.py` — the judge experiments (`--calibrate`, `--structural`) · ▶ `make_review.py` — the manual annotation form
- ▶ `score_gate.py`, `margin_gate.py` — refusal thresholds on reranker scores
- ▶ `injection_test.py` — indirect prompt injection · ▶ `perturb_test.py` — width variants of the query · ▶ `check_ctx.py` — silent context truncation
- ▶ `expand_golden.py` — the rejected golden-set expansion · ▶ `toc.py` — browse a statute to verify golden answers

**Agent** (see *Extension*)
- ▶ `agent.py` — bare ReAct loop · ▶ `agent_graph.py` — the LangGraph version with memory · `mcp_server.py` — the tool over MCP (started by an MCP client)

**Other**
- `tracing.py` — OpenTelemetry spans for Phoenix, content masked by default
- ▶ `test_*.py` — deterministic tests, no database or LLM needed

## Stack

Python 3.14 · Postgres 17 + pgvector · Ollama (gemma4:e4b; BGE-M3 for the earlier pipeline) · sentence-transformers (ruri-v3-310m, bge-reranker-v2-m3) · SudachiPy · FastAPI + SSE · React + Vite · Phoenix + OpenTelemetry · httpx · numpy

---

## Extension: the RAG as a tool for an agent

The same retrieval, wrapped as a tool, for questions that span several institutions (「育休中に残業させられる？」 touches both parental leave and working hours). Three forms, built in this order to see what each layer adds.

**`agent.py` — a bare ReAct loop, no framework.** The model decides what to search for and when it has enough; the code decides when to stop. Budgets on iterations, tokens, context growth and wall time, plus a loop detector that measures progress rather than repetition: a search round that brings no new articles counts as a stall, and several in a row stop the loop. A forced final turn when a budget runs out, and a deterministic fallback when the model is unavailable. Tool errors go back to the model as messages it can act on, not as exceptions. The tool searches the core first and widens to the full corpus when the reranker score is low. `test_agent.py` tests the loop detector and the budget as plain code on hand-made inputs — the live 4B model refused or answered before it ever looped, and provoking it into a loop to test a set comparison would be the wrong way round.

**`agent_graph.py` — the same agent as a LangGraph state graph.** The `if`s inside the loop become named nodes and edges, which buys three things the loop did not have:
- **a thread**: follow-up questions in one conversation share state, and old tool output is compacted — conclusions and the article registry stay, raw text goes, so a second question still fits the context
- **human in the loop**: when retrieval is weak, the graph pauses with `interrupt()` and asks the user to clarify
- **long-term memory with confirmation**: facts about the user that change the legal answer (prefecture, form of employment, company size) are extracted from what the user said, checked against closed sets, verified to be stated rather than inferred, and saved only after a yes. Memory goes into the system prompt, so anything written there is an injection that survives restarts — hence the closed sets and the confirmation. Isolated per `user_id`; facts older than 90 days are marked for re-confirmation. `test_memory.py`

**`mcp_server.py` — the tool over MCP.** Registered in Claude Desktop, so a different model with a different loop can use the same retrieval. The trade-off is explicit: an MCP client runs its own loop and gets none of the budgets above, and the text the tool returns lands in a client that may hold other tools with real permissions — protection on that side belongs to the client.

```bash
uv run python src/agent.py "育休中に残業させられる？"
uv run python src/agent_graph.py              # interactive; /mem shows memory, /forget clears it
uv run python src/test_agent.py
uv run python src/test_memory.py
```

⚠️ The widen-to-full-corpus branch depends on the score threshold, which no longer separates hits from misses on Ruri (see *Open problems*): with the current threshold it rarely fires.

---

## License

Code: [MIT](LICENSE). Statute texts are fetched from e-Gov 法令検索 at build time and are not stored in this repository; Japanese statutes are not subject to copyright (著作権法 第13条).
