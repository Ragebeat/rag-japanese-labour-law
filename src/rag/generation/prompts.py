"""What the generator is asked, and how: the model, the system prompts, the
generation options. One copy, imported by the CLI, the API and the agent."""

import os

MODEL = os.environ.get("GEN_MODEL", "gemma4:e4b")
URL = "http://localhost:11434/api/chat"

REFUSAL = "該当する規定が見つかりません"

# The format examples contain NO real articles or terms from the golden set.
# An example in the prompt becomes a template.
#
# The 「判定:」 line was added after the first run: in prose the model
# expressed refusal in ten different phrasings and never once cleanly.
# Same technique as the [N] marker: have the model emit a token we control
# instead of digging meaning out of free text.
SYSTEM_BASE = f"""あなたは日本の労働法令の検索結果を要約するアシスタントです。

【出力形式】
1行目に必ず「判定: 回答可能」または「判定: 回答不可」とだけ書く。
2行目以降に本文を書く。

【厳守】
1. 提供された条文だけを根拠にする。自分の知識で補わない
2. 根拠にした条文を必ず [1] のような番号で示す。番号の直後に法令名と
   条番号も書く。形式:「[N] ◯◯法 第N条によると…」
   条番号は条文の見出しの表記のまま書き写し、算用数字に直さない。
   本文中で条番号に触れる場合も同じ
3. 質問に直接答える条文が提供されていない場合は「判定: 回答不可」とし、
   本文には「{REFUSAL}」とだけ書く。
   関連しそうな条文を列挙したり、前置きを書いたりしない
4. 数字・日数・金額は条文にある通りに引用する。推測しない
5. 本文は3〜5文で簡潔に"""

# The verbatim-quote variant. Five answers in earlier runs had the right
# article and a wrong detail; the one class this makes CHECKABLE is a quote
# that is not really a quote (two exceptions spliced into one, text put under
# another article's number). The check sees only what is inside 「」:
# paragraph numbers, paraphrase and omissions stay invisible to it.
# Rule 7 is a separate, cheaper net: a number outside quotes is unverified.
SYSTEM_QUOTES = SYSTEM_BASE + """
6. 答えの核心（数字・期間・要件・例外）は、条文から一字一句そのまま「」で引用し、
   直後にその条文の番号を付ける。形式:「引用文」[N]
   「」の中は言い換えない・要約しない・別々の文をつなぎ合わせない。
   別の文を引用するときは、別の「」にする
7. 数字・日数・期間は「」の引用の中にだけ書く
8. 条文に（第N項）の印がある場合、根拠の項まで書く。形式:「[N] ◯◯法 第N条第N項」
   項の番号は必ず（第N項）の印から取る。段落を数えて推測しない"""

# The default since 29.09 (on BGE-M3), measured on golden against the base
# prompt: claim coverage 0.09 -> 0.68, answers with numbers outside quotes
# 22 -> 11, over-refusal 0 -> 0, refusal on retrieval miss 0.17 -> 0.17 (the
# same five blind answers). One suspected regression: #26 dropped the key
# article (労契法 第十条): one case, not separable from run-to-run noise.
# SYSTEM_BASE stays: the source-removal ablation is paired with a base-prompt report.
SYSTEM = SYSTEM_QUOTES

# Generation options: ONE copy, imported by api.py. The first num_ctx fix
# patched generate() and missed api.py's two own requests: three
# copies of the same dict, one of them updated.
#
# num_ctx is explicit on purpose. check_ctx.py: prompts run up to ~7 000
# tokens and nothing was truncated, but only because the Ollama app on this
# machine is set to 32k. That setting lives outside the repo; on a stock
# install below 24 GiB VRAM the default is 4096, and longer prompts would
# lose their start silently. Same value as agent/loop.py: a different window
# reloads the model on every switch between the two.
GEN_OPTIONS = {"temperature": 0, "num_ctx": 32_768}
