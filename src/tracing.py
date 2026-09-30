"""Tracing: OpenTelemetry spans with OpenInference attributes, viewed in Phoenix.

A trace is one request end to end; spans are its steps, nested: the agent
calls the model, then the tool; inside the tool, query expansion, then
retrieval... Each span has a start, an end and attributes (input, output,
tokens, timings). The run log (runs.jsonl) says what a configuration
RESULTED in; a trace says what HAPPENED inside one answer.

Instrumented by hand: auto-instrumentors hook the OpenAI SDK or LangChain,
and this project calls Ollama over plain httpx. Spans are OpenTelemetry
(vendor-neutral), attribute names follow OpenInference, so the backend is a
setting: Phoenix now, Langfuse later, same code.

Opt-in. Without TRACE=1 the tracer is OpenTelemetry's no-op and nothing is
sent — evaluation runs do not depend on a tracing server being up.

  docker run -d -p 6006:6006 --name phoenix arizephoenix/phoenix:latest
  $env:TRACE=1; uv run python src/agent.py "有給はいつから？"
  -> http://localhost:6006

Two things that break a trace silently, both handled here:
- OTel attribute values must be primitives or lists of primitives. Chat
  messages are flattened into llm.input_messages.{i}.message.role/content;
  a list of dicts is dropped, and the chat history never shows.
- The trace context does not cross threads by itself. The agent's tool runs
  its search in a worker thread (for the timeout); without carry() every
  search would become a separate orphan trace.
"""

import contextlib
import json
import os

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

ENABLED = os.environ.get("TRACE") == "1"
PROJECT = os.environ.get("TRACE_PROJECT", "labor-rag")
# HTTP on the UI port, set explicitly. register()'s default is gRPC on 4317:
# the first run pointed at it, only 6006 was published from Docker, and the
# exporter retried synchronously on every span — slowing the agent itself.
ENDPOINT = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
MAX_TEXT = 8000          # characters per attribute; whole articles are longer

_tracer = trace.get_tracer("labor-rag")   # no-op until setup() registers a provider


def setup(project=None):
    """Call once at an entry point. Does nothing unless TRACE=1."""
    global _tracer
    if not ENABLED:
        return
    from phoenix.otel import register
    # auto_instrument=False: there is nothing to auto-instrument (no OpenAI
    # SDK, no LangChain models), and hidden spans are worse than none.
    provider = register(project_name=project or PROJECT, endpoint=ENDPOINT,
                        auto_instrument=False)
    _tracer = provider.get_tracer("labor-rag")


def _text(v):
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= MAX_TEXT else s[:MAX_TEXT] + f"…[+{len(s) - MAX_TEXT}]"


def _ok(v):
    prim = (str, bool, int, float)
    return isinstance(v, prim) or (isinstance(v, (list, tuple)) and all(isinstance(x, prim) for x in v))


def set_attrs(sp, attrs):
    for k, v in (attrs or {}).items():
        if v is None:
            continue
        sp.set_attribute(k, v if _ok(v) else _text(v))


@contextlib.contextmanager
def span(name, kind, input=None, attrs=None):
    """A span of an OpenInference kind: AGENT, CHAIN, LLM, TOOL, RETRIEVER,
    EMBEDDING, RERANKER. Status is set explicitly — a manual span otherwise
    exports as UNSET, neither success nor failure."""
    with _tracer.start_as_current_span(name) as sp:
        sp.set_attribute("openinference.span.kind", kind)
        if input is not None:
            sp.set_attribute("input.value", _text(input))
        set_attrs(sp, attrs)
        try:
            yield sp
        except Exception as e:
            sp.record_exception(e)
            sp.set_status(Status(StatusCode.ERROR, str(e)[:200]))
            raise
        sp.set_status(Status(StatusCode.OK))


def llm_request(sp, model, messages, options=None, tools=None):
    """Flattened chat history, as OpenInference expects it."""
    attrs = {"llm.model_name": model, "llm.provider": "ollama",
             "llm.invocation_parameters": options or {}}
    for i, m in enumerate(messages):
        attrs[f"llm.input_messages.{i}.message.role"] = m.get("role", "")
        attrs[f"llm.input_messages.{i}.message.content"] = _text(m.get("content") or "")
    if tools:
        attrs["llm.tools.count"] = len(tools)
    set_attrs(sp, attrs)


def llm_response(sp, resp):
    """Tokens, output, and Ollama's own timings.

    load_duration is the one to watch: time spent loading the model into
    memory. Non-zero on a warm run means the model was evicted or reloaded
    (another model took the VRAM, or num_ctx differed from the last call).
    """
    msg = resp.get("message", {}) or {}
    p, c = resp.get("prompt_eval_count", 0), resp.get("eval_count", 0)
    attrs = {
        "llm.output_messages.0.message.role": msg.get("role", "assistant"),
        "llm.output_messages.0.message.content": _text(msg.get("content") or ""),
        "output.value": _text(msg.get("content") or ""),
        "llm.token_count.prompt": p,
        "llm.token_count.completion": c,
        "llm.token_count.total": p + c,
        "ollama.load_ms": round(resp.get("load_duration", 0) / 1e6),
        "ollama.prompt_eval_ms": round(resp.get("prompt_eval_duration", 0) / 1e6),
        "ollama.eval_ms": round(resp.get("eval_duration", 0) / 1e6),
        "ollama.total_ms": round(resp.get("total_duration", 0) / 1e6),
    }
    for j, call in enumerate(msg.get("tool_calls") or []):
        fn = call.get("function", {})
        pre = f"llm.output_messages.0.message.tool_calls.{j}.tool_call.function"
        attrs[f"{pre}.name"] = fn.get("name", "")
        attrs[f"{pre}.arguments"] = _text(fn.get("arguments") or {})
    set_attrs(sp, attrs)


def documents(sp, chunks, key="retrieval.documents"):
    """Retrieved articles as OpenInference documents: id, score, a preview."""
    attrs = {}
    for i, c in enumerate(chunks):
        pre = f"{key}.{i}.document"
        attrs[f"{pre}.id"] = f"{c.get('law_title', '')} {c.get('article_title', '')}"
        attrs[f"{pre}.score"] = float(c.get("rerank_score", 0.0) or 0.0)
        attrs[f"{pre}.content"] = _text(c.get("text", ""))[:500]
    set_attrs(sp, attrs)


def carry(fn):
    """Wrap fn to run in the CALLER's trace context — for thread pools.
    The context is captured now, at submit time, not when the worker runs."""
    ctx = otel_context.get_current()

    def run(*a, **kw):
        token = otel_context.attach(ctx)
        try:
            return fn(*a, **kw)
        finally:
            otel_context.detach(token)
    return run
