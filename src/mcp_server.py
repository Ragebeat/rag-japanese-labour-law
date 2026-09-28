"""labor_laws_rag as an MCP server. Phase 5, session 3.

MCP (Model Context Protocol) is a standard way to expose tools to any LLM
client: Claude Desktop, Claude Code, Cursor, or our own agent. The tool is
written once; every client gets the same contract.

What travels with the tool and what does not — the main point of this file:

  TRAVELS (lives inside Tool): the measured threshold, ok/weak/error per
  sub-query, weak candidates without text, widening to 259 documents, the
  per-call timeout, the warm-up.

  STAYS BEHIND (lives in the agent loop): the budget, the loop detector, the
  nudge to search, the reframe nudge, the final turn. A client like Claude
  Desktop runs ITS OWN loop and gets none of these.

This is session 1's rule seen from the other side: whatever was put into the
tool as code became portable; whatever lived in the loop did not.

  uv run mcp dev src/mcp_server.py     # MCP Inspector in the browser
  uv run mcp run src/mcp_server.py     # stdio, for Claude Desktop

SDK: mcp 2.x (MCPServer). The v1 import path mcp.server.fastmcp is gone.
"""

import builtins
import pathlib
import sys
import threading
from typing import Annotated

from pydantic import Field

from mcp.server import MCPServer

SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from agent import TOOL, Tool  # noqa: E402

# ⚠️ stdio transport: STDOUT IS THE PROTOCOL CHANNEL. Any print() to stdout
# lands in the JSON-RPC stream and breaks the client with an error that says
# nothing about print(). The pipeline does print: query_llm.py writes LLM
# errors to stdout.
#
# The first version wrapped our code in contextlib.redirect_stdout(stderr).
# That swaps sys.stdout for the WHOLE PROCESS, not one thread: while the
# background thread loaded models for 45 s, the SDK's own reply to the
# handshake went to stderr too, and the Inspector showed "Failed" with the
# initialize result sitting in the server console.
#
# So sys.stdout is never touched. Instead `print` is rebound inside our own
# modules only — a module looks `print` up in its globals before builtins.


def _print_to_stderr(*args, **kwargs):
    if kwargs.get("file") in (None, sys.stdout):
        kwargs["file"] = sys.stderr
    builtins.print(*args, **kwargs)


for _m in list(sys.modules.values()):
    _f = getattr(_m, "__file__", None)
    if _f and pathlib.Path(_f).resolve().parent == SRC:
        _m.print = _print_to_stderr

mcp = MCPServer("labor-laws-rag")

# The reranker and the embedder take ~45 s to load. Loading before the server
# starts would stall the client's handshake; loading on the first call would
# make that call look hung. So: start loading in the background at once, and
# let a call wait only for whatever is left.
_tool = None
_ready = threading.Event()
_load_error = None


def _load():
    global _tool, _load_error
    try:
        _tool = Tool()
    except Exception as e:                          # noqa: BLE001
        _load_error = f"{type(e).__name__}: {e}"
    finally:
        _ready.set()


threading.Thread(target=_load, daemon=True).start()


def _all_error(queries, message):
    return {"results": [{"query": q, "status": "error", "message": message,
                         "next": []} for q in queries]}


def labor_laws_rag(
    queries: Annotated[list[str], Field(
        min_length=1, max_length=4,
        description="One Japanese query per sub-question, one topic each. "
                    "Colloquial phrasing is fine.")],
    k: Annotated[int, Field(
        ge=1, le=10,
        description="How many articles per sub-question. 5 unless the "
                    "question genuinely spans several institutions.")] = 5,
) -> dict:
    if not _ready.wait(timeout=120):
        return _all_error(queries, "Search engine is still loading.")
    if _load_error:
        return _all_error(queries, f"Search engine failed to load: {_load_error}")
    payload, _keys, _tops, _exps = _tool(queries, k)
    # keys, tops and expansions are for the agent loop and the trace. An MCP
    # client gets exactly what the model in agent.py sees — nothing more.
    return payload


# One source of truth for the description: the same text agent.py sends to
# gemma. The decorator reads __doc__ at registration time.
labor_laws_rag.__doc__ = TOOL["function"]["description"]
mcp.tool()(labor_laws_rag)
