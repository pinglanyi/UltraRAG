"""Adaptive RAG RAGFlow API Service

Exposes a single POST /query endpoint:

  Request : {"query": "your question here"}
  Response: {"answer": "...", "thinking": "..."}

Three-stage quality-gated retrieval:
  Stage 1 – generate with top-5 passages, self-assess quality
  Stage 2 – generate with ext-5 passages (docs 6-10), self-assess quality
  Stage 3 – LLM rewrites query, re-retrieve, generate final answer

Configuration lives in serve_adaptive_rag_ragflow_config.yaml
(generation model, api_key, host/port, etc.).
RAGFlow connection parameters are read from
servers/ragflow_retriever/parameter.yaml.

Startup flow:
  1. ultrarag build examples/adaptive_rag_ragflow_api.yaml
     → generates examples/server/adaptive_rag_ragflow_api_server.yaml
       and     examples/parameter/adaptive_rag_ragflow_api_parameter.yaml
  2. All MCP server subprocesses are started ONCE and kept alive.
  3. Each POST /query reuses the running servers — no restart overhead.

Usage:
  python script/serve_adaptive_rag_ragflow.py
  python script/serve_adaptive_rag_ragflow.py --config script/serve_adaptive_rag_ragflow_config.yaml
"""

import argparse
import asyncio
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── path setup ────────────────────────────────────────────────────────────────
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

for _p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ultrarag.client as _client_mod
from ultrarag.mcp_logging import get_logger

# ── constants ────────────────────────────────────────────────────────────────
PIPELINE_FILE = "examples/adaptive_rag_ragflow_api.yaml"
PARAM_FILE    = "examples/parameter/adaptive_rag_ragflow_api_parameter.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_service_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app + persistent pipeline state
# ─────────────────────────────────────────────────────────────────────────────

fastapi_app = FastAPI(title="Adaptive RAG RAGFlow API")

_service_cfg: Dict[str, Any] = {}
_pipeline_client  = None   # fastmcp Client — started once, reused every request
_pipeline_context = None   # loaded once from config/param files
_request_lock     = asyncio.Lock()  # serialize requests (stdio MCP servers are sequential)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    thinking: Optional[str] = None


@fastapi_app.on_event("startup")
async def startup_event():
    global _pipeline_client, _pipeline_context

    _client_mod.logger = get_logger("Client", "info")

    from ultrarag.client import build as _build, create_mcp_client, load_pipeline_context

    print("[serve] Building pipeline configuration …")
    await _build(PIPELINE_FILE)
    print("[serve] Build complete.")

    # Load pipeline context once (server configs + base parameters)
    _pipeline_context = load_pipeline_context(PIPELINE_FILE, PARAM_FILE)

    # Start all MCP server subprocesses and keep them alive
    _pipeline_client = create_mcp_client(_pipeline_context["mcp_cfg"])
    await _pipeline_client.__aenter__()
    print("[serve] MCP servers started — ready to accept requests.")


@fastapi_app.on_event("shutdown")
async def shutdown_event():
    global _pipeline_client
    if _pipeline_client is not None:
        try:
            await _pipeline_client.__aexit__(None, None, None)
        except Exception:
            pass
        _pipeline_client = None
    print("[serve] MCP servers stopped.")


@fastapi_app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    if _pipeline_client is None or _pipeline_context is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet")

    try:
        # Build per-request override — injected directly into Data.local_vals,
        # no temp YAML file needed.
        override: Dict[str, Any] = {
            "query_input": {"queries": [req.query]},
        }
        gen_cfg = _service_cfg.get("generation", {})
        if gen_cfg:
            override["generation"] = copy.deepcopy(gen_cfg)

        from ultrarag.client import execute_pipeline

        # Serialize calls: stdio MCP servers handle one request at a time
        async with _request_lock:
            result = await execute_pipeline(
                _pipeline_client,
                _pipeline_context,
                return_all=True,
                override_params=override,
            )

        answer   = _extract_answer(result)
        thinking = _extract_thinking(result)
        return QueryResponse(answer=answer, thinking=thinking)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@fastapi_app.post("/query/stream")
async def query_stream_endpoint(req: QueryRequest):
    """Streaming variant — returns Server-Sent Events.

    Each event is a JSON line prefixed with ``data: ``.
    Event types:
      * ``{"type": "token",    "content": "<text>"}``  — incremental token
      * ``{"type": "step_end","name": "<step>", ...}``  — step completed
      * ``{"type": "sources", "data": [...]}``          — retrieved sources
      * ``{"type": "done",    "answer": "<text>"}``     — final answer (stream end)
      * ``{"type": "error",   "detail": "<msg>"}``      — error (stream end)
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    if _pipeline_client is None or _pipeline_context is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet")

    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    async def _callback(event: Dict[str, Any]) -> None:
        await queue.put(event)

    async def _run_pipeline():
        try:
            override: Dict[str, Any] = {"query_input": {"queries": [req.query]}}
            gen_cfg = _service_cfg.get("generation", {})
            if gen_cfg:
                override["generation"] = copy.deepcopy(gen_cfg)

            from ultrarag.client import execute_pipeline

            async with _request_lock:
                result = await execute_pipeline(
                    _pipeline_client,
                    _pipeline_context,
                    return_all=True,
                    stream_callback=_callback,
                    override_params=override,
                )

            answer = _extract_answer(result)
            await queue.put({"type": "done", "answer": answer})
        except Exception as exc:
            await queue.put({"type": "error", "detail": str(exc)})
        finally:
            await queue.put(_SENTINEL)

    async def _event_generator():
        asyncio.create_task(_run_pipeline())
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            line = "data: " + json.dumps(item, ensure_ascii=False) + "\n\n"
            yield line.encode()

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────────────────────
# Result extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _first_nonempty(val: Any) -> str:
    """Return the first non-empty string from a list, or the value itself."""
    if isinstance(val, list):
        for item in val:
            s = str(item).strip() if item is not None else ""
            if s:
                return s
        return ""
    return str(val).strip() if val is not None else ""


def _extract_answer(result: Dict[str, Any]) -> str:
    """Extract the final answer from the pipeline result.

    The last tool returns a JSON string as final_result, e.g.
    '{"pred_ls": ["answer text"]}'. Memory snapshot keys are prefixed
    with 'memory_', e.g. 'memory_pred_ls'.
    """
    if result is None:
        return ""

    # 1. Try final_result — the raw JSON output of the last pipeline step
    final = result.get("final_result")
    if final and isinstance(final, str):
        try:
            parsed = json.loads(final)
            for key in ("pred_ls", "ans_ls"):
                val = parsed.get(key)
                if val:
                    s = _first_nonempty(val)
                    if s:
                        return s
        except (json.JSONDecodeError, AttributeError):
            pass

    # 2. Walk snapshots — keys are stored as 'memory_<varname>'
    for snap in reversed(result.get("all_results", [])):
        mem = snap.get("memory", {})
        for key in ("memory_pred_ls", "memory_ans_ls", "pred_ls", "ans_ls"):
            val = mem.get(key)
            if val:
                s = _first_nonempty(val)
                if s:
                    return s
    return ""


def _extract_thinking(result: Dict[str, Any]) -> Optional[str]:
    """Pull intermediate reasoning from the last generate snapshot."""
    if result is None:
        return None
    for snap in reversed(result.get("all_results", [])):
        if "generate" in snap.get("step", ""):
            mem = snap.get("memory", {})
            for key in ("memory_ans_ls", "ans_ls"):
                val = mem.get(key)
                if val:
                    s = _first_nonempty(val)
                    if s:
                        return s
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Adaptive RAG RAGFlow API Service")
    p.add_argument(
        "--config",
        default="script/serve_adaptive_rag_ragflow_config.yaml",
        help="Path to service config YAML",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _service_cfg = load_service_config(args.config)

    host = _service_cfg.get("host", "0.0.0.0")
    port = _service_cfg.get("port", 8081)

    uvicorn.run(fastapi_app, host=host, port=port, reload=False)