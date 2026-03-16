"""Search-o1 RAGFlow API Service

Exposes a single POST /query endpoint:

  Request : {"query": "your question here"}
  Response: {"answer": "...", "thinking": "..."}

Configuration lives in serve_search_o1_ragflow_config.yaml
(generation model, api_key, host/port, etc.).
RAGFlow connection parameters are read from
servers/ragflow_retriever/parameter.yaml.

Startup flow:
  1. ultrarag build examples/search_o1_ragflow_api.yaml
     → generates examples/server/search_o1_ragflow_api_server.yaml
       and     examples/parameter/search_o1_ragflow_api_parameter.yaml
  2. Patch the parameter file from config
  3. Service starts; each POST /query runs the pipeline for that query

Usage:
  python script/serve_search_o1_ragflow.py
  python script/serve_search_o1_ragflow.py --config script/serve_search_o1_ragflow_config.yaml
"""

import asyncio
import json
import os
import sys
import tempfile
import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
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
PIPELINE_FILE = "examples/search_o1_ragflow_api.yaml"
PARAM_FILE    = "examples/parameter/search_o1_ragflow_api_parameter.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_service_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def patch_pipeline_param(service_cfg: Dict[str, Any], query_file: str) -> str:
    """Write a per-request parameter YAML and return its path."""
    param_path = Path(PARAM_FILE)
    if not param_path.exists():
        raise RuntimeError(
            f"Pipeline parameter file not found: {PARAM_FILE}\n"
            "Make sure startup_event() has completed successfully."
        )
    with open(param_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}

    # benchmark – point at the single-query temp file
    params.setdefault("benchmark", {})
    params["benchmark"]["path"]  = query_file
    params["benchmark"]["key_map"] = {"q_ls": "question", "gt_ls": "answer"}
    params["benchmark"]["limit"] = 1

    # generation – override from service config
    gen_cfg = service_cfg.get("generation", {})
    if gen_cfg:
        params.setdefault("generation", {})
        params["generation"].update(copy.deepcopy(gen_cfg))

    # write to a temp file so concurrent requests don't stomp each other
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.safe_dump(params, tmp, allow_unicode=True, sort_keys=False)
    tmp.close()
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

fastapi_app = FastAPI(title="Search-o1 RAGFlow API")
_service_cfg: Dict[str, Any] = {}


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    thinking: Optional[str] = None


@fastapi_app.on_event("startup")
async def startup_event():
    global _service_cfg

    _client_mod.logger = get_logger("Client", "info")

    # Step 1 – build pipeline config files (once)
    from ultrarag.client import build as _build
    print("[serve] Building pipeline configuration …")
    await _build(PIPELINE_FILE)
    print("[serve] Build complete.")


@fastapi_app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    # Write the single query to a temp JSONL file
    tmp_query = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    json.dump({"question": req.query, "answer": ""}, tmp_query)
    tmp_query.write("\n")
    tmp_query.close()

    tmp_param = None
    try:
        # Per-request parameter file (thread-safe)
        tmp_param = patch_pipeline_param(_service_cfg, tmp_query.name)

        # Run the pipeline
        from ultrarag.client import run as _run
        result = await _run(PIPELINE_FILE, tmp_param, return_all=True)

        # Extract the final boxed answer from pipeline output
        answer = _extract_answer(result)
        thinking = _extract_thinking(result)
        return QueryResponse(answer=answer, thinking=thinking)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        for p in [tmp_query.name, tmp_param]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# Result extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_answer(result: Dict[str, Any]) -> str:
    """Pull the final pred_ls answer from pipeline result."""
    if result is None:
        return ""

    final = result.get("final_result")
    if final and isinstance(final, str):
        return final.strip()

    # Walk snapshots to find the last pred_ls or ans_ls entry
    for snap in reversed(result.get("all_results", [])):
        mem = snap.get("memory", {})
        for key in ("pred_ls", "ans_ls"):
            val = mem.get(key)
            if val:
                if isinstance(val, list) and val:
                    return str(val[0]).strip()
                return str(val).strip()
    return ""


def _extract_thinking(result: Dict[str, Any]) -> Optional[str]:
    """Pull intermediate reasoning from the last ans_ls snapshot."""
    if result is None:
        return None
    for snap in reversed(result.get("all_results", [])):
        step = snap.get("step", "")
        if "generate" in step:
            mem = snap.get("memory", {})
            val = mem.get("ans_ls")
            if val:
                text = val[0] if isinstance(val, list) else val
                return str(text).strip()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Search-o1 RAGFlow API Service")
    p.add_argument(
        "--config",
        default="script/serve_search_o1_ragflow_config.yaml",
        help="Path to service config YAML",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _service_cfg = load_service_config(args.config)

    host = _service_cfg.get("host", "0.0.0.0")
    port = _service_cfg.get("port", 8080)

    uvicorn.run(fastapi_app, host=host, port=port, reload=False)