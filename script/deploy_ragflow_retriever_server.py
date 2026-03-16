"""Standalone HTTP service that wraps the RAGFlow retriever.

Usage:
    python script/deploy_ragflow_retriever_server.py \
        --config_path script/deploy_ragflow_retriever_config.json \
        --host 0.0.0.0 \
        --port 64502

The service exposes a single POST /search endpoint compatible with the
UltraRAG retriever_deploy_search tool, so it can be used as a drop-in
replacement for the standard vector-based retriever service.
"""

import os
import sys
import argparse
from typing import Any, Dict, List, Optional

import orjson
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RAGFLOW_SRC = os.path.join(PROJECT_ROOT, "servers", "ragflow_retriever", "src")
if RAGFLOW_SRC not in sys.path:
    sys.path.insert(0, RAGFLOW_SRC)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from servers.ragflow_retriever.src.ragflow_retriever import RAGFlowRetriever, app


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise RuntimeError(f"Config file does not exist: {path}")
    with open(path, "rb") as f:
        return orjson.loads(f.read())


# ---------------------------------------------------------------------------
# FastAPI models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query_list: List[str]
    top_k: int = 5


class SearchResponse(BaseModel):
    ret_psg: List[List[str]]


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

fastapi_app = FastAPI(title="UltraRAG RAGFlow Retriever Service")

retriever: Optional[RAGFlowRetriever] = None
retriever_cfg: Optional[Dict[str, Any]] = None


@fastapi_app.on_event("startup")
async def startup_event():
    global retriever, retriever_cfg

    assert retriever_cfg is not None, "retriever_cfg is not set"

    app.logger.info(f"[ragflow_retriever_server] Using configuration: {retriever_cfg}")

    retriever = RAGFlowRetriever(app)

    await retriever.ragflow_retriever_init(
        ragflow_base_url=retriever_cfg["ragflow_base_url"],
        ragflow_api_key=retriever_cfg["ragflow_api_key"],
        ragflow_dataset_ids=retriever_cfg["ragflow_dataset_ids"],
        top_k=retriever_cfg.get("top_k", 5),
        similarity_threshold=retriever_cfg.get("similarity_threshold", 0.2),
        keywords_similarity_weight=retriever_cfg.get("keywords_similarity_weight", 0.7),
        max_doc_len=retriever_cfg.get("max_doc_len", 2000),
    )

    app.logger.info("[ragflow_retriever_server] ragflow_retriever_init completed")


@fastapi_app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    global retriever

    assert retriever is not None, "Retriever is not initialized"

    result = await retriever.ragflow_retriever_search(
        query_list=req.query_list,
        top_k=req.top_k,
    )
    return SearchResponse(ret_psg=result["ret_psg"])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Standalone RAGFlow Retriever HTTP Service")
    parser.add_argument(
        "--config_path",
        type=str,
        default="script/deploy_ragflow_retriever_config.json",
        help="Path to deploy_ragflow_retriever_config.json",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the HTTP server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=64502,
        help="Port to bind the HTTP server",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    retriever_cfg = load_config(args.config_path)

    uvicorn.run(
        fastapi_app,
        host=args.host,
        port=args.port,
        reload=False,
    )