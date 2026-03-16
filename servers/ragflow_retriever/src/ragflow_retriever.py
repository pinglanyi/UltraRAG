import asyncio
import os
import sys
from typing import Any, Dict, List, Optional

import aiohttp

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ultrarag.server import UltraRAG_MCP_Server

app = UltraRAG_MCP_Server("ragflow_retriever")


class RAGFlowRetriever:
    def __init__(self, mcp_inst: UltraRAG_MCP_Server):
        mcp_inst.tool(
            self.ragflow_retriever_init,
            output="ragflow_base_url,ragflow_api_key,ragflow_dataset_ids,top_k,similarity_threshold,keywords_similarity_weight,max_doc_len->None",
        )
        mcp_inst.tool(
            self.ragflow_retriever_search,
            output="q_ls,top_k->ret_psg",
        )

        # Internal state
        self.base_url: str = ""
        self.api_key: str = ""
        self.dataset_ids: List[str] = []
        self.top_k: int = 5
        self.similarity_threshold: float = 0.2
        self.keywords_similarity_weight: float = 0.7
        self.max_doc_len: int = 2000

    async def ragflow_retriever_init(
        self,
        ragflow_base_url: str,
        ragflow_api_key: str,
        ragflow_dataset_ids: List[str],
        top_k: int = 5,
        similarity_threshold: float = 0.2,
        keywords_similarity_weight: float = 0.7,
        max_doc_len: int = 2000,
    ) -> None:
        """Initialize the RAGFlow retriever with connection and search parameters.

        Args:
            ragflow_base_url: Base URL of the RAGFlow service (e.g. http://host:port)
            ragflow_api_key: API key for RAGFlow authentication
            ragflow_dataset_ids: List of dataset (knowledge base) IDs to search in
            top_k: Number of top chunks to retrieve per query
            similarity_threshold: Minimum similarity score for a chunk to be included
            keywords_similarity_weight: Weight for keyword similarity vs vector similarity
            max_doc_len: Maximum character length of each returned passage
        """
        self.base_url = ragflow_base_url.rstrip("/")
        self.api_key = ragflow_api_key
        self.dataset_ids = ragflow_dataset_ids
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.keywords_similarity_weight = keywords_similarity_weight
        self.max_doc_len = max_doc_len
        app.logger.info(
            f"[ragflow_retriever] Initialized — base_url={self.base_url}, "
            f"datasets={self.dataset_ids}, top_k={self.top_k}"
        )

    async def ragflow_retriever_search(
        self,
        query_list: List[str],
        top_k: Optional[int] = None,
    ) -> Dict[str, List[List[str]]]:
        """Search RAGFlow knowledge bases for a list of queries.

        Args:
            query_list: List of query strings to retrieve for
            top_k: Override number of top passages per query (uses init value if None)

        Returns:
            Dictionary with 'ret_psg': a list of passage lists, one per query
        """
        effective_top_k = top_k if top_k is not None else self.top_k

        async def _search_one(query: str) -> List[str]:
            payload: Dict[str, Any] = {
                "question": query,
                "dataset_ids": self.dataset_ids,
                "top_k": effective_top_k,
                "similarity_threshold": self.similarity_threshold,
                "keyword_similarity_weight": self.keywords_similarity_weight,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            url = f"{self.base_url}/api/v1/retrieval"
            app.logger.debug(f"[ragflow_retriever] POST {url} query={query!r}")

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        app.logger.error(
                            f"[ragflow_retriever] Request failed status={resp.status} body={err_text}"
                        )
                        return []
                    data = await resp.json()

            # Log full response to help diagnose structure issues
            app.logger.warning(f"[ragflow_retriever] raw response keys={list(data.keys())} data={str(data)[:500]}")

            if data.get("code", -1) != 0:
                app.logger.error(
                    f"[ragflow_retriever] API error code={data.get('code')} "
                    f"message={data.get('message', '')}"
                )
                return []

            # RAGFlow v0.14+ uses "retcode"/"retmsg" instead of "code"/"message"
            # Handle both response conventions
            ret_data = data.get("data", {})
            if isinstance(ret_data, list):
                # some versions return data as a list of chunks directly
                chunks = ret_data
            else:
                chunks = ret_data.get("chunks", ret_data.get("docs", []))

            app.logger.warning(f"[ragflow_retriever] found {len(chunks)} chunks")

            passages: List[str] = []
            for chunk in chunks:
                # field may be "content", "content_with_weight", or "text"
                content: str = (
                    chunk.get("content")
                    or chunk.get("content_with_weight")
                    or chunk.get("text")
                    or ""
                )
                if self.max_doc_len and len(content) > self.max_doc_len:
                    content = content[: self.max_doc_len]
                passages.append(content)
            return passages

        results = await asyncio.gather(*[_search_one(q) for q in query_list])
        return {"ret_psg": list(results)}


retriever = RAGFlowRetriever(app)

if __name__ == "__main__":
    cfg = app.load_config(
        os.path.join(os.path.dirname(CURRENT_DIR), "parameter.yaml")
    )
    app.run(transport="stdio")