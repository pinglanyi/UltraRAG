from typing import Dict, List

from ultrarag.server import UltraRAG_MCP_Server

app = UltraRAG_MCP_Server("query_input")


@app.tool(output="queries->q_ls")
def query_input_init(queries: List[str]) -> Dict[str, List[str]]:
    """Inject query strings directly from parameter config (no file I/O).

    The parameter key 'queries' is flat in parameter.yaml so the build step
    generates a simple PARAM_FILE entry: query_input.queries = [...].
    The serve script patches that single key directly — no nested lookups needed.

    Args:
        queries: List of query strings (read from parameter config via $queries).

    Returns:
        Dictionary with 'q_ls' containing the query strings for the pipeline.
    """
    app.logger.info(f"[query_input] q_ls={queries}")
    return {"q_ls": queries}


if __name__ == "__main__":
    app.run(transport="stdio")