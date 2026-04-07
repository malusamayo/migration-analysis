"""Task setup for browsecompplus.

A single HTTP MCP server is started once per run (via setup_servers) and shared
across all parallel workers, so the BM25 index is loaded only once.
"""

import subprocess
import time

_BROWSECOMP_PLUS_ROOT = "/mnt/datasets/BrowseComp-Plus"
_BM25_INDEX_PATH = f"{_BROWSECOMP_PLUS_ROOT}/indexes/bm25"
_SERVER_PORT = 18765
_MCP_URL = f"http://localhost:{_SERVER_PORT}/mcp"

PROMPT_TEMPLATE = """\
You are a deep research agent. Answer the question below by calling the \
search and get_document tools as many times as needed. Think step by step \
and cross-check findings across multiple sources before concluding.

When you have found the answer, write ONLY the final answer (a concise \
phrase or name, no explanation) to `answer.txt` in your workspace.

Question: {question}"""


def format_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def start_server() -> subprocess.Popen:
    proc = subprocess.Popen(
        [
            "uv", "--directory", _BROWSECOMP_PLUS_ROOT,
            "run", "python", "searcher/mcp_server.py",
            "--searcher-type", "bm25",
            "--index-path", _BM25_INDEX_PATH,
            "--transport", "streamable-http",
            "--port", str(_SERVER_PORT),
            "--get-document",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for the server to load the index and start listening
    time.sleep(15)
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def get_mcp_config() -> dict:
    return {
        "mcpServers": {
            "search-server": {
                "type": "url",
                "url": _MCP_URL,
            }
        }
    }
